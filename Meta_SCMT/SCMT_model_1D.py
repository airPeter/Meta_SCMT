import numpy as np
import torch
import torch.nn as nn
from .utils import Model
from .sputil_1D import gen_coo_sparse, gen_dis_CK_input, gen_input_neff
from scipy import special

class Metalayer(torch.nn.Module):
    def __init__(self, GP, COUPLING, N, ln, lc, lk, le):
        '''

        '''
        super(Metalayer, self).__init__()
        self.COUPLING = COUPLING
        self.h_paras = torch.nn.Parameter(torch.empty((N,), dtype = torch.float))
        self.hs = None
        self.neffs = None
        self.GP = GP
        self.h_min = GP.h_min
        self.h_max = GP.h_max
        self.dh = GP.dh
        self.wh = GP.wh
        self.k = GP.k
        self.neffnn = gen_neff(GP.modes, ln).requires_grad_(requires_grad=True)
        self.genc = gen_C(GP.modes, lc, N, GP.Knn)
        self.genk = gen_K(GP.modes, lk, N, GP.Knn)
        self.genu0 = gen_U0(GP.modes, ln, le, GP.res, N, GP.n0, GP.C_EPSILON, GP.dx, GP.Knn)
        self.genen = gen_En(GP.modes, GP.res, N, GP.n0, GP.C_EPSILON, GP.dx, GP.Knn)
        self.sig = torch.nn.Sigmoid()
        self.gen_neff_input = gen_input_neff(N, GP.Knn)
        dis = torch.tensor(gen_dis_CK_input(N, GP.Knn), dtype = torch.float, requires_grad = False)
        self.register_buffer('dis', dis)
    def forward(self, E0):
        '''
        size of E0: (N + 2 * (Knnc + 1)) * period_resolution
        '''
        self.hs = self.sig(self.h_paras) * (self.h_max - self.h_min) + self.h_min
        self.neffs = self.neffnn(self.hs.view(-1, 1))
        with torch.set_grad_enabled(False):
            Eys, U0 = self.genu0(self.hs, E0)
        if not self.COUPLING:
            P = torch.exp(self.neffs.view(-1,) * self.k * self.wh * 1j)
            Uz = P * U0 #shape [N*modes,]
        else:
            with torch.set_grad_enabled(False):
                neff_input = self.gen_neff_input(self.hs)
                CK_input = torch.cat([neff_input, self.dis],dim = -1)
                CK_input = CK_input.view(-1,3)
                C = self.genc(CK_input)
                K = self.genk(CK_input)
                C_inv = torch.inverse(C)
            B = torch.diag(self.neffs.view(-1,) * self.k)
            Eig_M = C_inv @ (B @ C + K)
            A = Eig_M * self.wh * 1j
            P = torch.matrix_exp(A) #waveguide propagator
            Uz = P @ U0
        with torch.set_grad_enabled(False):
            hs_no_grad = self.hs
            neffs_no_grad = self.neffs
        En = self.genen(hs_no_grad, Uz, neffs_no_grad, Eys) #near field
        return En
    
    def reset(self, path):
        #nn.init_normal_(self.phase, 0, 0.02)
        torch.nn.init.constant_(self.h_paras, val = 0.0)
        self.neffnn.reset(path)
        self.genc.reset(path)
        self.genk.reset(path)
        self.genu0.reset(path)
    
class SCMT_Model(nn.Module):
    def __init__(self, prop_dis, GP, COUPLING, N, layer_neff, layer_C, layer_K, layer_E):
        super(SCMT_Model, self).__init__()

        self.prop = prop_dis
        self.matalayer1 = Metalayer(GP, COUPLING, N, layer_neff, layer_C, layer_K, layer_E)
        self.freelayer1 = freespace_layer(GP.k, self.prop, N, GP.Knn, GP.res, GP.dx)
    def forward(self, E0):
        En = self.matalayer1(E0)
        Ef = self.freelayer1(En)
        If = torch.abs(Ef)**2
        #If = If/If.max()
        return If
    def reset(self):
        self.matalayer1.reset()

class gen_neff(nn.Module):
    def __init__(self, modes, layers):
        super(gen_neff, self).__init__()    
        self.model = Model(in_size = 1, out_size = modes, layers = layers)
    def forward(self, hs):
        '''
        input: 
            hs: array of waveguide widths [N,]
        output:
            neffs of each mode. shape: [N, number of modes for each waveguide.]
        '''
        return self.model(hs)
    def reset(self, path):
        model_state = torch.load(path + "fitting_neffs_state_dict")
        self.model.load_state_dict(model_state) 


class gen_U0(nn.Module):
    def __init__(self, modes, ln, le, res, N, n0, C_EPSILON, dx, Knn):
        super(gen_U0, self).__init__()
        self.N = N
        self.n0 = n0
        self.C_EPSILON = C_EPSILON
        self.dx = dx
        self.Knn = Knn
        self.modes = modes
        self.res = res
        self.neffnn = gen_neff(modes, ln).requires_grad_(requires_grad=False)
        enn_out_size = modes * 2 * (Knn + 1) * res
        self.Ey_size = 2 * (Knn + 1) * res
        self.enn = Model(1, enn_out_size, layers= le, nodes = 128).requires_grad_(requires_grad=False)
        self.register_buffer('E0_slice', torch.zeros((N, 1, self.Ey_size), dtype= torch.complex64))
    def forward(self, hs, E0):
        '''
        input:
            hs: array of waveguide widths [N,]
            E0: input field [(N + Ey_size - 1) * period_resolution]
        output:
            neff: refractive index of each mode. shape [N, modes]
            T: modes amplitude coupled in. shape [N, number of modes]
        '''
        neff = self.neffnn(hs.view(-1, 1))
        for i in range(6):
            self.E0_slice[:,0, i * self.res: (i + 1) * self.res] = \
                E0[self.res//2 + i * self.res: self.res//2 + (self.N + i) * self.res].view(self.N, self.res)
            Ey = self.enn(hs.view(-1, 1))
            Ey = Ey.view(self.N, self.modes, self.Ey_size)
            E_sum = torch.sum(Ey * self.E0_slice, dim= -1, keepdim= False) # shape: [N, modes]
            eta = (neff * self.n0) / (neff + self.n0) #shape [N, modes]
            T = 2 * self.C_EPSILON * eta * E_sum * self.dx
            T = T.view(-1,)
        return Ey, T
    def reset(self, path):
        model_state = torch.load(path + "fitting_E_state_dict")
        self.enn.load_state_dict(model_state)
        self.neffnn.reset(path)

class gen_En(nn.Module):
    def __init__(self, modes, res, N, n0, C_EPSILON, dx, Knn):
        super(gen_En, self).__init__()
        self.N = N
        self.n0 = n0
        self.C_EPSILON = C_EPSILON
        self.dx = dx
        self.Knn = Knn
        self.modes = modes
        self.res = res
        self.Ey_size = 2 * (Knn + 1) * res
        self.total_size = (N + 2 * (Knn + 1)) * res
        self.register_buffer('En', torch.zeros((self.total_size,), dtype= torch.complex64))
    def forward(self, hs, U, neff, Ey):
        '''
            neff: shape [N, modes]
            Ey: shape [N, modes, fields]
        '''
        hs = hs.view(-1, 1)
        eta = (neff * self.n0) / (neff + self.n0) #shape: [N, modes]
        Ey = eta.view(-1, self.modes, 1) * Ey * U.view(-1, self.modes, 1)
        self.En = torch.zeros((self.total_size,), dtype= torch.complex64).to(hs.device)
        for i in range(self.N):
            for m in range(self.modes):
                temp_Ey = Ey[i, m]
                center = i * self.res + self.res//2 + (self.Knn + 1) * self.res
                center = int(center)
                radius = int((self.Knn + 1) * self.res)
                self.En[center - radius: center + radius] += temp_Ey
        return self.En
        
class gen_C(nn.Module):
    def __init__(self, modes, lc, N, Knn):
        super(gen_C, self).__init__()
        self.channels = modes**2
        self.cnn = Model(3, self.channels, layers= lc, nodes = 64).requires_grad_(requires_grad=False)
        coo = torch.tensor(gen_coo_sparse(N, Knn),dtype=int, requires_grad = False)
        self.register_buffer('coo', coo)
        self.N = N
        self.modes = modes 
        
    def forward(self, CK_inputs):
        '''
        input:
            CK_inputs: the cnn input is (hi, hj, dis), output is cij for all the channels.
            the CK_inputs includes all the possiable couplings for N waveguides. shape [N, 2 * (Knn + 1), 3]
        '''
        C_stripped = self.cnn(CK_inputs.view(-1, 3))
        C_sparses = []
        for mi in range(self.modes):
            for mj in range(self.modes):
                ch = mi * self.modes + mj
                val = C_stripped[:,ch]
                coo = self.coo * self.modes + mj
                C_sparse = torch.sparse_coo_tensor(coo, val, (self.modes * self.N, self.modes * self.N), requires_grad= False)
                C_sparses.append(C_sparse)
        for C_temp in C_sparses[:-1]:
            C_sparse += C_temp
        C_sparse = C_sparse.coalesce()
        C_dense = C_sparse.to_dense()
        return C_dense
    def reset(self, path):
        model_state = torch.load(path + "fitting_C_state_dict")
        self.cnn.load_state_dict(model_state)

class gen_K(nn.Module):
    def __init__(self, modes, lk, N, Knn):
        super(gen_K, self).__init__()
        self.channels = modes**2
        self.knn = Model(3, self.channels, layers= lk, nodes = 64).requires_grad_(requires_grad=False)
        coo = torch.tensor(gen_coo_sparse(N, Knn),dtype=int, requires_grad = False)
        self.register_buffer('coo', coo)
        self.N = N
        self.modes = modes 

    def forward(self, CK_inputs):
        '''
        input:
            CK_inputs: the cnn input is (hi, hj, dis), output is cij for all the channels.
            the CK_inputs includes all the possiable couplings for N waveguides. shape [N, 2 * (Knn + 1), 3]
        '''
        K_stripped = self.knn(CK_inputs.view(-1, 3))
        K_sparses = []
        for mi in range(self.modes):
            for mj in range(self.modes):
                ch = mi * self.modes + mj
                val = K_stripped[:,ch]
                coo = self.coo * self.modes + mj
                K_sparse = torch.sparse_coo_tensor(coo, val, (self.modes * self.N, self.modes * self.N), requires_grad= False)
                K_sparses.append(K_sparse)
        for K_temp in K_sparses[:-1]:
            K_sparse += K_temp
        K_sparse = K_sparse.coalesce()
        K_dense = K_sparse.to_dense()
        return K_dense
    def reset(self, path):
        model_state = torch.load(path + "fitting_K_state_dict")
        self.knn.load_state_dict(model_state)
        
class freespace_layer(nn.Module):
    def __init__(self, k, prop, N, Knn, res, dx):
        super(freespace_layer, self).__init__()
        G = torch.tensor(gen_G(k, prop, N, Knn, res, dx))
        self.register_buffer('G', G)
    def forward(self, En):
        Ef = self.G @ En
        return Ef

def gen_G(k, prop, N, Knn, res, dx):
    total_size = (N + 2 * (Knn + 1)) * res
    x = np.arange(total_size) * dx
    dx = np.reshape(x, (1,-1)) - np.reshape(x, (-1, 1))
    r = np.sqrt(prop**2 + dx**2)
    v = 1
    G = -1j * k / 4 * special.hankel1(v, k * r) * prop / r * dx
    return G