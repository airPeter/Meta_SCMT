import numpy as np
import torch
import torch.nn as nn
from .utils import Model, fourier_conv
from .sputil_2D import gen_coo_sparse, gen_dis_CK_input, gen_input_hs, gen_Cinv_rows

class Metalayer(torch.nn.Module):
    def __init__(self, Euler_steps, devs, GP, COUPLING, APPROX, Ni, k_row, N, ln, lc, lk, le):
        '''

        '''
        super(Metalayer, self).__init__()
        self.N = N
        self.Euler_steps = Euler_steps
        self.devs = devs
        self.num_devs = len(devs)
        if N%self.num_devs != 0:
            raise ValueError(" num_devs should be divided by N")
        gen_cinv_modules = []
        rows_per_dev = N**2//self.num_devs
        for i in range(self.num_devs):
            row_start = i * rows_per_dev
            row_end = (i + 1) * rows_per_dev
            gen_cinv_modules.append(gen_Cinv_rows(N, Ni, k_row, row_start, row_end).to(self.devs[i]))
        self.gen_cinv = nn.ModuleList(gen_cinv_modules)
        
        self.COUPLING = COUPLING
        self.APPROX = APPROX
        self.h_paras = torch.nn.Parameter(torch.empty((N * N,), dtype = torch.float))
        self.hs = None
        self.neffs = None
        self.GP = GP
        self.h_min = GP.h_min
        self.h_max = GP.h_max
        self.dh = GP.dh
        self.wh = GP.wh
        self.k = GP.k
        self.neffnn = gen_neff(GP.modes, ln).to(self.devs[0])
        self.genc = gen_C(GP.modes, lc, N).to(self.devs[0])
        self.genk = gen_K(GP.modes, lk, N).to(self.devs[0])
        self.genu0 = gen_U0(GP.modes, ln, le, GP.out_res, N, GP.n0, GP.C_EPSILON, GP.period, GP.Knn).to(self.devs[0])
        self.genen = gen_En(GP.modes, GP.out_res, N, GP.n0, GP.C_EPSILON, GP.Knn).to(self.devs[0])
        self.sig = torch.nn.Sigmoid()
        if GP.Knn != 2:
            raise Exception("Knn = 2 is hardcode in sputil_2D module. So only Knn = 2 is supported.")
        self.gen_hs_input = gen_input_hs(N).to(self.devs[0])
        dis = torch.tensor(gen_dis_CK_input(N), dtype = torch.float, requires_grad = False).to(self.devs[0])
        self.register_buffer('dis', dis)
        coo = torch.tensor(gen_coo_sparse(N),dtype=int, requires_grad = False).to(self.devs[0])
        self.register_buffer('coo', coo)
    def forward(self, E0):
        '''
        size of E0: (N + 2 * (Knnc + 1)) * self.out_res
        '''
        self.hs = self.sig(self.h_paras.to(self.devs[0])) * (self.h_max - self.h_min) + self.h_min
        self.neffs = self.neffnn(self.hs.view(-1, 1))
        with torch.set_grad_enabled(False):
            Eys, U0 = self.genu0(self.hs, E0)
        if not self.COUPLING:
            P = torch.exp(self.neffs.view(-1,) * self.k * self.wh * 1j)
            Uz = P * U0 #shape [N*modes,]
        else:
            with torch.set_grad_enabled(False):
                hs_input = self.gen_hs_input(self.hs)
                CK_input = torch.cat([hs_input, self.dis],dim = -1)
                CK_input = CK_input.view(-1,4)                
                C_stripped = self.genc(CK_input)
                K_stripped = self.genk(CK_input)
                K_sparse = torch.sparse_coo_tensor(self.coo, K_stripped.view(-1,), (self.N**2, self.N**2))
                K_sparse = K_sparse.coalesce()

            BC_stripped = self.neffs.view(-1, 1) * C_stripped.view(-1, 13)
            BC_sparse = self.k * torch.sparse_coo_tensor(self.coo, BC_stripped.view(-1,), (self.N**2, self.N**2))
            if self.APPROX == 0:
                C_sparse = torch.sparse_coo_tensor(self.coo, C_stripped.view(-1,), (self.N**2, self.N**2))
                C_sparse = C_sparse.coalesce()
                C_dense = C_sparse.to_dense()
                C_inv = torch.inverse(C_dense)
                K_dense = K_sparse.to_dense()
                A = C_inv @ (BC_sparse.to_dense() + K_dense)
                P = torch.matrix_exp(1j * A * self.wh) #waveguide propagator
                Uz = P @ U0
            else:
                C_inv_sparse_list = []
                for i in range(self.num_devs):
                    C_stripped = C_stripped.view(-1,)
                    C_inv_sparse = self.gen_cinv[i](C_stripped.to(self.devs[i]), self.coo.to(self.devs[i]))
                    C_inv_sparse = C_inv_sparse.coalesce()
                    C_inv_sparse_list.append(C_inv_sparse)
                A_list = []
                for i in range(self.num_devs):
                    C_inv_sparse_i = C_inv_sparse_list[i]
                    BC_sparse_i = BC_sparse.to(self.devs[i])
                    K_sparse_i = K_sparse.to(self.devs[i])
                    A_i = torch.sparse.mm(C_inv_sparse_i, (BC_sparse_i + K_sparse_i))               
                    A_i = A_i.coalesce()
                    A_list.append(A_i)
                Uz = gen_Euler_with_backward_devs.apply(self.Euler_steps, self.wh, self.devs, U0.view(-1, 1), *A_list)
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

class gen_neff(nn.Module):
    def __init__(self, modes, layers):
        super(gen_neff, self).__init__()    
        self.model = Model(in_size = 1, out_size = modes, layers = layers, nodes = 128).requires_grad_(requires_grad=False)
    def forward(self, hs):
        '''
        input: 
            hs: array of waveguide widths [N**2,]
        output:
            neffs of each mode. shape: [N**2, number of modes for each waveguide.]
        '''
        return self.model(hs)
    def reset(self, path):
        model_state = torch.load(path + "fitting_neffs_state_dict", map_location=torch.device('cpu'))
        self.model.load_state_dict(model_state) 


class gen_U0(nn.Module):
    def __init__(self, modes, ln, le, out_res, N, n0, C_EPSILON, period, Knn):
        super(gen_U0, self).__init__()
        self.N = N
        self.n0 = n0
        self.C_EPSILON = C_EPSILON
        self.dx = period / out_res
        self.Knn = Knn
        self.modes = modes
        self.out_res = out_res
        self.neffnn = gen_neff(modes, ln).requires_grad_(requires_grad=False)
        enn_out_size = modes * (2 * (Knn + 1) * out_res)**2
        self.Ey_size = 2 * (Knn + 1) * out_res
        self.enn = Model(1, enn_out_size, layers= le, nodes = 128).requires_grad_(requires_grad=False)
        self.register_buffer('E0_slice', torch.zeros((N**2, 1, self.Ey_size, self.Ey_size), dtype= torch.complex64))
    def forward(self, hs, E0):
        '''
        input:
            hs: array of waveguide widths [N**2,]
            E0: input field [(N +  2 * Knn + 1) * out_res, (N +  2 * Knn + 1) * out_res]
        output:
            neff: refractive index of each mode. shape [N**2, modes]
            T: modes amplitude coupled in. shape [N**2, number of modes]
        '''
        neff = self.neffnn(hs.view(-1, 1))
        for i in range(2 * (self.Knn + 1)):
            for j in range(2 * (self.Knn + 1)):
                self.E0_slice[:,0, i * self.out_res: (i + 1) * self.out_res, j * self.out_res: (j + 1) * self.out_res] = \
                    (E0[i * self.out_res: (self.N + i) * self.out_res, j * self.out_res: (self.N + j) * self.out_res]).reshape(self.N**2, self.out_res, self.out_res)
            Ey = self.enn(hs.view(-1, 1))
            Ey = Ey.view(self.N**2, self.modes, self.Ey_size, self.Ey_size)
            E_sum = torch.sum(Ey * self.E0_slice, dim= (-2, -1), keepdim= False) # shape: [N**2, modes]
            eta = (neff * self.n0) / (neff + self.n0) #shape [N**2, modes]
            T = 2 * self.C_EPSILON * eta * E_sum * self.dx**2
            T = T.view(-1,)
        return Ey, T
    def reset(self, path):
        model_state = torch.load(path + "fitting_E_state_dict_outres_" + str(self.out_res), map_location=torch.device('cpu'))
        self.enn.load_state_dict(model_state)
        self.neffnn.reset(path)

class gen_En(nn.Module):
    def __init__(self, modes, out_res, N, n0, C_EPSILON, Knn):
        super(gen_En, self).__init__()
        self.N = N
        self.n0 = n0
        self.C_EPSILON = C_EPSILON
        self.Knn = Knn
        self.modes = modes
        self.out_res = out_res
        self.Ey_size = 2 * (Knn + 1) * out_res
        self.total_size = (N + 2 * Knn + 1) * out_res
        self.register_buffer('En', torch.zeros((self.total_size,self.total_size), dtype= torch.complex64))
    def forward(self, hs, U, neff, Ey):
        '''
            neff: shape [N**2, modes]
            Ey: shape [N**2, modes, fields]
        '''
        hs = hs.view(-1, 1)
        eta = (neff * self.n0) / (neff + self.n0) #shape: [N**2, modes]
        Ey = eta.view(-1, self.modes, 1, 1) * Ey * U.view(-1, self.modes, 1, 1)
        self.En = torch.zeros((self.total_size,self.total_size), dtype= torch.complex64).to(hs.device)
        for i in range(self.N):
            for j in range(self.N):
                for m in range(self.modes):
                    temp_Ey = Ey[i * self.N + j, m]
                    ci = int(i * self.out_res + (self.Knn + 1) * self.out_res)
                    cj = int(j * self.out_res + (self.Knn + 1) * self.out_res)
                    radius = int((self.Knn + 1) * self.out_res)
                    self.En[ci - radius: ci + radius, cj - radius: cj + radius] += temp_Ey
        return self.En
        
class gen_C(nn.Module):
    def __init__(self, modes, lc, N):
        super(gen_C, self).__init__()
        self.channels = modes**2
        self.cnn = Model(4, self.channels, layers= lc, nodes = 128).requires_grad_(requires_grad=False)
        self.N = N
        self.modes = modes 
        
    def forward(self, CK_inputs):
        '''
        input:
            CK_inputs: the cnn input is (hi, hj, dis), output is cij for all the channels.
            the CK_inputs includes all the possiable couplings for N**2 waveguides. shape [N**2, feasible distances, 4]
        '''
        C_stripped = self.cnn(CK_inputs.view(-1, 4))
        return C_stripped
    def reset(self, path):
        model_state = torch.load(path + "fitting_C_state_dict", map_location=torch.device('cpu'))
        self.cnn.load_state_dict(model_state)

class gen_K(nn.Module):
    def __init__(self, modes, lk, N):
        super(gen_K, self).__init__()
        self.channels = modes**2
        self.knn = Model(4, self.channels, layers= lk, nodes = 256).requires_grad_(requires_grad=False)
        self.N = N
        self.modes = modes 

    def forward(self, CK_inputs):
        '''
        input:
            CK_inputs: the cnn input is (hi, hj, dis), output is cij for all the channels.
            the CK_inputs includes all the possiable couplings for N waveguides. shape [N, 2 * (Knn + 1), 3]
        '''
        K_stripped = self.knn(CK_inputs.view(-1, 4))
        return K_stripped
    def reset(self, path):
        model_state = torch.load(path + "fitting_K_state_dict", map_location=torch.device('cpu'))
        self.knn.load_state_dict(model_state)

class SCMT_Model(nn.Module):
    def __init__(self, prop_dis, Euler_steps, devs, GP, COUPLING, APPROX, Ni, k_row, N, layer_neff, layer_C, layer_K, layer_E):
        super(SCMT_Model, self).__init__()
        self.prop = prop_dis
        total_size = (N + 2 * GP.Knn + 1) * GP.out_res
        self.metalayer1 = Metalayer(Euler_steps, devs, GP, COUPLING, APPROX, Ni, k_row, N, layer_neff, layer_C, layer_K, layer_E)
        self.freelayer1 = freespace_layer(self.prop, GP.lam, total_size, GP.period / GP.out_res).to(devs[0])
    def forward(self, E0):
        En = self.metalayer1(E0)
        Ef = self.freelayer1(En)
        If = torch.abs(Ef)**2
        #If = If/If.max()
        return If
    def reset(self, path):
        self.metalayer1.reset(path)

class Ideal_model(nn.Module):
    def __init__(self, prop_dis, GP, total_size):
        super(Ideal_model, self).__init__()
        self.prop = prop_dis
        self.phase = torch.nn.Parameter(torch.empty((total_size, total_size), dtype = torch.float))
        self.freelayer1 = freespace_layer(self.prop, GP.lam, total_size, GP.period / GP.out_res)
    def forward(self, E0):
        E = E0 * torch.exp(1j * self.phase)
        Ef = self.freelayer1(E)
        If = torch.abs(Ef)**2
        return If
      
class freespace_layer(nn.Module):
    def __init__(self, prop, lam, total_size, dx):
        super(freespace_layer, self).__init__()
        f_kernel = gen_f_kernel(prop, lam, total_size, dx)
        self.register_buffer('fk_const', f_kernel)
    def forward(self, En):
        Ef = fourier_conv(En, self.fk_const)
        return Ef

def propagator(prop, lam, total_size, dx):
    '''
        prop distance in free space
    '''
    def W(x, y, z, wavelength):
        r = np.sqrt(x*x+y*y+z*z)
        #w = z/r**2*(1/(np.pi*2*r)+1/(relative_wavelength*1j))*np.exp(1j*2*np.pi*r/relative_wavelength)
        w = z/(r**2)*(1/(wavelength*1j))*np.exp(1j*2*np.pi*r/wavelength)
        return w
    #plane_size: the numerical size of plane, this is got by (physical object size)/(grid)

    x = np.arange(-(total_size-1), total_size,1) * dx
    y = x.copy()
    coord_x, coord_y = np.meshgrid(x,y, sparse = False)
    G = W(coord_x, coord_y, prop, lam)
    #solid angle Sigma = integral(integral(sin(theta))dthtea)dphi
    theta = np.arctan(total_size * dx/prop)
    Sigma = 2 * np.pi * (1 - np.cos(theta))
    G_norm = (np.abs(G)**2).sum() * 4 * np.pi / Sigma 
    print("Free space energy conservation normalization G_norm:", str(G_norm))
    G = G / G_norm
    return G

def gen_f_kernel(prop, lam, total_size, dx):
    G = propagator(prop, lam, total_size, dx)
    f_kernel = np.fft.fft2(np.fft.ifftshift(G))
    f_kernel = torch.tensor(f_kernel, dtype = torch.complex64)
    print("f_kernel generated.")
    return f_kernel

class gen_Euler_with_backward_devs(torch.autograd.Function):
    @staticmethod
    def forward(ctx,Euler_steps, prop_dis, devs, U0_dev1, *As):
        ctx.save_for_backward(*As)
        ctx.Euler_steps = Euler_steps
        ctx.prop_dis = prop_dis
        ctx.devs = devs
        Urs = []
        Uis = []
        Ur0_dev1 = U0_dev1.real
        Ui0_dev1 = U0_dev1.imag
        dz = prop_dis/Euler_steps
        for i in range(Euler_steps):
            Urs.append(Ur0_dev1)
            Uis.append(Ui0_dev1)
            Ur1_dev1 = Ur0_dev1.clone()
            Ui1_dev1 = Ui0_dev1.clone()
            for j in range(len(As)):
                Ur0_devj = Ur0_dev1.to(devs[j])
                Ui0_devj = Ui0_dev1.to(devs[j])
                Ur1_devj = -dz * torch.sparse.mm(As[j], Ui0_devj)
                Ui1_devj = dz * torch.sparse.mm(As[j], Ur0_devj)
                Ur1_dev1 += Ur1_devj.to(devs[0])
                Ui1_dev1 += Ui1_devj.to(devs[0])
            Ur0_dev1 = Ur1_dev1
            Ui0_dev1 = Ui1_dev1
        ctx.Urs = Urs
        ctx.Uis = Uis
        Uz = Ur0_dev1 + 1j * Ui0_dev1 
        return Uz
    @staticmethod
    def backward(ctx, L_grad):
        As = ctx.saved_tensors
        Urs_dev1 = ctx.Urs
        Uis_dev1 = ctx.Uis     
        dz = ctx.prop_dis/ctx.Euler_steps
        def step_back(grad_output, A, x, dz):
            x_grad = torch.sparse.mm(A, grad_output)
            coo = A.indices()
            cooi = coo[0]
            cooj = coo[1]
            X = torch.take(x, cooj)
            B = torch.take(grad_output, cooi)
            Values = B * X
            A_grad = Values
            #A_grad = torch.sparse_coo_tensor(coo, Values, A.shape, dtype = A.dtype, requires_grad = False)
            #A_grad = A_grad.coalesce()
            return dz * A_grad, dz * x_grad
        #backward process
        step_Ui1_grad = L_grad.imag
        step_Ur1_grad = L_grad.real
        As_grad = []
        coos = []
        for j in range(len(As)):
            coo = As[j].indices()
            Aj_grad = torch.zeros((coo.shape[1],), dtype = As[j].dtype, requires_grad = False, device= As[j].device)  
            As_grad.append(Aj_grad)
            coos.append(coo)
        for i in range(ctx.Euler_steps - 1, -1, -1):
            Ur0_dev1 = Urs_dev1[i]
            Ui0_dev1 = Uis_dev1[i]
            step_Ur0_add = step_Ur1_grad.clone()
            step_Ui0_add = step_Ui1_grad.clone()
            for j in range(len(As)):
                Ur0_devj = Ur0_dev1.to(ctx.devs[j])
                Ui0_devj = Ui0_dev1.to(ctx.devs[j])
                step_Ui1_grad_devj = step_Ui1_grad.to(ctx.devs[j])
                step_Ur1_grad_devj = step_Ur1_grad.to(ctx.devs[j])
                #Ur1 = -dz * torch.sparse.mm(A, Ui0) + Ur0
                #Ui1 = dz * torch.sparse.mm(A, Ur0) + Ui0
                step_AjUr_grad, step_Ui0_grad_mm_devj = step_back(step_Ur1_grad_devj, As[j], Ui0_devj, -dz)
                step_AjUi_grad, step_Ur0_grad_mm_devj = step_back(step_Ui1_grad_devj, As[j], Ur0_devj, dz)
                As_grad[j] += step_AjUr_grad
                As_grad[j] += step_AjUi_grad
                step_Ur0_add += step_Ur0_grad_mm_devj.to(ctx.devs[0])
                step_Ui0_add += step_Ui0_grad_mm_devj.to(ctx.devs[0])
            step_Ur1_grad = step_Ur0_add
            step_Ui1_grad = step_Ui0_add
        
        out_As_grad = []
        for j in range(len(As)):
            Aj_grad = torch.sparse_coo_tensor(coos[j], As_grad[j], As[j].shape, dtype = As_grad[j].dtype, requires_grad = False)
            Aj_grad = Aj_grad.coalesce()
            out_As_grad.append(Aj_grad)
        return None, None, None, None, *out_As_grad