import numpy as np
import torch
import torch.nn as nn
from .SCMT_model_1D import gen_neff, gen_C, gen_En, gen_K, gen_U0
from .sputil_1D import gen_dis_CK_input, gen_input_hs
from scipy import special


class Metalayer(torch.nn.Module):
    def __init__(self, GP, COUPLING, N):
        super(Metalayer, self).__init__()
        self.COUPLING = COUPLING
        self.h_paras = torch.nn.Parameter(torch.empty((N,), dtype = torch.float))
        self.hs = None
        self.GP = GP
        self.paths = self.GP.paths
        self.h_min = GP.h_min
        self.h_max = GP.h_max
        self.wh = GP.wh
        self.sig = torch.nn.Sigmoid()
        self.lams = self.GP.lams
        meta_subs = []
        for lam in self.lams:
            meta_subs.append(Metalayer_sub(self.GP, self.COUPLING, N, 2 * np.pi / lam))
        self.meta_subs = nn.ModuleList(meta_subs)
        
    def forward(self, E0):
        '''
        size of E0: (N + 2 * (Knnc + 1)) * period_resolution
        '''
        self.hs = self.sig(self.h_paras) * (self.h_max - self.h_min) + self.h_min
        Ens = []
        for metasub in self.meta_subs:
            Ens.append(metasub(E0, self.hs))
        return Ens
    def reset(self):
        if len(self.paths) != len(self.lams):
            raise Exception("number of paths != number of lams.")
        for i, path in enumerate(self.paths):
            self.meta_subs[i].reset(path)
            
#sub module for each wavelength
class Metalayer_sub(torch.nn.Module):
    def __init__(self, GP, COUPLING, N, k):
        '''

        '''
        super(Metalayer_sub, self).__init__()
        self.COUPLING = COUPLING
        self.neffs = None
        self.GP = GP
        self.wh = GP.wh
        self.k = k
        neff_paras = np.load(self.GP.path + "neff_paras.npy", allow_pickle= True)
        neff_paras = neff_paras.item()
        C_paras = np.load(self.GP.path + "C_paras.npy", allow_pickle= True)
        C_paras = C_paras.item()
        K_paras = np.load(self.GP.path + "K_paras.npy", allow_pickle= True)
        K_paras = K_paras.item()
        E_paras = np.load(self.GP.path + "E_paras.npy", allow_pickle= True)
        E_paras = E_paras.item()
        self.neffnn = gen_neff(GP.modes, neff_paras['nodes'], neff_paras['layers'])
        self.genc = gen_C(GP.modes, C_paras['nodes'], C_paras['layers'], N, GP.Knn)
        self.genk = gen_K(GP.modes, K_paras['nodes'], K_paras['layers'], N, GP.Knn)
        self.genu0 = gen_U0(GP.modes, neff_paras['nodes'], neff_paras['layers'], E_paras['nodes'], E_paras['layers'], GP.res, N, GP.n0, GP.C_EPSILON, GP.dx, GP.Knn)
        self.genen = gen_En(GP.modes, GP.res, N, GP.n0, GP.C_EPSILON, GP.dx, GP.Knn)
        self.gen_hs_input = gen_input_hs(N, GP.Knn)
        dis = torch.tensor(gen_dis_CK_input(N, GP.Knn), dtype = torch.float, requires_grad = False)
        self.register_buffer('dis', dis)
    def forward(self, E0, hs):
        '''
        size of E0: (N + 2 * (Knnc + 1)) * period_resolution
        '''
        self.neffs = self.neffnn(hs.view(-1, 1))
        with torch.set_grad_enabled(False):
            Eys, U0 = self.genu0(hs, E0)
        if not self.COUPLING:
            P = torch.exp(self.neffs.view(-1,) * self.k * self.wh * 1j)
            Uz = P * U0 #shape [N*modes,]
        else:
            with torch.set_grad_enabled(False):
                hs_input = self.gen_hs_input(hs)
                CK_input = torch.cat([hs_input, self.dis],dim = -1)
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
            hs_no_grad = hs
            neffs_no_grad = self.neffs
        En = self.genen(hs_no_grad, Uz, neffs_no_grad, Eys) #near field
        return En
    
    def reset(self, path):
        self.neffnn.reset(path)
        self.genc.reset(path)
        self.genk.reset(path)
        self.genu0.reset(path)


class SCMT_Model(nn.Module):
    def __init__(self, prop_dis, GP, COUPLING, N):
        super(SCMT_Model, self).__init__()
        self.prop = prop_dis
        total_size = (N + 2 * GP.Knn + 1) * GP.res
        self.metalayer1 = Metalayer(GP, COUPLING, N)
        self.lams = GP.lams
        freelayers = []
        for lam in self.lams:
            freelayers.append(freespace_layer(2 * np.pi / lam, self.prop, total_size, GP.dx))
        self.freelayers = nn.ModuleList(freelayers)
        
    def forward(self, E0):
        Ens = self.metalayer1(E0)
        Efs = []
        for i in range(len(self.freelayers)):
            Efs.append(self.freelayers[i](Ens[i]))
        Ef = Efs[0]
        for tmp_E in Efs[1:]:
            Ef = Ef + tmp_E
        If = torch.abs(Ef)**2
        return If
    def reset(self):
        self.metalayer1.reset()
           
class freespace_layer(nn.Module):
    def __init__(self, k, prop, total_size, dx):
        super(freespace_layer, self).__init__()
        G = torch.tensor(gen_G(k, prop, total_size, dx), dtype= torch.complex64)
        self.register_buffer('G', G)
    def forward(self, En):
        Ef = self.G @ En
        return Ef

def gen_G(k, prop, total_size, dx):
    x = (np.arange(total_size)) * dx
    inplane_dis = np.reshape(x, (1,-1)) - np.reshape(x, (-1, 1))
    r = np.sqrt(prop**2 + inplane_dis**2)
    v = 1
    G = -1j * k / 4 * special.hankel1(v, k * r) * prop / r * dx
    return G

# def propagator(k, prop, total_size, dx):
#     '''
#         prop distance in free space
#     '''
#     def W(x, y, z, wavelength):
#         r = np.sqrt(x*x+y*y+z*z)
#         #w = z/r**2*(1/(np.pi*2*r)+1/(relative_wavelength*1j))*np.exp(1j*2*np.pi*r/relative_wavelength)
#         w = z/(r**2)*(1/(wavelength*1j))*np.exp(1j*2*np.pi*r/wavelength)
#         return w
#     #plane_size: the numerical size of plane, this is got by (physical object size)/(grid)

#     x = np.arange(-(total_size-1), total_size,1) * dx
#     lam = 2 * np.pi / k
#     G = W(x, 0, prop, lam)
#     #solid angle Sigma = integral(integral(sin(theta))dthtea)dphi
#     # theta = np.arctan(total_size * dx/prop)
#     # Sigma = 2 * np.pi * (1 - np.cos(theta))
#     # G_norm = (np.abs(G)**2).sum() * 4 * np.pi / Sigma 
#     # print(f"Free space energy conservation normalization G_norm: {G_norm:.2f}")
#     # G = G / G_norm
#     return G
