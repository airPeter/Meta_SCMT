import numpy as np
import torch
import torch.nn as nn
from .SCMT_model_1D import gen_neff, gen_C, gen_En, gen_K, gen_U0, freespace_layer
from .sputil_1D import gen_dis_CK_input, gen_input_hs

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
        for meta_idx in range(len(self.lams)):
            meta_subs.append(Metalayer_sub(self.GP, self.COUPLING, N, meta_idx))
        self.meta_subs = nn.ModuleList(meta_subs)
        
    def forward(self, E0):
        '''
        size of E0: (N + 2 * (Knnc + 1)) * period_resolution
        '''
        self.hs = self.sig(self.h_paras) * (self.h_max - self.h_min) + self.h_min
        #self.hs = torch.div(self.hs, self.GP.dh, rounding_mode = 'floor') * self.GP.dh
        Ens = []
        for metasub in self.meta_subs:
            Ens.append(metasub(E0, self.hs))
        return Ens
    def reset(self):
        #nn.init_normal_(self.phase, 0, 0.02)
        torch.nn.init.constant_(self.h_paras, val = 0.0)
        if len(self.paths) != len(self.lams):
            raise Exception("number of paths != number of lams.")
        for i, path in enumerate(self.paths):
            self.meta_subs[i].reset(path)
            
#sub module for each wavelength
class Metalayer_sub(torch.nn.Module):
    def __init__(self, GP, COUPLING, N, meta_idx):
        '''

        '''
        super(Metalayer_sub, self).__init__()
        self.COUPLING = COUPLING
        self.neffs = None
        self.GP = GP
        self.wh = GP.wh
        path = GP.paths[meta_idx]
        lam = GP.lams[meta_idx]
        self.k = 2 * np.pi / lam
        neff_paras = np.load(path + "neff_paras.npy", allow_pickle= True)
        neff_paras = neff_paras.item()
        C_paras = np.load(path + "C_paras.npy", allow_pickle= True)
        C_paras = C_paras.item()
        K_paras = np.load(path + "K_paras.npy", allow_pickle= True)
        K_paras = K_paras.item()
        E_paras = np.load(path + "E_paras.npy", allow_pickle= True)
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
        size of E0: (N) * period_resolution
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
        total_size = (N) * GP.res
        self.metalayer1 = Metalayer(GP, COUPLING, N)
        self.lams = GP.lams
        freelayers = []
        for lam in self.lams:
            freelayers.append(freespace_layer(self.prop, lam, total_size, GP.dx))
        self.freelayers = nn.ModuleList(freelayers)
        
    def forward(self, E0):
        Ens = self.metalayer1(E0)
        Efs = []
        for i in range(len(self.freelayers)):
            Efs.append(self.freelayers[i](Ens[i]))
        # If = torch.abs(Efs[0])**2
        # for tmp_E in Efs[1:]:
        #     If = If + torch.abs(tmp_E)**2
        If = [torch.abs(E)**2 for E in Efs]
        return If
    def reset(self):
        self.metalayer1.reset()
