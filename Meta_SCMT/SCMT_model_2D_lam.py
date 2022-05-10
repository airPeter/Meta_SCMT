'''
    when use multi-GPU, how to assign device need special care.
    for devs[i], it will be the main host of lam[i].
    For the matrix inverse and exponential operation, the work load will be shared by all devices.

'''
import numpy as np
import torch
import torch.nn as nn
from .SCMT_model_2D import gen_neff, gen_C, gen_En, gen_K, gen_U0, gen_Euler_with_backward_devs, freespace_layer
from .sputil_2D import gen_coo_sparse, gen_dis_CK_input, gen_input_hs, gen_Cinv_rows

class Metalayer(torch.nn.Module):
    def __init__(self, Euler_steps, devs, GP, COUPLING, APPROX, Ni, k_row, N):
        super(Metalayer, self).__init__()
        self.COUPLING = COUPLING
        self.devs = devs
        self.h_paras = torch.nn.Parameter(torch.empty((N * N,), dtype = torch.float))
        self.hs = None
        self.GP = GP
        self.paths = self.GP.paths
        self.h_min = GP.h_min
        self.h_max = GP.h_max
        self.wh = GP.wh
        self.sig = torch.nn.Sigmoid()
        self.lams = self.GP.lams
        dis = gen_dis_CK_input(N)
        coo = gen_coo_sparse(N)

        meta_subs = []
        for meta_idx in range(len(self.lams)):
            meta_subs.append(Metalayer_sub(Euler_steps, devs, self.GP, COUPLING, APPROX, Ni, k_row, N, meta_idx, dis, coo))
        self.meta_subs = nn.ModuleList(meta_subs)

    def forward(self, E0):
        '''
        size of E0: (N + 2 * (Knnc + 1)) * period_resolution
        '''
        self.hs = self.sig(self.h_paras.to(self.devs[0])) * (self.h_max - self.h_min) + self.h_min
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
            
class Metalayer_sub(torch.nn.Module):
    def __init__(self, Euler_steps, devs, GP, COUPLING, APPROX, Ni, k_row, N, meta_idx, dis, coo):
        '''

        '''
        super(Metalayer_sub, self).__init__()
        self.N = N
        self.Euler_steps = Euler_steps
        self.devs = devs
        self.num_devs = len(devs)
        if N%self.num_devs != 0:
            raise ValueError(" num_devs should be divided by N")
        if len(self.devs) >= len(GP.lams):
            self.dev_idx = meta_idx
        else:
            self.dev_idx = 0
        gen_cinv_modules = []
        rows_per_dev = N**2//self.num_devs
        for i in range(self.num_devs):
            row_start = i * rows_per_dev
            row_end = (i + 1) * rows_per_dev
            gen_cinv_modules.append(gen_Cinv_rows(N, Ni, k_row, row_start, row_end).to(self.devs[i]))
        self.gen_cinv = nn.ModuleList(gen_cinv_modules)
        
        self.COUPLING = COUPLING
        self.APPROX = APPROX
        self.neffs = None
        self.GP = GP
        self.h_min = GP.h_min
        self.h_max = GP.h_max
        self.dh = GP.dh
        self.wh = GP.wh
        self.k = 2 * np.pi / self.GP.lams[meta_idx]
        path = GP.paths[meta_idx]
        neff_paras = np.load(path + "neff_paras.npy", allow_pickle= True)
        neff_paras = neff_paras.item()
        C_paras = np.load(path + "C_paras.npy", allow_pickle= True)
        C_paras = C_paras.item()
        K_paras = np.load(path + "K_paras.npy", allow_pickle= True)
        K_paras = K_paras.item()
        E_paras = np.load(path + "E_paras.npy", allow_pickle= True)
        E_paras = E_paras.item()
        self.neffnn = gen_neff(GP.modes, neff_paras['nodes'], neff_paras['layers']).to(self.devs[self.dev_idx])
        self.genc = gen_C(GP.modes, C_paras['nodes'], C_paras['layers'], N).to(self.devs[self.dev_idx])
        self.genk = gen_K(GP.modes, K_paras['nodes'], K_paras['layers'], N).to(self.devs[self.dev_idx])
        self.genu0 = gen_U0(GP.modes, neff_paras['nodes'], neff_paras['layers'], E_paras['nodes'], E_paras['layers'], GP.out_res, N, GP.n0, GP.C_EPSILON, GP.period, GP.Knn).to(self.devs[self.dev_idx])
        self.genen = gen_En(GP.modes, GP.out_res, N, GP.n0, GP.C_EPSILON, GP.Knn).to(self.devs[self.dev_idx])
        if GP.Knn != 2:
            raise Exception("Knn = 2 is hardcode in sputil_2D module. So only Knn = 2 is supported.")
        self.gen_hs_input = gen_input_hs(N).to(self.devs[self.dev_idx])
        dis = torch.tensor(dis, dtype = torch.float, requires_grad = False).to(self.devs[self.dev_idx])
        self.register_buffer('dis', dis)
        coo = torch.tensor(coo,dtype=int, requires_grad = False).to(self.devs[self.dev_idx])
        self.register_buffer('coo', coo)
    def forward(self, E0, hs):
        '''
        size of E0: (N + 2 * (Knnc + 1)) * self.out_res
        '''
        hs = hs.to(self.devs[self.dev_idx])
        E0 = E0.to(self.devs[self.dev_idx])
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
    def __init__(self, prop_dis, Euler_steps, devs, GP, COUPLING, APPROX, Ni, k_row, N):
        super(SCMT_Model, self).__init__()
        self.prop = prop_dis
        total_size = (N) * GP.res
        self.metalayer1 = Metalayer(Euler_steps, devs, GP, COUPLING, APPROX, Ni, k_row, N)
        self.lams = GP.lams
        freelayers = []
        self.devs = devs
        if len(devs) >= len(self.lams):
            for idx, lam in enumerate(self.lams):
                freelayers.append(freespace_layer(self.prop, lam, total_size, GP.period / GP.out_res).to(devs[idx]))            
        else:
            for lam in self.lams:
                freelayers.append(freespace_layer(self.prop, lam, total_size, GP.period / GP.out_res).to(devs[0]))
        self.freelayers = nn.ModuleList(freelayers)
        
    def forward(self, E0):
        Ens = self.metalayer1(E0)
        Efs = []
        if len(self.devs) >= len(self.lams):
            for i in range(len(self.freelayers)):
                Efs.append(self.freelayers[i](Ens[i]))       
        else:
            for i in range(len(self.freelayers)):
                Efs.append(self.freelayers[i](Ens[i]))
        If = [torch.abs(E.to(E0.device))**2 for E in Efs]     
        # If = torch.abs(Efs[0])**2
        # for tmp_E in Efs[1:]:
        #     If = If + torch.abs(tmp_E)**2
        return If
    def reset(self):
        self.metalayer1.reset()

# '''
#     single host (devs[0]) for all lams
# '''
# import numpy as np
# import torch
# import torch.nn as nn
# from .SCMT_model_2D import gen_neff, gen_C, gen_En, gen_K, gen_U0, gen_Euler_with_backward_devs, freespace_layer
# from .sputil_2D import gen_coo_sparse, gen_dis_CK_input, gen_input_hs, gen_Cinv_rows

# class Metalayer(torch.nn.Module):
#     def __init__(self, Euler_steps, devs, GP, COUPLING, APPROX, Ni, k_row, N):
#         super(Metalayer, self).__init__()
#         self.COUPLING = COUPLING
#         self.devs = devs
#         self.h_paras = torch.nn.Parameter(torch.empty((N * N,), dtype = torch.float))
#         self.hs = None
#         self.GP = GP
#         self.paths = self.GP.paths
#         self.h_min = GP.h_min
#         self.h_max = GP.h_max
#         self.wh = GP.wh
#         self.sig = torch.nn.Sigmoid()
#         self.lams = self.GP.lams
#         dis = gen_dis_CK_input(N)
#         coo = gen_coo_sparse(N)
#         meta_subs = []
#         for meta_idx in range(len(self.lams)):
#             meta_subs.append(Metalayer_sub(Euler_steps, devs, self.GP, COUPLING, APPROX, Ni, k_row, N, meta_idx, dis, coo))
#         self.meta_subs = nn.ModuleList(meta_subs)
        
#     def forward(self, E0):
#         '''
#         size of E0: (N + 2 * (Knnc + 1)) * period_resolution
#         '''
#         self.hs = self.sig(self.h_paras.to(self.devs[0])) * (self.h_max - self.h_min) + self.h_min
#         Ens = []
#         for metasub in self.meta_subs:
#             Ens.append(metasub(E0, self.hs))
#         return Ens
#     def reset(self):
#         #nn.init_normal_(self.phase, 0, 0.02)
#         torch.nn.init.constant_(self.h_paras, val = 0.0)
#         if len(self.paths) != len(self.lams):
#             raise Exception("number of paths != number of lams.")
#         for i, path in enumerate(self.paths):
#             self.meta_subs[i].reset(path)
            
# class Metalayer_sub(torch.nn.Module):
#     def __init__(self, Euler_steps, devs, GP, COUPLING, APPROX, Ni, k_row, N, meta_idx, dis, coo):
#         '''

#         '''
#         super(Metalayer_sub, self).__init__()
#         self.N = N
#         self.Euler_steps = Euler_steps
#         self.devs = devs
#         self.num_devs = len(devs)
#         if N%self.num_devs != 0:
#             raise ValueError(" num_devs should be divided by N")
#         gen_cinv_modules = []
#         rows_per_dev = N**2//self.num_devs
#         for i in range(self.num_devs):
#             row_start = i * rows_per_dev
#             row_end = (i + 1) * rows_per_dev
#             gen_cinv_modules.append(gen_Cinv_rows(N, Ni, k_row, row_start, row_end).to(self.devs[i]))
#         self.gen_cinv = nn.ModuleList(gen_cinv_modules)
        
#         self.COUPLING = COUPLING
#         self.APPROX = APPROX
#         self.neffs = None
#         self.GP = GP
#         self.h_min = GP.h_min
#         self.h_max = GP.h_max
#         self.dh = GP.dh
#         self.wh = GP.wh
#         self.k = 2 * np.pi / self.GP.lams[meta_idx]
#         path = GP.paths[meta_idx]
#         neff_paras = np.load(path + "neff_paras.npy", allow_pickle= True)
#         neff_paras = neff_paras.item()
#         C_paras = np.load(path + "C_paras.npy", allow_pickle= True)
#         C_paras = C_paras.item()
#         K_paras = np.load(path + "K_paras.npy", allow_pickle= True)
#         K_paras = K_paras.item()
#         E_paras = np.load(path + "E_paras.npy", allow_pickle= True)
#         E_paras = E_paras.item()
#         self.neffnn = gen_neff(GP.modes, neff_paras['nodes'], neff_paras['layers']).to(self.devs[0])
#         self.genc = gen_C(GP.modes, C_paras['nodes'], C_paras['layers'], N).to(self.devs[0])
#         self.genk = gen_K(GP.modes, K_paras['nodes'], K_paras['layers'], N).to(self.devs[0])
#         self.genu0 = gen_U0(GP.modes, neff_paras['nodes'], neff_paras['layers'], E_paras['nodes'], E_paras['layers'], GP.out_res, N, GP.n0, GP.C_EPSILON, GP.period, GP.Knn).to(self.devs[0])
#         self.genen = gen_En(GP.modes, GP.out_res, N, GP.n0, GP.C_EPSILON, GP.Knn).to(self.devs[0])
#         if GP.Knn != 2:
#             raise Exception("Knn = 2 is hardcode in sputil_2D module. So only Knn = 2 is supported.")
#         self.gen_hs_input = gen_input_hs(N).to(self.devs[0])
#         dis = torch.tensor(dis, dtype = torch.float, requires_grad = False).to(self.devs[0])
#         self.register_buffer('dis', dis)
#         coo = torch.tensor(coo,dtype=int, requires_grad = False).to(self.devs[0])
#         self.register_buffer('coo', coo)
        
#     def forward(self, E0, hs):
#         '''
#         size of E0: (N + 2 * (Knnc + 1)) * self.out_res
#         '''
#         self.neffs = self.neffnn(hs.view(-1, 1))
#         with torch.set_grad_enabled(False):
#             Eys, U0 = self.genu0(hs, E0)
#         if not self.COUPLING:
#             P = torch.exp(self.neffs.view(-1,) * self.k * self.wh * 1j)
#             Uz = P * U0 #shape [N*modes,]
#         else:
#             with torch.set_grad_enabled(False):
#                 hs_input = self.gen_hs_input(hs)
#                 CK_input = torch.cat([hs_input, self.dis],dim = -1)
#                 CK_input = CK_input.view(-1,4)                
#                 C_stripped = self.genc(CK_input)
#                 K_stripped = self.genk(CK_input)
#                 K_sparse = torch.sparse_coo_tensor(self.coo, K_stripped.view(-1,), (self.N**2, self.N**2))
#                 K_sparse = K_sparse.coalesce()

#             BC_stripped = self.neffs.view(-1, 1) * C_stripped.view(-1, 13)
#             BC_sparse = self.k * torch.sparse_coo_tensor(self.coo, BC_stripped.view(-1,), (self.N**2, self.N**2))
#             if self.APPROX == 0:
#                 C_sparse = torch.sparse_coo_tensor(self.coo, C_stripped.view(-1,), (self.N**2, self.N**2))
#                 C_sparse = C_sparse.coalesce()
#                 C_dense = C_sparse.to_dense()
#                 C_inv = torch.inverse(C_dense)
#                 K_dense = K_sparse.to_dense()
#                 A = C_inv @ (BC_sparse.to_dense() + K_dense)
#                 P = torch.matrix_exp(1j * A * self.wh) #waveguide propagator
#                 Uz = P @ U0
#             else:
#                 C_inv_sparse_list = []
#                 for i in range(self.num_devs):
#                     C_stripped = C_stripped.view(-1,)
#                     C_inv_sparse = self.gen_cinv[i](C_stripped.to(self.devs[i]), self.coo.to(self.devs[i]))
#                     C_inv_sparse = C_inv_sparse.coalesce()
#                     C_inv_sparse_list.append(C_inv_sparse)
#                 A_list = []
#                 for i in range(self.num_devs):
#                     C_inv_sparse_i = C_inv_sparse_list[i]
#                     BC_sparse_i = BC_sparse.to(self.devs[i])
#                     K_sparse_i = K_sparse.to(self.devs[i])
#                     A_i = torch.sparse.mm(C_inv_sparse_i, (BC_sparse_i + K_sparse_i))               
#                     A_i = A_i.coalesce()
#                     A_list.append(A_i)
#                 Uz = gen_Euler_with_backward_devs.apply(self.Euler_steps, self.wh, self.devs, U0.view(-1, 1), *A_list)
#         with torch.set_grad_enabled(False):
#             hs_no_grad = hs
#             neffs_no_grad = self.neffs
#         En = self.genen(hs_no_grad, Uz, neffs_no_grad, Eys) #near field
#         return En
    
#     def reset(self, path):
#         self.neffnn.reset(path)
#         self.genc.reset(path)
#         self.genk.reset(path)
#         self.genu0.reset(path)

# class SCMT_Model(nn.Module):
#     def __init__(self, prop_dis, Euler_steps, devs, GP, COUPLING, APPROX, Ni, k_row, N):
#         super(SCMT_Model, self).__init__()
#         self.prop = prop_dis
#         total_size = (N + 2 * GP.Knn + 1) * GP.res
#         self.metalayer1 = Metalayer(Euler_steps, devs, GP, COUPLING, APPROX, Ni, k_row, N)
#         self.lams = GP.lams
#         freelayers = []
#         for lam in self.lams:
#             freelayers.append(freespace_layer(self.prop, lam, total_size, GP.period / GP.out_res).to(devs[0]))
#         self.freelayers = nn.ModuleList(freelayers)
        
#     def forward(self, E0):
#         Ens = self.metalayer1(E0)
#         Efs = []
#         for i in range(len(self.freelayers)):
#             Efs.append(self.freelayers[i](Ens[i]))
#         # If = torch.abs(Efs[0])**2
#         # for tmp_E in Efs[1:]:
#         #     If = If + torch.abs(tmp_E)**2
#         If = [torch.abs(E)**2 for E in Efs]
#         return If
#     def reset(self):
#         self.metalayer1.reset()
