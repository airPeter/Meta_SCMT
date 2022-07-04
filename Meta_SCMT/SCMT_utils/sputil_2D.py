'''
    only support for Knn = 2.
'''

import numpy as np
import torch
#from torch.cuda import device_count
import torch.nn as nn
#import matplotlib.pyplot as plt

def gen_nn_index(i, N):
    pi = i//N
    qi = i%N
    index_list = []
    # (i-2, j)
    if pi - 2 < 0:
        index_list.append(None)
    else:
        index_list.append((pi - 2) * N + qi)
    # (i-1, j-1, j, j+1)
    if pi - 1 < 0:
        for x in range(-1, 2):
            index_list.append(None)
    else:
        for x in range(-1, 2):
            if (qi == 0 and x == -1) or (qi == N - 1 and x == 1):
                index_list.append(None)
            else:
                index_list.append((pi - 1) * N + qi + x)
    #(i, j-2, j-1, j, j+1, j+2)
    for x in range(-2, 3):
        if (qi + x) < 0 or (qi + x) >= N:
            index_list.append(None)
        else:
            index_list.append(pi * N + qi + x)
    #(i+1, j -1, j, j+1)
    if pi + 1 >= N:
        for x in range(-1, 2):
            index_list.append(None)
    else:
        for x in range(-1, 2):
            if (qi == 0 and x == -1) or (qi == N - 1 and x == 1):
                index_list.append(None)
            else:
                index_list.append((pi + 1) * N + qi + x)
    #(i+2, j)
    if pi + 2 >= N:
        index_list.append(None)
    else:
        index_list.append((pi + 2) * N + qi)
    return index_list

def gen_coo(N):
    out = []
    for i in range(N**2):
        index = gen_nn_index(i, N)
        temp_out = [(i, j) for j in index]
        out.append(temp_out)
    out = np.array(out)
    return out

def gen_coo_sparse(N):
    #the coordinate used to create a sparse matrix C.
    coo = gen_coo(N)
    for i in range(N**2):
        for j in range(13):
            if coo[i,j,-1] == None:
                coo[i,j,-1] = i #a specitial point that c(ni,nj,2,2) = 0
    coo = coo.reshape((-1, 2))
    coo = coo.T
    coo = coo.astype(int)
    print("coo_sparse generated.")
    return coo

def gen_coo_B_sparse(N):
    coo = []
    for i in range(N**2):
        coo.append([i, i])
    coo = np.array(coo, dtype= int)
    coo = coo.T
    print("coo_B_sparse generated.")
    return coo

def gen_dis_CK_input(N):
    #output shape: (N**2, 13, 2)
    dis_list = [[2 , 0]] +\
                    [[1, x] for x in range(1,-2, -1)] +\
                    [[0, x] for x in range(2,-3, -1)] +\
                    [[-1, x] for x in range(1,-2, -1)] +\
                    [[-2,0]]
    dis_list = np.array(dis_list)
    input_dis = np.zeros((N**2, 13, 2), dtype=int)
    coo = gen_coo(N)
    for i in range(N**2):
        for j in range(13):
            if coo[i,j,-1] == None:
                input_dis[i,j] = np.array([2,2],dtype=int) #a specitial point that c(ni,nj,2,2) = 0
            else:
                input_dis[i,j] = dis_list[j]
    print("dis model input generated.")
    return input_dis

def gen_index_list(N):
    out = []
    for i in range(N**2):
        temp = gen_nn_index(i, N)
        for j in range(13):
            if temp[j] == None: #if the neighor doesn't exist, we set the nj = ni
                temp[j] = i
        out.append(temp)
    out = np.array(out, dtype= int)
    out = out.reshape((-1,))
    return out

class gen_input_hs(nn.Module):
    def __init__(self, N):
        super(gen_input_hs, self).__init__()
        #index_list shape: (N**2 * 13,)
        index_list = torch.tensor(gen_index_list(N),dtype = torch.long)
        self.register_buffer('index_list', index_list)
        self.N = N
        
    def forward(self, hs):
        input_hs = torch.empty((self.N**2, 13, 2), dtype= torch.float).type_as(hs)
        ONEs = torch.ones((self.N**2, 13), dtype= torch.float).type_as(hs)
        input_hs[:,:,0] = ONEs * hs.view(self.N**2, 1)
        input_neighbor_hs = torch.take(hs, self.index_list)
        input_hs[:,:,1] = input_neighbor_hs.view(self.N**2, 13)
        return input_hs

def gen_Cinv_coo(N, Ni_half, k_row):
    coo = []
    for i in range(0, N**2, k_row):
        for k in range(k_row):
            temp = []
            for j in range(- (Ni_half), Ni_half):
                cj = i + j  + k_row//2
                if cj < 0 or cj >= N**2:
                    #if the point is out of the matrix boundary, the cinv value will be zero, we add that zero to diagonal element,
                    #when when we do the _coalease function.
                    temp.append([i + k,i + k])
                else:
                    temp.append([i + k, cj])
            coo.append(temp)
    coo = np.array(coo)
    coo = coo.reshape((-1, 2))
    coo = coo.T
    return coo

def gen_C_sub(i, N, Ni_half, Nj, C_stripped, coo):
    #C_stripped shape: C_stripped shape : (N**2*13,)
    #coo shape: (2, N**2 * 13)
    N_shift = Nj //2
    start = (i - Ni_half) * 13
    end = (i + Ni_half) * 13
    if start < 0:
        C_stripped_sub = C_stripped[0: end]
        coo_sub = coo[:,0: end]
    else:
        if end < N**2 * 13:
            C_stripped_sub = C_stripped[start: end]
            coo_sub = coo[:,start: end]
        else:
            C_stripped_sub = C_stripped[start:]
            coo_sub = coo[:,start:]
    coo_sub = coo_sub - i + N_shift
    sparse_C = torch.sparse_coo_tensor(coo_sub, C_stripped_sub, (Nj, Nj))
    C_sub = sparse_C.to_dense()
    C_sub = C_sub[N_shift - Ni_half: N_shift + Ni_half, N_shift - Ni_half: N_shift + Ni_half]
    if i - Ni_half < 0:
        for j in range(Ni_half - i):
            C_sub[j,j] = 1
    if i + Ni_half > N**2:
        for j in range((i + Ni_half - N**2) + 1):
            C_sub[-j, -j] = 1
    return C_sub
    

class gen_Cinv_rows(nn.Module):
    def __init__(self, N, Ni, k_row, start, end) -> None:
        super(gen_Cinv_rows, self).__init__()
        self.Ni_half = Ni//2
        self.Ni = self.Ni_half * 2
        self.N = N
        self.start = start
        self.end = end
        if N%k_row != 0:
            raise ValueError(" k_row should be divided by N")
        self.k_row = k_row
        self.Nj = self.Ni + 2 * (2 * N + 1)
        Cinv_stripped = torch.zeros((end - start, self.Ni), dtype= torch.float)
        self.register_buffer('Cinv_stripped', Cinv_stripped)
        #Cinv_coo shape: (2, N**2 * Ni)
        Cinv_coo = torch.tensor(gen_Cinv_coo(self.N, self.Ni_half, self.k_row)[:, start * self.Ni: end * self.Ni], dtype= torch.long)
        self.register_buffer('Cinv_coo', Cinv_coo)
        I_sub = torch.zeros((self.Ni, self.k_row), dtype= torch.float)
        for k in range(k_row):
            I_sub[self.Ni_half + k - k_row//2, k] = 1
        self.register_buffer('I_sub', I_sub)
        
    def forward(self, C_stripped, coo):
        for i in range(self.start, self.end, self.k_row):
            C_sub_center = i + self.k_row//2
            C_sub = gen_C_sub(C_sub_center, self.N, self.Ni_half, self.Nj, C_stripped, coo)
            CC_inv = torch.inverse(C_sub @ C_sub.T)
            Cinv_sub = CC_inv @ C_sub @ self.I_sub
            Cinv_sub = Cinv_sub.T
            self.Cinv_stripped[i - self.start: i - self.start + self.k_row] = Cinv_sub
        Cinv_sparse = torch.sparse_coo_tensor(self.Cinv_coo, self.Cinv_stripped.view(-1,), (self.N**2, self.N**2))
        return Cinv_sparse
    
def to_stripped(C, N):
    stripped_C = torch.zeros((N**2, 13), dtype= torch.float).type_as(C)
    for i in range(-2, 0):
        stripped_C[-i:,i + 6] = torch.diagonal(C, offset = i)
    stripped_C[:,6] = torch.diagonal(C, offset = 0)
    for i in range(1, 3):
        stripped_C[:-i,i + 6] = torch.diagonal(C, offset = i)
    for i in range(-1, 2):
        stripped_C[:-(N + i),i + 10] = torch.diagonal(C, offset = N + i)
    stripped_C[:-2 *N,12] = torch.diagonal(C, offset = 2 * N)
    
    for i in range(-1, 2):
        stripped_C[(N - i):,i + 2] = torch.diagonal(C, offset = - N + i)
    stripped_C[2 * N:,0] = torch.diagonal(C, offset = - 2 * N)
    return stripped_C


        