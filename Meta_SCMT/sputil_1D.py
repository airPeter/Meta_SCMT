'''
    the functions will needed for generate sparse C and K matrix.
'''
import numpy as np
import torch
import torch.nn as nn

def gen_coo(N, Knn):
    out = []
    num_neighbor = 2 * Knn + 1
    for i in range(N):
        index = gen_nn_index(i, N, Knn)
        temp_out = [(i, j) for j in index]
        out.append(temp_out)
    out = np.array(out)
    return out

def gen_coo_sparse(N, Knn):
    #the coordinate used to create a sparse matrix C.
    coo = gen_coo(N, Knn)
    num_neighbor = 2 * Knn + 1
    for i in range(N):
        for j in range(num_neighbor):
            if coo[i,j,-1] == None:
                coo[i,j,-1] = i #a specitial point that c(ni,nj,Knn + 1) = 0
    coo = coo.reshape((-1, 2))
    coo = coo.T
    coo = coo.astype(int)
    print("coo_sparse generated.")
    return coo

def gen_index_list(N, Knn):
    out = []
    num_neighbors = 2 * Knn + 1
    for i in range(N):
        temp = gen_nn_index(i, N, Knn)
        for j in range(num_neighbors):
            if temp[j] == None: #if the neighor doesn't exist, we set the nj = ni
                temp[j] = i
        out.append(temp)
    out = np.array(out, dtype= int)
    out = out.reshape((-1,))
    return out

def gen_nn_index(i, N, Knn):
    '''
    output:
        shape [2 * Knn + 1]
    '''
    index_list = []
    #(i-2, i-1, i, i+1, i+2)
    for x in range(-Knn, Knn + 1):
        if (i + x) < 0 or (i + x) >= N:
            index_list.append(None)
        else:
            index_list.append(i + x)
    return index_list

class gen_input_hs(nn.Module):
    def __init__(self, N, Knn):
        super(gen_input_hs, self).__init__()
        #index_list shape: (N**2 * 13,)
        self.num_neighbors = 2 * Knn + 1
        index_list = torch.tensor(gen_index_list(N, Knn),dtype = torch.long)
        self.register_buffer('index_list', index_list)
        self.N = N
    def forward(self, hs):
        input_hs = torch.empty((self.N, self.num_neighbors, 2), dtype= torch.float).type_as(hs)
        ones = torch.ones((self.N, self.num_neighbors), dtype= torch.float).type_as(hs)
        input_hs[:,:,0] = ones * hs.view(self.N, 1)
        input_neighbor_hs = torch.take(hs, self.index_list)
        input_hs[:,:,1] = input_neighbor_hs.view(self.N, self.num_neighbors)
        return input_hs

def gen_dis_CK_input(N, Knn):
    #output shape: (N, 2 * Knn + 1, 2)
    dis_list = np.arange(-Knn, Knn + 2)/Knn #dis is normalized by Knn
    num_neighbors = 2 * Knn + 1
    input_dis = np.zeros((N, num_neighbors, 1), dtype=float)
    coo = gen_coo(N, Knn)
    for i in range(N):
        for j in range(num_neighbors):
            if coo[i,j,-1] == None:
                input_dis[i,j] = dis_list[-1]
            else:
                input_dis[i,j] = dis_list[j]
    print("dis model input generated.")
    return input_dis