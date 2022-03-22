'''
    fit a Fully connect met, that take (hi, hj, dis/self.Knn) as input, output Kij for each channels. 
    number of channel equals to modes**2.
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import k_means
import torch
from .utils import h2index, Model, train, resize_field2D
from tqdm import tqdm
import os

class Fitting_K_matrix_2D():
    def __init__(self, gen_modes, modes, res, dh, dx, Knn, path, n_wg, n0, k, C_EPSILON, period) -> None:
        self.gen_modes = gen_modes
        self.res = res
        self.period = period
        self.dh = dh
        self.upsample_res = int(round(self.period/self.dh))
        self.dx = dx
        self.Knn = Knn
        self.modes = modes
        self.channels = self.modes**2
        self.model = None
        self.path = path
        self.n_wg = n_wg
        self.n0 = n0
        self.k = k
        self.C_EPSILON = C_EPSILON
        
    def fit(self, layers = 4, nodes = 256, steps = 1000, lr = 0.001, vis = True, load = True, save_fig = False):
        X, Y = self.gen_fitting_data(load)
        self.model = Model(4, self.channels, layers= layers, nodes = nodes)
        batch_size = 512
        Y_pred = train(self.model, X, Y, steps, lr, batch_size)
        torch.save(self.model.state_dict(), self.path + "fitting_K_state_dict")
        print("model saved.")
        K_paras = {'nodes': nodes, 'layers': layers}
        np.save(self.path + "K_paras.npy", K_paras)
        feasible_dis = self.gen_feasible_dis()
        feasible_dis_len = len(feasible_dis)
        if vis:
            Y_pred = Y_pred.reshape(-1,  feasible_dis_len, self.channels)
            Y = Y.reshape(-1,  feasible_dis_len, self.channels)
            for dis_index, dis in enumerate(feasible_dis):
                plt.figure()
                for ch in range(self.channels):
                    plt.plot( Y[:, dis_index, ch], label = "ch:" + str(ch))
                    plt.plot(Y_pred[:, dis_index, ch], linestyle = '--', label = "ch:" + str(ch))
                    plt.legend()
                plt.xlabel("vary widths" + "dis:" + str(dis))
                plt.ylabel("Kij")
                if not save_fig:
                    plt.show()
                else:
                    plt.savefig(self.path + "fit_K_" + "vary widths" + "dis:" + str(dis)+ ".png")
        return None
    
    def gen_fitting_data(self,load):
        '''
            output:
            K_input: shape: [widths * widths]
            K_map: shape: [widths * widths, modes**2]
        '''
        map_path  = self.path + "K_map.npy"
        input_path = self.path + "K_input.npy"
        if load:
            if os.path.exists(map_path) and os.path.exists(input_path):
                K_map = np.load(map_path)
                K_input = np.load(input_path)
            else:
                raise Exception("K map, K_input not generated. set load to false")
        else:
            modes_lib = self.gen_modes.modes_lib
            if modes_lib == None:
                raise Exception("gen modes first!")
            widths = np.fromiter(modes_lib.keys(), dtype=float) * self.dh
            K_map = []
            K_input = []
            feasible_dis = self.gen_feasible_dis()
            for hi in tqdm(widths):
                for hj in widths:
                    for dis in feasible_dis:
                        
                        K_input.append([hi,hj,dis[0], dis[1]])
                        K_map_modes = []
                        for mi in range(self.modes):
                            for mj in range(self.modes):
                                kij = self.cal_k(modes_lib, mi, mj, hi, hj, dis)
                                K_map_modes.append(kij)
                        K_map.append(K_map_modes)
            K_map = np.array(K_map)
            K_input = np.array(K_input)
            print("K dataset generated. dataset size: " + str(K_map.shape[0]))
            np.save(self.path + "K_map.npy", K_map)
            np.save(self.path + "K_input.npy", K_input)
            print("K dataset saved.")
        return K_input, K_map
    
    def gen_feasible_dis(self,):
        feasible_dis = []
        for i in range(-self.Knn, self.Knn + 1):
            for j in range(-self.Knn, self.Knn + 1):
                if np.sqrt(i**2 + j**2) <= self.Knn:
                    feasible_dis.append((i,j))
        feasible_dis.append((self.Knn, self.Knn))
        return feasible_dis
    
    def cal_k(self, modes_lib, mi, mj, hi, hj, dis):
        '''
            when fit K, to make h change continuiously, we need to increase the resolution. to make the step size = dh
            i, j is the index of waveguides.
            h: waveguide width
            m: mode
            dis = [iy - jy, ix - jx]: type: tuple.
        '''
        if dis == (self.Knn, self.Knn):
            return 0
        if dis == (0,0):
            return 0
        ds = dis[0] * self.upsample_res
        dt = dis[1] * self.upsample_res
        hi_index = h2index(hi, self.dh)
        hj_index = h2index(hj, self.dh)
        Eyi = modes_lib[hi_index][mi]['Ey']
        Eyj = modes_lib[hj_index][mj]['Ey']
        new_size = self.upsample_res * 2 * (self.Knn + 1)
        Eyi = resize_field2D(Eyi, new_size)
        Eyj = resize_field2D(Eyj, new_size)

        ds_max = self.upsample_res * self.Knn
        Eyi_ext = np.zeros((new_size + 2 * ds_max, new_size + 2 * ds_max))
        Eyj_ext = Eyi_ext.copy()
        delta_epsilon = Eyi_ext.copy()
        Eyi_ext[ds_max:ds_max + new_size, ds_max:ds_max + new_size] = Eyi
        Eyj_ext[ds_max - ds:ds_max - ds + new_size, ds_max - dt:ds_max - dt + new_size] = Eyj
        
        radius = int(round(hj/2/self.dh))
        c1 = int(round(ds_max + new_size//2 - ds))
        c2 = int(round(ds_max + new_size//2 - dt))
        delta_epsilon[c1 - radius: c1 + radius, c2 - radius: c2 + radius] = (self.n_wg**2 - self.n0**2)
        k_out = self.k * self.C_EPSILON *  np.sum(delta_epsilon * Eyi_ext * Eyj_ext) * self.dh**2
        return k_out
