'''
    fit a Fully connect met, that take (hi, hj, dis/self.Knn) as input, output Kij for each channels. 
    number of channel equals to modes**2.
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import k_means
import torch
from .utils import h2index, Model, train
from tqdm import tqdm
import os

class Fitting_K_matrix_1D():
    def __init__(self, gen_modes, modes, res, dh, dx, Knn, path, n_wg, n0, k, C_EPSILON, period) -> None:
        self.gen_modes = gen_modes
        self.res = res
        self.dx = dx
        self.dh = dh
        self.Knn = Knn
        self.modes = modes
        self.channels = self.modes**2
        self.model = None
        self.path = path
        self.n_wg = n_wg
        self.n0 = n0
        self.k = k
        self.C_EPSILON = C_EPSILON
        self.period = period
        
    def fit(self, layers = 6, steps = 1000, lr = 0.001, vis = True, load = True):
        X, Y = self.gen_fitting_data(load)
        self.model = Model(3, self.channels, layers= layers, nodes = 64)
        batch_size = 512
        Y_pred = train(self.model, X, Y, steps, lr, batch_size)
        torch.save(self.model, self.path + "fitting_K_state_dict")
        print("model saved.")
        if vis:
            Y_pred = Y_pred.reshape(-1, self.Knn * 2 + 1, self.channels)
            Y = Y.reshape(-1, self.Knn * 2 + 1, self.channels)
            for dis in range(-self.Knn, self.Knn + 1):
                plt.figure()
                for ch in range(self.channels):
                    dis_index = dis + self.Knn
                    plt.plot( Y[:, dis_index, ch], label = "ch:" + str(ch))
                    plt.plot(Y_pred[:, dis_index, ch], linestyle = '--', label = "ch:" + str(ch))
                    plt.legend()
                plt.xlabel("vary widths" + "dis:" + str(dis))
                plt.ylabel("Kij")
                plt.show()
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
            for hi in tqdm(widths):
                for hj in widths:
                    for dis in range(-self.Knn, self.Knn + 1):
                        dis_norm = dis / self.Knn
                        K_input.append([hi,hj,dis_norm])
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

    def cal_k(self, modes_lib, mi, mj, hi, hj, dis):
        '''
            when fit K, to make h change continuiously, we need to increase the resolution. to make the step size = dh
            i, j is the index of waveguides.
            h: waveguide width
            m: mode
            dis = i - j: -Knn, -Knn - 1, ..., 0, 1, ... Knn
        '''
        if dis == 0:
            return 0
        hi_index = h2index(hi, self.dh)
        hj_index = h2index(hj, self.dh)
        Eyi = modes_lib[hi_index][mi]['Ey']
        Eyj = modes_lib[hj_index][mj]['Ey']
        #increase resolution.
        xp = np.arange(0, Eyj.size * self.dx, self.dx)
        x_eval = np.arange(0, Eyj.size * self.dx, self.dh)
        Eyi = np.interp(x_eval, xp, Eyi)
        Eyj = np.interp(x_eval, xp, Eyj)
        delta_epsilon = np.zeros((Eyj.size))
        center = int(round(Eyj.size//2))
        radius = int((hj/2)//self.dh)
        delta_epsilon[center - radius: center + radius] = (self.n_wg**2 - self.n0**2)
        dis = dis * int(round(self.period/self.dh))
        if dis < 0:
            #np.pad(a, (2, 3), 'linear_ramp', end_values=(5, -4))
            Eyi = np.pad(Eyi, (0, -dis), 'constant', constant_values = (0, 0))
            Eyj = np.pad(Eyj, (-dis, 0), 'constant', constant_values = (0, 0))
            delta_epsilon = np.pad(delta_epsilon, (-dis, 0), 'constant', constant_values = (0, 0))
        elif dis > 0:
            Eyi = np.pad(Eyi, (dis, 0), 'constant', constant_values = (0, 0))
            Eyj = np.pad(Eyj, (0, dis), 'constant', constant_values = (0, 0))
            delta_epsilon = np.pad(delta_epsilon, (0, dis), 'constant', constant_values = (0, 0))

        k_out = self.k * self.C_EPSILON * np.sum(delta_epsilon * Eyi * Eyj) * self.dh
        return k_out
