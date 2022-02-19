import matplotlib.pyplot as plt
import numpy as np
import torch
from .utils import h2index, Model, train
from tqdm import tqdm
import os

class Fitting_C_matrix_1D():
    def __init__(self, gen_modes, modes, res, dh, dx, Knnc, path) -> None:
        self.gen_modes = gen_modes
        self.res = res
        self.dx = dx
        self.dh = dh
        self.Knnc = Knnc
        self.modes = modes
        self.channels = self.modes**2
        self.model = None
        self.path = path
        
    def fit(self, layers = 6, steps = 500, lr = 0.001, vis = True, load = True):
        X, Y = self.gen_fitting_data(load)
        self.model = Model(3, self.channels, layers= layers, nodes = 64)
        batch_size = 512
        Y_pred = train(self.model, X, Y, steps, lr, batch_size)
        torch.save(self.model, self.path + "fitting_C_state_dict")
        print("model saved.")
        if vis:
            Y_pred = Y_pred.reshape(-1, self.Knnc * 2 + 1, self.channels)
            Y = Y.reshape(-1, self.Knnc * 2 + 1, self.channels)
            for dis in range(-self.Knnc, self.Knnc + 1):
                plt.figure()
                for ch in range(self.channels):
                    dis_index = dis + self.Knnc
                    plt.plot( Y_pred[:, dis_index, ch], label = "ch:" + str(ch))
                    plt.plot(Y[:, dis_index, ch], linestyle = '--', label = "ch:" + str(ch))
                    plt.legend()
                plt.xlabel("vary widths" + "dis:" + str(dis))
                plt.ylabel("Cij")
                plt.show()
        return None
    
    def gen_fitting_data(self,load):
        '''
            output:
            C_input: shape: [widths * widths]
            C_map: shape: [widths * widths, modes**2]
        '''
        map_path  = self.path + "C_map.npy"
        input_path = self.path + "C_input.npy"
        if load:
            if os.path.exists(map_path) and os.path.exists(input_path):
                C_map = np.load(map_path)
                C_input = np.load(input_path)
            else:
                raise Exception("C map, C_input not generated. set load to false")
        else:
            modes_lib = self.gen_modes.modes_lib
            if modes_lib == None:
                raise Exception("gen modes first!")
            widths = np.fromiter(modes_lib.keys(), dtype=float) * self.dh
            C_map = []
            C_input = []
            for hi in tqdm(widths):
                for hj in widths:
                    for dis in range(-self.Knnc, self.Knnc + 1):
                        dis_norm = dis / self.Knnc
                        C_input.append([hi,hj,dis_norm])
                        C_map_modes = []
                        for mi in range(self.modes):
                            for mj in range(self.modes):
                                cij = self.cal_c(modes_lib, mi, mj, hi, hj, dis)
                                C_map_modes.append(cij)
                        C_map.append(C_map_modes)
            C_map = np.array(C_map)
            C_input = np.array(C_input)
            print("C dataset generated. dataset size: " + str(C_map.shape[0]))
            np.save(self.path + "C_map.npy", C_map)
            np.save(self.path + "C_input.npy", C_input)
            print("C dataset saved.")
        return C_input, C_map

    def cal_c(self, modes_lib, mi, mj, hi, hj, dis):
        '''
            i, j is the index of waveguides.
            h: waveguide width
            m: mode
            dis = i - j: -Knnc, -Knnc - 1, ..., 0, 1, ... Knnc
        '''
        dis = dis * self.res
        hi_index = h2index(hi, self.dh)
        hj_index = h2index(hj, self.dh)
        Ey = modes_lib[hi_index][mi]['Ey']
        Hx = modes_lib[hj_index][mj]['Hx']
        if dis < 0:
            #np.pad(a, (2, 3), 'linear_ramp', end_values=(5, -4))
            Ey = np.pad(Ey, (0, -dis), 'constant', constant_values = (0, 0))
            Hx = np.pad(Hx, (-dis, 0), 'constant', constant_values = (0, 0))
        elif dis > 0:
            Ey = np.pad(Ey, (dis, 0), 'constant', constant_values = (0, 0))
            Hx = np.pad(Hx, (0, dis), 'constant', constant_values = (0, 0))
        c_out = - 2 * np.sum(Ey * Hx) * self.dx
        if dis == 0:
            c_out = 1
        return c_out
