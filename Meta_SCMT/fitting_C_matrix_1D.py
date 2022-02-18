import matplotlib.pyplot as plt
import numpy as np
import torch
from .utils import h2index, Model, train
from tqdm import tqdm


class fitting_C_matrix_1D():
    def __init__(self, modes_lib, modes, res, dh, dx, Knnc) -> None:
        self.modes_lib = modes_lib
        self.res = res
        self.dx = dx
        self.dh = dh
        self.Knnc = Knnc
        self.modes = modes
        self.channels = self.modes**2
        self.model = None

    def fit(self,steps, lr):
        X, Y = self.gen_fitting_data()
        self.model = Model(3, self.channels, layers= 4)
        Y_pred = train(self.model, X, Y, steps, lr)
    def gen_fitting_data(self,):
        '''
            output:
            C_input: shape: [widths * widths]
            C_map: shape: [widths * widths, modes**2]
        '''
        widths = self.modes_lib.keys() * self.dh
        C_map = []
        C_input = []
        for hi in widths:
            for hj in widths:
                for dis in range(-self.Knnc, self.Knnc + 1):
                    dis_norm = dis / self.Knnc
                    C_input.append([hi,hj,dis_norm])
                    C_map_modes = []
                    for mi in range(self.modes):
                        for mj in range(self.modes):
                            cij = self.cal_c(mi, mj, hi, hj, dis)
                            C_map_modes.append(cij)
                    C_map.append(C_map_modes)
        C_map = np.array(C_map)
        C_input = np.array(C_input)
        return C_input, C_map

    def cal_c(self, mi, mj, hi, hj, dis):
        '''
            i, j is the index of waveguides.
            h: waveguide width
            m: mode
            dis = i - j: -Knnc, -Knnc - 1, ..., 0, 1, ... Knnc
        '''
        dis = dis * self.res
        hi_index = h2index(hi)
        hj_index = h2index(hj)
        Ey = self.modes_lib[hi_index][mi]['Ey']
        Hx = self.modes_lib[hj_index][mj]['Hx']
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
