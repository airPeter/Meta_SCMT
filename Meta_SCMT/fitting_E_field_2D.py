import matplotlib.pyplot as plt
import numpy as np
import torch
from .utils import Model, train, resize_field2D

class Fitting_E_field_2D():
    def __init__(self, gen_modes, modes, out_res, dh, dx, Knn, path) -> None:
        self.gen_modes = gen_modes
        self.out_res = out_res
        self.dx = dx
        self.dh = dh
        self.Knn = Knn
        self.modes = modes
        self.model = None
        self.path = path 
        
    def fit(self, layers = 4, nodes = 128, steps = 10000, lr = 0.001, vis = True, save_fig = False):
        modes_lib = self.gen_modes.modes_lib
        if modes_lib is None:
            raise Exception("gen modes first!")
        out_size = 2 * (self.Knn + 1) * self.out_res
        X, Y, size_Ey = gen_fitting_data(self.modes, modes_lib, self.dh, out_size)
        self.model = Model(1, Y.shape[-1], layers= layers, nodes = nodes)
        batch_size = X.shape[0]
        Y_pred = train(self.model, X, Y, steps, lr, batch_size)
        torch.save(self.model.state_dict(), self.path + "fitting_E_state_dict_outres_" + str(self.out_res))
        print("model saved.")
        E_paras = {'nodes': nodes, 'layers': layers}
        np.save(self.path + "E_paras.npy", E_paras)
        if vis:
            indexs = np.random.randint(0, Y.shape[0], size = (3,))
            Y_pred = Y_pred.reshape(-1, self.modes, size_Ey, size_Ey)
            Y = Y.reshape(-1, self.modes, size_Ey, size_Ey)
            for idx in indexs:
                for mode in range(self.modes):
                    fig, axs = plt.subplots(1, 2, figsize = (12, 6))
                    Ey = Y[idx, mode]
                    Ey_pred = Y_pred[idx, mode]
                    L = "idx:" + str(idx) + "mode:" + str(mode) + "Ey"
                    plot0 = axs[0].imshow(Ey)
                    axs[0].set_title(L)
                    plt.colorbar(plot0, ax = axs[0])
                    L = "idx:" + str(idx) + "mode:" + str(mode) + "Ey_pred"
                    plot1 = axs[1].imshow(np.real(Ey_pred))
                    axs[1].set_title(L)
                    plt.colorbar(plot1, ax = axs[1])
                    if not save_fig:
                        plt.show()
                    else:
                        plt.savefig(self.path + "fit_Ey_" + "idx:" + str(idx) + "mode:" + str(mode) + ".png")
        return None
    
def gen_fitting_data(modes, modes_lib, dh, out_size):
    X = []
    Y = []
    for key in modes_lib.keys():
        h = key * dh
        X.append(h)
        y = []
        for m in range(modes):
            Ey = modes_lib[key][m]['Ey']
            Ey = resize_field2D(Ey, out_size)
            y.append(Ey)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    size_Ey = Y.shape[-1]
    X = X.reshape((X.shape[0], 1))
    Y = Y.reshape((Y.shape[0], -1))
    widths = X
    Eys = Y
    return widths, Eys, size_Ey
    
    