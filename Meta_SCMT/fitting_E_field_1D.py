import matplotlib.pyplot as plt
import numpy as np
import torch
from .utils import Model, train

class Fitting_E_field_1D():
    def __init__(self, gen_modes, modes, res, dh, dx, Knn, path) -> None:
        self.gen_modes = gen_modes
        self.res = res
        self.dx = dx
        self.dh = dh
        self.Knn = Knn
        self.modes = modes
        self.model = None
        self.path = path
        
    def fit(self, layers = 4, steps = 10000, lr = 0.001, vis = True):
        modes_lib = self.gen_modes.modes_lib
        if modes_lib == None:
            raise Exception("gen modes first!")
        X, Y, size_Ey = gen_fitting_data(self.modes, modes_lib, self.dh)
        self.model = Model(1, Y.shape[-1], layers= layers, nodes = 128)
        batch_size = X.shape[0]
        Y_pred = train(self.model, X, Y, steps, lr, batch_size)
        torch.save(self.model.state_dict(), self.path + "fitting_E_state_dict")
        print("model saved.")
        if vis:
            indexs = np.random.randint(0, Y.shape[0], size = (3,))
            Y_pred = Y_pred.reshape(-1, self.modes, size_Ey)
            Y = Y.reshape(-1, self.modes, size_Ey)
            x_plot = (np.arange(0, size_Ey) - size_Ey//2) * self.dx
            for idx in indexs:
                plt.figure()
                for mode in range(self.modes):
                    plt.plot(x_plot, Y_pred[idx, mode], label = "mode:" + str(mode))
                    plt.plot(x_plot, Y[idx, mode], linestyle = '--', label = "mode:" + str(mode))
                    plt.legend()
                plt.xlabel("[um]")
                plt.ylabel("Ey")
                plt.show()
        return None
    
def gen_fitting_data(modes, modes_lib, dh):
    X = []
    Y = []
    for key in modes_lib.keys():
        h = key * dh
        X.append(h)
        y = []
        for m in range(modes):
            y.append(modes_lib[key][m]['Ey'])
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    size_Ey = Y.shape[-1]
    X = X.reshape((X.shape[0], 1))
    Y = Y.reshape((Y.shape[0], -1))
    widths = X
    Eys = Y
    return widths, Eys, size_Ey
    
    