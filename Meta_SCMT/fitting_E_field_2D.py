import matplotlib.pyplot as plt
import numpy as np
import torch
from .utils import Model, train

class Fitting_E_field_2D():
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
    
    