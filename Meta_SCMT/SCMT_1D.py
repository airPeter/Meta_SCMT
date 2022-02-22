import numpy as np
import matplotlib.pyplot as plt
import os
from .SCMT_model_1D import metalayer
import torch
import torch.nn as nn
from torch import optim

class SCMT_1D():
    def __init__(self, GP):
        self.GP = GP
        self.model = None
        self.APPROX = None
        self.COUPLING = None
        self.Euler_steps = None
        self.N = None
        self.Ni = None
        self.k_row = None
        
    def init_model(self, N, COUPLING = True, layer_neff = 2, layer_C = 6, layer_K = 6, layer_E = 4, init_h_paras = None, far_field = False):
        '''
            the layers will be used when re building the fitted model. If you change any of this default values when you do the fitting.
            you should also change at here.
        '''
        self.N = N
        self.total_size = (self.N + 2 * (self.GP.Knn + 1)) * self.GP.res
        # if not Ni:
        #     Ni = 5 * N
        # if not k_row:
        #     k_row = N
        # self.APPROX = APPROX
        self.COUPLING = COUPLING
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = metalayer(self.GP, COUPLING, N, layer_neff, layer_C, layer_K, layer_E)
        self.init_paras(self.model, self.GP.path, init_h_paras)
        self.model = self.model.to(self.device)
        return None
    
    def forward(self, theta = 0):
        #incident field plane wave
        E0 = (torch.ones((self.total_size), dtype = torch.complex128)) * 10
        E0 = E0.to(self.device)
        En = self.model(E0)
        En = En.cpu().detach().numpy()
        return En
    
    def optimize(self, step, lr):
        
        return None

    def init_paras(self, model, cache_path, h_paras = None):
        model.reset(cache_path)
        if h_paras is None:
            print('initialized by default h_paras.')
            return None
        else:
            #h_paras = np.genfromtxt(path, delimiter=',')
            h_paras_initial = torch.tensor(h_paras, dtype = torch.float)
            state_dict = model.state_dict()
            state_dict['h_paras'] = h_paras_initial
            model.load_state_dict(state_dict)
            #with torch.no_grad():
            #    model.matalayer1.h_paras.data = h_paras_initial
            print('initialized by loaded h_paras.')
            return None 
    def vis_field(self, E):
        fig, axs = plt.subplots(1, 2, figsize = (12, 6))
        px = (np.arange(self.total_size) - self.total_size//2) * self.GP.dx
        plot1 = axs[0].plot(px, np.angle(E), label = "phase")
        axs[0].legend()
        plot2 = axs[1].plot(px, np.abs(E), label = "amp")
        axs[1].legend()
        plt.show()
        return None
    