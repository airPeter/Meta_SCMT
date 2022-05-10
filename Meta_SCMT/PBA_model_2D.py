import numpy as np
import torch
import torch.nn as nn
from .PBA_model_1D import gen_Phase
from .SCMT_model_2D import freespace_layer

class PBA_model(nn.Module):
    def __init__(self, prop_dis, GP, N, total_size, near_field = True):
        super(PBA_model, self).__init__()
        self.prop = prop_dis
        self.GP = GP
        self.N = N
        self.total_size = total_size
        self.h_min = GP.h_min
        self.h_max = GP.h_max
        self.sig = torch.nn.Sigmoid()
        self.h_paras = torch.nn.Parameter(torch.empty((N,N), dtype = torch.float))
        self.freelayer1 = freespace_layer(self.prop, self.GP.lam, total_size, self.GP.period/self.GP.out_res)
        paras = np.load(self.GP.path + "PBA_paras.npy", allow_pickle= True)
        paras = paras.item()
        self.genphase = gen_Phase(nodes = paras['nodes'], layers = paras['layers'])
        self.near_field = near_field
    def forward(self, E0):
        self.hs = self.sig(self.h_paras) * (self.h_max - self.h_min) + self.h_min
        self.phase = self.genphase(self.hs.view(-1, 1))
        self.phase =self.phase.view(1, 1, self.N, self.N)
        self.phase = torch.nn.functional.interpolate(self.phase, size=(self.GP.out_res * self.N, self.GP.out_res * self.N), mode='bilinear',align_corners=False)
        # pad_size = (self.total_size - self.GP.out_res * self.N)
        # pad1 = pad_size//2
        # pad2 = pad_size - pad1
        # self.phase = torch.nn.functional.pad(self.phase, pad = (pad1, pad2, pad1, pad2), mode = 'replicate')
        self.phase = self.phase.view(self.total_size, self.total_size)
        E = E0 * torch.exp(1j * self.phase)
        if self.near_field:
            return E
        Ef = self.freelayer1(E)
        If = torch.abs(Ef)**2
        return If
    def reset(self, path):
        torch.nn.init.constant_(self.h_paras, val = 0.0)
        self.genphase.reset(path)