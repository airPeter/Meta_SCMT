import numpy as np
import torch
import torch.nn as nn
from .utils import Model
from scipy import special
from .SCMT_model_1D import freespace_layer

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
        self.h_paras = torch.nn.Parameter(torch.empty((N,), dtype = torch.float))
        freelayers = []
        genphases = []
        for idx, lam in enumerate(self.GP.lams):
            freelayers.append(freespace_layer(self.prop, lam, total_size, GP.dx))
            paras = np.load(self.GP.paths[idx] + "PBA_paras.npy", allow_pickle= True)
            paras = paras.item()
            genphases.append(gen_Phase(nodes = paras['nodes'], layers = paras['layers']))
        self.freelayers = nn.ModuleList(freelayers)
        self.genphases = nn.ModuleList(genphases)
        
        self.near_field = near_field
    def forward(self, E0):
        self.hs = self.sig(self.h_paras) * (self.h_max - self.h_min) + self.h_min
        E_outs = []
        for idx, genphase in enumerate(self.genphases):
            tmp_phase = genphase(self.hs.view(-1, 1))
            tmp_phase =tmp_phase.view(1, 1, -1)
            tmp_phase = torch.nn.functional.interpolate(tmp_phase, size=(self.GP.res * self.N), mode='linear',align_corners=False)
            pad_size = (self.total_size - self.GP.res * self.N)
            pad1 = pad_size//2
            pad2 = pad_size - pad1
            tmp_phase = torch.nn.functional.pad(tmp_phase, pad = (pad1, pad2), mode = 'replicate')
            tmp_phase = tmp_phase.view(-1,)
            E = E0 * torch.exp(1j * tmp_phase)
            if self.near_field:
                E_outs.append(E)
            else:
                Ef = self.freelayers[idx](E)
                If = torch.abs(Ef)**2
                E_outs.append(If)
        return E_outs

    def reset(self):
        torch.nn.init.constant_(self.h_paras, val = 0.0)
        if len(self.GP.paths) != len(self.GP.lams):
            raise Exception("number of paths != number of lams.")
        for i, path in enumerate(self.GP.paths):
            self.genphases[i].reset(path)
        
class gen_Phase(nn.Module):
    def __init__(self, layers, nodes):
        super(gen_Phase, self).__init__()
        self.cnn = Model(1, 1, layers= layers, nodes = nodes).requires_grad_(requires_grad=False)
        
    def forward(self, widths):
        '''
        input:
            widths: shape [-1, 1]
        '''
        phase = self.cnn(widths)
        return phase
    def reset(self, path):
        model_state = torch.load(path + "fitting_PBA_state_dict")
        self.cnn.load_state_dict(model_state)
