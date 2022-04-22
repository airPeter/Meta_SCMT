'''
    multi wavelength SCMT model.
'''
import numpy as np
import os
from .SCMT_1D_lam import SCMT_1D
from .PBA_1D_lam import PBA_1D
class GP():
    def __init__(self,dim, modes, period, res, downsample_ratio, wh, lams, n_sub, n_wg, h_min, h_max, dh, paths):
        self.dim = dim #dim = 1 or 2.
        self.modes = modes #number of modes with in a single waveguide. modes <= 2 is usually good enough.
        self.C_EPSILON = 3 * 8.85 * 10**-4 # C * EPSILON
        self.Knn = 2 #number of nearest neighbors for the C and K matrix.
        self.period = period
        self.res = res #resolution within one period
        if dim == 1 and downsample_ratio != 1:
            raise Exception("only support downsample_ratio = 1 for dim = 1.")
        self.downsample_ratio = downsample_ratio
        self.out_res = int(round(self.downsample_ratio * self.res))
        self.dx = self.period/self.res
        self.wh = wh #waveguide height
        self.lams = lams # list
        self.n_sub = n_sub #the refractive index of substrate.
        self.n_wg = n_wg# the refractive index of waveguide
        self.n0 = 1 #the refractive index of air.
        self.h_min = h_min #h_min and h_max define the range of the width of waveguide.
        self.h_max = h_max
        self.dh = dh #the step size of h.
        self.paths = paths # a list of path 
        for tmp_path in self.paths:           
            if not os.path.exists(tmp_path):
                raise Exception(tmp_path, " not exist. gen the cache using single lam model first. then run the multi lam model.")
    def __eq__(self, other) : 
        return self.__dict__ == other.__dict__
    
class SimLam():
    def __init__(self,**keyword_args) -> None:
        self.GP = GP(**keyword_args)
        if self.GP.dim == 1:
            self.scmt = SCMT_1D(self.GP)
            self.pba_opt = PBA_1D(self.GP)

