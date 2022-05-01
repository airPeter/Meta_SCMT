'''design metasurface by modeling the meta unit as waveguide. 
    the coupling between waveguide is modeled by spacial couple mode theory (SCMT).
    currently, only support S wave, the polarization is vertical to the incident plane.
    note:
        the unit is [um].
        waveguide only has TE mode. (Ey, and Hx are non zero, other polarizations are zero.
        the wave propagation direction is z direction.
        the incident direction is within x-z plane.
        the incident angle theta is the angle between incident direction and the z direction.
        the 1D waveguide (slabs) modes are calculated analytically, only for quick idea validation.
        the 2D waveguide (square rod) modes are claculated numerically using Tidy3d.
        the forward and backward are implemented using pytorch.
        the number of waveguide for one side is N, for 1D, number of waveguides in metasurface is N; for 2D, N^2.
        the problem size is propotional to total_num_waveguides^3 * modes_within_each_waveguide^2.'''
import numpy as np
import os
import warnings
import time 
from .modes1D import Gen_modes1D
from .fitting_neffs import Fitting_neffs
from .fitting_C_matrix_1D import Fitting_C_matrix_1D
from .fitting_E_field_1D import Fitting_E_field_1D
from .fitting_K_matrix_1D import Fitting_K_matrix_1D
from .SCMT_1D import SCMT_1D
from .fullwave_1D import Fullwave_1D
from .modes2D import Gen_modes2D
from .fitting_C_matrix_2D import Fitting_C_matrix_2D
from .fitting_E_field_2D import Fitting_E_field_2D
from .fitting_K_matrix_2D import Fitting_K_matrix_2D
from .SCMT_2D import SCMT_2D
from .fullwave_2D import Fullwave_2D
from .PBA_design import PBA
from .PBA_1D import PBA_1D
from .PBA_2D import PBA_2D
from .ideal_meta import Ideal_meta
from .ideal_meta_1D import Ideal_meta_1D
class GP():
    def __init__(self,dim, modes, period, res, downsample_ratio, wh, lam, n_sub, n_wg, h_min, h_max, dh, path = 'sim_cache/'):
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
        self.lam = lam
        self.k = 2 * np.pi / lam
        self.n_sub = n_sub #the refractive index of substrate.
        self.n_wg = n_wg# the refractive index of waveguide
        self.n0 = 1 #the refractive index of air.
        self.h_min = h_min #h_min and h_max define the range of the width of waveguide.
        self.h_max = h_max
        self.dh = dh #the step size of h.
        self.path = path #the inter state store path            
        if not os.path.exists(path):
            os.mkdir(path)
    def __eq__(self, other) : 
        return self.__dict__ == other.__dict__
    
class Sim():
    def __init__(self,**keyword_args) -> None:
        self.GP = GP(**keyword_args)
        gp_path = self.GP.path + "GP.npy"
        if not os.path.exists(gp_path):
            np.save(gp_path, self.GP, allow_pickle= True)
        else:
            saved_GP = np.load(gp_path, allow_pickle= True)
            saved_GP = saved_GP.item()
            if not self.GP == saved_GP:
                warnings.warn('Your global parameters have changed. be careful loading any cached data, it may be in consist!')
                print("If dowmsample_ratio changed, it only need to rerun fitting E.")
                print("It is ok to change n_sub, which is decoupled with the cached data.")
                time.sleep(3)
        if self.GP.dim == 1:
            self.gen_modes = Gen_modes1D(self.GP)
            #always pass the object instead of the data until you realy need it. So that the data is up to date.
            self.fitting_neffs = Fitting_neffs(self.GP.modes, self.gen_modes, self.GP.dh, self.GP.path)
            self.fftting_C = Fitting_C_matrix_1D(self.gen_modes, self.GP.modes, self.GP.res, self.GP.dh, self.GP.dx, self.GP.Knn, self.GP.path)
            self.fftting_E = Fitting_E_field_1D(self.gen_modes, self.GP.modes, self.GP.res, self.GP.dh, self.GP.dx, self.GP.Knn, self.GP.path)
            self.fftting_K = Fitting_K_matrix_1D(self.gen_modes, self.GP.modes, self.GP.res, self.GP.dh, self.GP.dx, self.GP.Knn, self.GP.path,
                                                self.GP.n_wg, self.GP.n0, self.GP.k, self.GP.C_EPSILON, self.GP.period)
                    
            self.scmt = SCMT_1D(self.GP)
            self.fullwave = Fullwave_1D(self.GP)
            self.PBA = PBA(self.GP, 1)
            self.pba_opt = PBA_1D(self.GP)
            self.ideal_meta = Ideal_meta_1D(self.GP)
        if self.GP.dim == 2:
            if self.GP.modes > 1:
                raise Exception("Currently, only support modes = 1 for 2 dimensional waveguide.")
            self.gen_modes = Gen_modes2D(self.GP)
            self.fitting_neffs = Fitting_neffs(self.GP.modes, self.gen_modes, self.GP.dh, self.GP.path)
            self.fftting_C = Fitting_C_matrix_2D(self.gen_modes, self.GP.modes, self.GP.res, self.GP.dh, self.GP.dx, self.GP.Knn, self.GP.path)
            self.fftting_E = Fitting_E_field_2D(self.gen_modes, self.GP.modes, self.GP.out_res, self.GP.dh, self.GP.dx, self.GP.Knn, self.GP.path)
            self.fftting_K = Fitting_K_matrix_2D(self.gen_modes, self.GP.modes, self.GP.res, self.GP.dh, self.GP.dx, self.GP.Knn, self.GP.path,
                                                self.GP.n_wg, self.GP.n0, self.GP.k, self.GP.C_EPSILON, self.GP.period)
            self.scmt = SCMT_2D(self.GP)
            self.fullwave = Fullwave_2D(self.GP)
            self.PBA = PBA(self.GP, 2)
            self.pba_opt = PBA_2D(self.GP)
            self.ideal_meta = Ideal_meta(self.GP)
