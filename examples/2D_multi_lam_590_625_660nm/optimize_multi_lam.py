import sys
module_path = '/home/zhicheng/cmt_design_metasurface2.0/'
sys.path.insert(1, module_path)
import Meta_SCMT
import numpy as np
import matplotlib.pyplot as plt

GP = {
    'dim' : 2, #dim : 1 or 2.
    'modes' : 1, #number of modes with in a single waveguide. modes <= 2 is usually good enough.
    'period' : 0.32,
    'res' : 20, #resolution within one period
    'downsample_ratio': 0.5, #between (0, 1] for the output periodic resolution, it will be int(round(downsample_ratio * res))
    'wh' : 0.6, #waveguide height
    'lams' : [0.66, 0.59, 0.532],
    'n_sub' : 1, #the refractive index of substrate.
    'n_wg' : 2.4,# the refractive index of waveguide
    'h_min' : 0.12, #h_min, and h_max define the range of the width of waveguide.
    'h_max' : 0.31,
    'dh' : 0.005, #the step size of h.
    'paths' : ["sim_cache_TiO2_lam660nm/", "sim_cache_TiO2_lam590nm/", "sim_cache_TiO2_lam532nm/"], #the inter state store path            
}
sim = Meta_SCMT.SimLam(**GP)

N = 200
NA = 0.8
prop_dis = 0.5 * N * GP['period'] * np.sqrt((1 - NA**2)/NA**2)
wgs = np.load("PBA_widths_N" + str(N) + "lam_590.npy")
sim.scmt.init_model(N, prop_dis, APPROX = 1, COUPLING = True, devs = ["cuda:1", "cuda:0", "cuda:2", "cuda:3"], init_hs= wgs, far_field= True)
sim.scmt.optimize(notes = 'Apr25_N' + str(N) + 'PBA_590nm_init_minmax', lr = 0.1, steps = 350, minmax = True)