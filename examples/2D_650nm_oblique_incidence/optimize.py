import sys
module_path = '/home/zhicheng/cmt_design_metasurface2.0/'
sys.path.insert(1, module_path)
import Meta_SCMT
import numpy as np
import matplotlib.pyplot as plt

GP = {
    'dim' : 2, #dim : 1 or 2.
    'modes' : 1, #number of modes with in a single waveguide. modes <= 2 is usually good enough.
    'period' : 0.4,
    'res' : 20, #resolution within one period
    'downsample_ratio': 0.5, #between (0, 1] for the output periodic resolution, it will be int(round(downsample_ratio * res))
    'wh' : 0.8, #waveguide height
    'lam' : 0.65,
    'n_sub' : 1.46, #the refractive index of substrate.
    'n_wg' : 2.27,# the refractive index of waveguide
    'h_min' : 0.16, #h_min, and h_max define the range of the width of waveguide.
    'h_max' : 0.36,
    'dh' : 0.01, #the step size of h.
    'path' : "sim_cache_650nm/", #the inter state store path            
}
sim = Meta_SCMT.Sim(**GP)

N = 100
NA = 0.8
prop_dis = 0.5 * N * GP['period'] * np.sqrt((1 - NA**2)/NA**2)
PBA_widths = sim.PBA.design_lens(N, prop_dis, load = True)
# wgs = np.random.uniform(GP['h_min'], GP['h_max'], (N, N))
# wgs_name = np.random.randint(0, 10, size = 6)
# wgs_name = [str(x) for x in wgs_name]
# wgs_name = ''.join(wgs_name)
# np.save("random_widths_N" + str(N) + "_" + wgs_name + ".npy", wgs)
sim.scmt.init_model(N, prop_dis, APPROX= 1, Ni = 11 * N, COUPLING = True, devs = ["cuda"], init_hs= PBA_widths, far_field= True)
sim.scmt.optimize(notes = 'May14_N' + str(N) + 'init_PBA', steps = 200)