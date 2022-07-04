import sys
module_path = 'C:/Users/94735/OneDrive - UW-Madison/My Projects/CMT_and_inverse_design/Meta_SCMT'
sys.path.insert(1, module_path)
import Meta_SCMT
import numpy as np
import math
GP = {
    'dim' : 1, #dim : 1 or 2.
    'modes' : 1, #number of modes with in a single waveguide. modes <= 2 is usually good enough.
    'period' : 0.28,
    'res' : 20, #resolution within one period
    'downsample_ratio': 1, #between (0, 1] for the output periodic resolution, it will be int(round(downsample_ratio * res))
    'wh' : 0.6, #waveguide height
    'lam' : 0.66,
    'n_sub' : 1.46, #the refractive index of substrate.
    'n_wg' : 2.4,# the refractive index of waveguide
    'h_min' : 0.06, #h_min, and h_max define the range of the width of waveguide.
    'h_max' : 0.27,
    'dh' : 0.01, #the step size of h.
    'path' : "sim_cache_TiO2_1mode/", #the inter state store path            
}
sim = Meta_SCMT.Sim(**GP)

N = 200
theta_deg = 45
# for NA = 0.8 max theta = 53deg.
#theta = (math.radians(-theta_deg), math.radians(theta_deg))
theta = math.radians(theta_deg)
# theta_deg = 0
# theta = math.radians(theta_deg)
NA = 0.8
prop_dis = 0.5 * N * GP['period'] * np.sqrt((1 - NA**2)/NA**2)
wgs = sim.PBA.design_lens(N, prop_dis, load = True, vis = False)

sim.scmt.init_model(N, prop_dis, COUPLING = True, init_hs= wgs, far_field= True)
#E_out =sim.scmt.forward(theta = theta)
#sim.scmt.vis_field(E_out)
theta = (math.radians(-theta_deg), math.radians(theta_deg))
sim.scmt.optimize(notes = 'Apr29_N' + str(N) + 'None_init' + 'theta_' + str(theta_deg), lr = 0.01, steps = 2000, theta = theta)