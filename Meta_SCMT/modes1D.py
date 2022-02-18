import numpy as np
from scipy.optimize import fsolve
from .utils import h2index

def gen_modes1D(GP):
    H = np.arange(GP.h_min, GP.h_max + GP.dh, GP.dh)
    modes_lib = {}
    for h in H:
        h_index = h2index(h, GP.dh)
        modes_lib[h_index] = {}
        roots = auto_root_finder(GP.k, h, GP.n0, GP.n_wg)
        count_roots = len(roots)
        #define the span of field
        #set the calculated field with the middle of the waveguide as center
        h_shift = h/2
        half_x = (GP.Knnc + 1)*GP.period
        Xc = np.linspace(-half_x - h_shift, half_x - h_shift, 2*(GP.Knnc + 1) * GP.res)
        delta_x = GP.period / GP.res
        for n_mode in range(GP.modes):
            modes_lib[h_index][n_mode] = {}
            if n_mode < count_roots:
                beta = roots[n_mode]
                modes_lib[h_index][n_mode]['neff'] = np.round(beta/GP.k, 3)
                Ey = gen_Ey(Xc, beta, GP.k, h, GP.n0, GP.n_wg)
                Hx = gen_Hx(Xc, beta, GP.k, h, GP.n0, GP.n_wg, GP.C_EPSILON)
                normalization = np.sqrt(- 2 * np.sum(Ey * Hx) * delta_x)
                modes_lib[h_index][n_mode]['Ey'] = Ey / normalization
                modes_lib[h_index][n_mode]['Hx'] = Hx / normalization
            else:
                modes_lib[h_index][n_mode]['neff'] = None
                modes_lib[h_index][n_mode]['Ey'] = np.zeros(Xc.shape)
                modes_lib[h_index][n_mode]['Hx'] = np.zeros(Xc.shape)  
    return modes_lib

def find_root(beta, h, k, n0, n1):
    '''
        n1 is the refractive index of the waveguide.
        n0 is ... the surrounding.
        k is the wavenumber of the input light
        beta is the propagate constant
    '''
    gamma = np.sqrt(beta**2 - (k*n0)**2)
    kai = np.sqrt((k*n1)**2 - beta**2)
    return np.tan(kai * h) - (2*gamma)/(kai * (1 - gamma**2/kai**2))

def auto_root_finder(k, h, n0, n1):
    '''
        return: roots [list]
    '''
    k_lowerbound = k * n0
    k_upperbound = k * n1
    #print("upper bound:", k_lowerbound, "lower bound:", k_upperbound)
    bata_inits = np.linspace(k_lowerbound * 0.9999, k_upperbound * 0.9999, 200)
    roots = set()
    threshold = 10**-8
    cannot_find_times = 0
    for bata_init in bata_inits:
        root = fsolve(find_root, args = (h, k, n0, n1), x0 = bata_init)
        if np.abs(find_root(root, h, k, n0, n1)) < threshold:
            roots.add(np.round(root[0],5))
        else:
            cannot_find_times += 1
    #print("number of times that cannot find root:", cannot_find_times)
    roots = list(roots)
    roots.sort(reverse=True)
    return roots        
      
def gen_Ey(X, beta, k, h, n0, n1):
    Ey = []
    for x in X:
        gamma = np.sqrt(beta**2 - (k*n0)**2)
        kai = np.sqrt((k*n1)**2 - beta**2)
        if x > 0:
            Ey_temp = np.exp(-gamma * x)
        elif x > -h:
            Ey_temp = np.cos(kai * x) - gamma/kai * np.sin(kai * x)
        else:
            Ey_temp = (np.cos(kai * h) + gamma/kai * np.sin(kai * h)) * np.exp(gamma * (x + h))
        Ey.append(Ey_temp)
    return np.array(Ey)

def gen_Hx(X, beta, k, h, n0, n1, C_EPSILON):
    Hx = []
    for x in X:
        constant = C_EPSILON / (beta * k)
        gamma = np.sqrt(beta**2 - (k*n0)**2)
        kai = np.sqrt((k*n1)**2 - beta**2)
        if x > 0:
            Hx_temp = -constant * ((n0 * k)**2  + gamma**2) * np.exp(-gamma * x)
        elif x > -h:
            Hx_temp = -constant * (((n1*k)**2 - kai**2) *np.cos(kai * x) + (gamma * kai - (n1*k)**2*gamma/kai) * np.sin(kai * x))
        else:
            Hx_temp = -constant * ((n0*k)**2 + gamma**2) * (np.cos(kai * h) + gamma/kai * np.sin(kai * h)) * np.exp(gamma * (x + h))    
        Hx.append(Hx_temp)
    return np.array(Hx)
