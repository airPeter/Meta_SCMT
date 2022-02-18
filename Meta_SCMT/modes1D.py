import numpy as np
from scipy.optimize import fsolve
from .utils import h2index
import matplotlib.pyplot as plt
import os
import warnings
class Gen_modes1D():
    def __init__(self, GP):
        self.GP = GP
        self.modes_lib = None
    def gen(self,load = False):
        '''
            generate a dict that for each unique h, and mode, the neff, Ey, Hx are included.
        '''
        load_path = os.path.join(self.GP.path, "modes_lib.npy")
        if load:
            if not os.path.exists(load_path):
                raise Exception('gen modes first!')
            modes_lib = np.load(load_path, allow_pickle= True)
            modes_lib = modes_lib.item()
            print("modes lib load sucessed.")
            warnings.warn('You may change the physical setup without regenerate the modes!')
            #consistency check
            total_hs = (self.GP.h_max - self.GP.h_min + self.GP.dh)//self.GP.dh + 1
            load_total_hs = len(modes_lib.keys())
            if total_hs != load_total_hs:
                print("expected total waveguides:" + str(total_hs) + "loaded:" + str(load_total_hs))
                raise Exception('You indeed change the physical setup without regenerate the modes!')
            self.modes_lib = modes_lib
        else:
            GP = self.GP
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
                for n_mode in range(GP.modes):
                    modes_lib[h_index][n_mode] = {}
                    if n_mode < count_roots:
                        beta = roots[n_mode]
                        modes_lib[h_index][n_mode]['neff'] = np.round(beta/GP.k, 3)
                        Ey = gen_Ey(Xc, beta, GP.k, h, GP.n0, GP.n_wg)
                        Hx = gen_Hx(Xc, beta, GP.k, h, GP.n0, GP.n_wg, GP.C_EPSILON)
                        normalization = np.sqrt(- 2 * np.sum(Ey * Hx) * GP.dx)
                        modes_lib[h_index][n_mode]['Ey'] = Ey / normalization
                        modes_lib[h_index][n_mode]['Hx'] = Hx / normalization
                    else:
                        modes_lib[h_index][n_mode]['neff'] = GP.n0
                        modes_lib[h_index][n_mode]['Ey'] = np.zeros(Xc.shape)
                        modes_lib[h_index][n_mode]['Hx'] = np.zeros(Xc.shape)  
            self.modes_lib =  modes_lib
            np.save(load_path, self.modes_lib)
            print("generated modes lib saved at:" + load_path)
        return None
    
    def vis_field(self,H):
        '''
            H an list of wg width you want to plot.
        '''
        if self.modes_lib == None:
            raise Exception("gen modes first!")
        fig, axs = plt.subplots(1, 2, figsize = (12, 6))
        half_x = (self.GP.Knnc + 1)*self.GP.period
        Xc = np.linspace(-half_x, half_x, 2*(self.GP.Knnc + 1) * self.GP.res)
        for h in H:
            index = h2index(h, self.GP.dh)
            for n_mode in range(self.GP.modes):
                Ey = self.modes_lib[index][n_mode]['Ey']
                neff = self.modes_lib[index][n_mode]['neff']
                L = "h:" + str(round(h,3)) + "neff" + str(neff) + "mode:" + str(n_mode) + "Ey"
                axs[0].plot(Xc, Ey, label = L)
                axs[0].set_xlabel("[um]")
                Hx = self.modes_lib[index][n_mode]['Hx']
                L = "h:" + str(round(h,3)) + "neff" + str(neff)  + "mode:" + str(n_mode) + "Hx"
                axs[1].plot(Xc, Hx, label = L)
                axs[1].set_xlabel("[um]")
        axs[0].legend()
        axs[1].legend()
        plt.show()
        return None
    
    def vis_neffs(self,):
        '''
            plot h vs neff
        '''
        if self.modes_lib == None:
            raise Exception("gen modes first!")
        plt.figure()
        for mode in range(self.GP.modes):
            neffs = []
            widths = []
            for key in self.modes_lib.keys():
                widths.append(key * self.GP.dh)
                neffs.append(self.modes_lib[key][mode]['neff'])
            plt.plot(widths, neffs, label = "mode:" + str(mode))
            plt.legend()
        plt.xlabel("widths [um]")
        plt.ylabel("neffs")
        plt.show()
        return None
    
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
