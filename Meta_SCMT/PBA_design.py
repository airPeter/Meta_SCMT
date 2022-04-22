'''
    design metasurface by periodic boundary approximation.
    need to do: add optimize method.
'''
# standard python imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import grcwa
from .utils import lens_2D, lens_1D, Model, train
import torch
#grcwa.set_backend('autograd')  # important!!
grcwa.set_backend('numpy')

class PBA():
    def __init__(self, GP, dim):
        self.GP = GP
        self.dim = dim # dim = 1 or 2.
        self.width_phase_map = None
        self.model = None
        
    def create_sim(self, width):   
        # set up objective function, x is the dielctric constant on the 2D grids, of size Nx*Ny
        # Qabs is a parameter for relatxation to better approach global optimal, at Qabs = inf, it will describe the real physics.
        # It also be used to resolve the singular matrix issues by setting a large but finite Qabs, e.g. Qabs = 1e5
        # Truncation order (actual number might be smaller)
        Qabs = np.inf
        nG = 101
        # lattice constants
        Period = self.GP.period
        wh = self.GP.wh #waveguide height
        L1 = [Period,0]
        L2 = [0, Period]
        # frequency and angles
        #!!!!!!!!!!!The vacuum permittivity, permeability, and speed of light are 1.
        wavelength = self.GP.lam
        freq = 1/wavelength #speed of light are 1.
        theta = 0.
        phi = 0.
        # the patterned layer has a griding: Nx*Ny
        Nx = 500
        Ny = 500
        # now consider 3 layers: vacuum + patterned + vacuum
        ep0 = self.GP.n_sub**2# dielectric for layer 1 (uniform)
        epH = self.GP.n_wg**2
        epL = 1
        epN = 1.  # dielectric for layer N (uniform)

        thick0 = 1. # thickness for vacuum layer 1
        thickp = wh # thickness of patterned layer (half wavelength)
        thickN = 1.
        pillar_eps = np.ones((Nx, Ny)) * epL
        delta_x = Period / Nx
        width_res = int(round(width / delta_x))
        start = int(round((Nx - width_res)/2))
        if self.dim == 1:
            pillar_eps[:, start:start + width_res] = epH
        elif self.dim == 2:
            pillar_eps[start:start + width_res, start:start + width_res] = epH
        else:
            raise Exception("dim invalid.")
        pillar_eps = pillar_eps.reshape(-1,)
        # planewave excitation
        planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
        freqcmp = freq*(1+1j/2/Qabs)
        ######### setting up RCWA
        obj = grcwa.obj(nG,L1,L2,freqcmp,theta,phi,verbose=0)
        # input layer information
        obj.Add_LayerUniform(thick0,ep0)
        obj.Add_LayerGrid(thickp,Nx,Ny)
        obj.Add_LayerUniform(thickN,epN)
        obj.Init_Setup()
        obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)    
        obj.GridLayer_geteps(pillar_eps)
        #R,T= obj.RT_Solve(normalize=1)
        field = obj.Solve_FieldOnGrid(1,z_offset = wh)
        return obj, pillar_eps.reshape(Ny, Nx), field

    def gen_lib(self, vis = True):
        widths = np.arange(self.GP.h_min, self.GP.h_max + self.GP.dh, self.GP.dh)
        phases = []
        for w in tqdm(widths):
            _, _, field = self.create_sim(w)
            ph = get_phase(field)
            phases.append(ph)
        phases = np.array(phases)
        if vis:
            plt.figure()
            plt.plot(widths, phases)
            plt.xlabel("Waveguide width [um]")
            plt.ylabel("Phase")
            plt.savefig(self.GP.path + "PBA_phase_vs_width.png")
            plt.show()
        L = phases.shape[0]
        width_phase_map = np.zeros((2, L))
        width_phase_map[0] = widths
        width_phase_map[1] = phases
        np.save(self.GP.path + "rcwa_width_phase_map.npy", width_phase_map)
        print("PBA width phase map saved.")
        self.width_phase_map = width_phase_map
        return None

    def fit(self, layers = 6, nodes = 64, steps = 1000, lr = 0.001, vis = True, load = True):
        if load:
            self.width_phase_map = np.load(self.GP.path + "rcwa_width_phase_map.npy")
        else:
            if self.width_phase_map is None:
                self.gen_lib()
        X, Y = self.width_phase_map[0], self.width_phase_map[1]
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        self.model = Model(1, 1, layers= layers, nodes = nodes)
        batch_size = 512
        Y_pred = train(self.model, X, Y, steps, lr, batch_size)
        torch.save(self.model.state_dict(), self.GP.path + "fitting_PBA_state_dict")
        paras = {'nodes': nodes, 'layers': layers}
        np.save(self.GP.path + "PBA_paras.npy", paras)
        print("model saved.")
        if vis:
            plt.figure()
            plt.plot(X, Y, label = "ground truth")
            plt.plot(X, Y_pred, linestyle = '--', label = "predicted")
            plt.legend()
            plt.xlabel("widths [um]")
            plt.ylabel("phase")
            plt.show()
        return None
    
    def design_lens(self, N, focal_length, load = False, vis = True):
        if load:
            self.width_phase_map = np.load(self.GP.path + "rcwa_width_phase_map.npy")
        else:
            if self.width_phase_map is None:
                self.gen_lib()
        if self.dim == 1:
            x_lens, lens = lens_1D(N, self.GP.period, focal_length, self.GP.k)
        elif self.dim == 2:
            x_lens, lens = lens_2D(N, self.GP.period, focal_length, self.GP.k)
        lens_phase = lens%(2 * np.pi) - np.pi
        widths_map = gen_width_from_phase(self.width_phase_map, lens_phase)
        widths_map = np.around(widths_map, 3)
        
        if vis:
            if self.dim == 2:
                fig, axs = plt.subplots(1, 2, figsize = (12, 6))
                plot1 = axs[0].imshow(lens, cmap = 'magma', extent = (x_lens.min(), x_lens.max(),x_lens.min(), x_lens.max()))
                plt.colorbar(plot1, ax = axs[0])
                plot2 = axs[1].imshow(widths_map, cmap = 'magma', extent = (x_lens.min(), x_lens.max(),x_lens.min(), x_lens.max()))
                plt.colorbar(plot2, ax = axs[1])
                axs[0].set_title("Lens phase")
                axs[0].set_xlabel("Position [um]")
                axs[0].set_ylabel("Position [um]")
                axs[1].set_title("Lens widths")
                axs[1].set_xlabel("Position [um]")
                axs[1].set_ylabel("Position [um]")
                plt.show()
            elif self.dim == 1:
                fig, axs = plt.subplots(2, 1, figsize = (12, 12))
                axs[0].plot(x_lens, lens)
                axs[1].plot(x_lens, widths_map)
                axs[0].set_title("Lens phase")
                axs[0].set_xlabel("Position [um]")
                axs[1].set_title("Lens widths")
                axs[1].set_xlabel("Position [um]")
                plt.show()
        return widths_map

    def width_to_phase(self, widths, dx, load = False, vis = True):
        '''
        input:
            widths: can be 1 or 2 dim. 
        output:
            phases: same size with widths
        '''
        if load:
            self.width_phase_map = np.load(self.GP.path + "rcwa_width_phase_map.npy")
        else:
            if self.width_phase_map is None:
                self.gen_lib()
        phases = gen_phase_from_width(self.width_phase_map, widths)
        if vis:
            x_lens = (np.arange(widths.shape[0]) - (widths.shape[0] - 1)/2) * dx
            if len(widths.shape) == 2:
                fig, axs = plt.subplots(1, 2, figsize = (12, 6))
                plot1 = axs[0].imshow(phases, cmap = 'magma', extent = (x_lens.min(), x_lens.max(),x_lens.min(), x_lens.max()))
                plt.colorbar(plot1, ax = axs[0])
                plot2 = axs[1].imshow(widths, cmap = 'magma', extent = (x_lens.min(), x_lens.max(),x_lens.min(), x_lens.max()))
                plt.colorbar(plot2, ax = axs[1])
                axs[0].set_title("Lens phase")
                axs[0].set_xlabel("Position [um]")
                axs[0].set_ylabel("Position [um]")
                axs[1].set_title("Lens widths")
                axs[1].set_xlabel("Position [um]")
                axs[1].set_ylabel("Position [um]")
                plt.show()
            else:
                fig, axs = plt.subplots(2, 1, figsize = (12, 12))
                axs[0].plot(x_lens, phases)
                axs[1].plot(x_lens, widths)
                axs[0].set_title("Lens phase")
                axs[0].set_xlabel("Position [um]")
                axs[1].set_title("Lens widths")
                axs[1].set_xlabel("Position [um]")
                plt.show()
        return phases
            
def gen_width_from_phase(width_phase_map, target_phase_profile):
    phases = width_phase_map[1]
    widths = width_phase_map[0]
    phases = phases.reshape(1,-1)
    target_shape = target_phase_profile.shape
    target_phase_profile = target_phase_profile.reshape(-1,1)
    diff = np.abs(target_phase_profile - phases)
    indexes = np.argmin(diff, axis = -1)
    widths_map = np.take(widths, indexes)
    widths_map = widths_map.reshape(target_shape)
    return widths_map

def gen_phase_from_width(width_phase_map, width_profile):
    phases = width_phase_map[1]
    widths = width_phase_map[0]
    widths = widths.reshape(1,-1)
    shape = width_profile.shape
    width_profile = width_profile.reshape(-1,1)
    diff = np.abs(width_profile - widths)
    indexes = np.argmin(diff, axis = -1)
    phases_map = np.take(phases, indexes)
    phases_map = phases_map.reshape(shape)
    return phases_map
    
def get_phase(field):
    Ey = field[0][1]
    phase = np.angle(Ey)
    shape = Ey.shape[0]
    center = shape//2
    return phase[center, center]