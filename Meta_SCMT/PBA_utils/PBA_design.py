'''
    design metasurface by periodic boundary approximation.
    need to do: add optimize method.
'''
# standard python imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ..utils import lens_2D, lens_1D, Model, train, deflector_1D, deflector_2D
import torch
import warnings
import os

class PBA():
    def __init__(self, GP, dim):
        self.GP = GP
        self.dim = dim # dim = 1 or 2.
        self.width_phase_map = None
        self.model = None
    def create_sim_tidy3d(self, width, inverse):
        import tidy3d as td
        # Simulation domain size (in micron)
        spacing = self.GP.lam
        z_size = self.GP.wh + 2 * spacing
        x_size = self.GP.period
        res = int(round(1 / (self.GP.period / 40)))
        if self.dim == 1:
            y_size = 0
        elif self.dim == 2:
            y_size = x_size
        sim_size = [x_size, y_size, z_size]
        # Central frequency and bandwidth of pulsed excitation, in Hz
        fcen = td.constants.C_0 / self.GP.lam
        self.fcen = fcen
        fwidth = fcen/10
        # Total time to run in seconds
        run_time = 200/fwidth
        # Lossless dielectric
        material1 = td.Medium(permittivity=self.GP.n_wg**2)
        material2 = td.Medium(permittivity=self.GP.n_sub**2)
        z_plane = -z_size/2 + spacing
        wg = td.Structure(
                        geometry = td.Box(center=[0, 0, z_plane + self.GP.wh/2],
                                size=[width, width, self.GP.wh]),
                        medium = material1,
                        name = 'wg')
        if inverse:
            sub = td.Structure(
                            geometry = td.Box(center=[0, 0, z_size/2],
                                    size=[x_size + 1, y_size + 1, 2 * spacing]),
                            medium = material2,
                            name = 'sub')
        else:
            sub = td.Structure(
                            geometry = td.Box(center=[0, 0, -z_size/2],
                                    size=[x_size + 1, y_size + 1, 2 * spacing]),
                            medium = material2,
                            name = 'sub')

        gaussian = td.GaussianPulse(freq0=fcen, fwidth=fwidth, phase=0)
        psource = td.PlaneWave(
            source_time=gaussian,
            size=(td.inf, td.inf, 0),
            center=(0,0,-z_size/2 + 0.5 * spacing),
            direction='+',
            pol_angle = np.pi/2, #Ey polarization.
        )
        #print(psource)
        freq_mnt1 = td.FieldMonitor(center=[0, 0, z_plane + self.GP.wh + 0.5 * spacing], size=[x_size, y_size, 0], freqs=[fcen], name = 'freq')

        grid_x = td.UniformGrid(dl=1/res)
        grid_y = td.UniformGrid(dl=1/res)
        grid_z = td.UniformGrid(dl=1/res)
        # Initialize simulation
        if self.dim == 1:
            sim = td.Simulation(size=sim_size,
                                grid_spec=td.GridSpec(wavelength=self.GP.lam, grid_x=grid_x, grid_z=grid_z),
                                structures=[wg, sub],
                                sources=[psource],
                                monitors=[freq_mnt1],
                                run_time=run_time,
                                boundary_spec=td.BoundarySpec(
                                    x=td.Boundary.periodic(),
                                    z=td.Boundary.pml(num_layers=20)
                                ))
        elif self.dim == 2:
            sim = td.Simulation(size=sim_size,
                                grid_spec=td.GridSpec(wavelength=self.GP.lam, grid_x=grid_x, grid_y=grid_y, grid_z=grid_z),
                                structures=[wg, sub],
                                sources=[psource],
                                monitors=[freq_mnt1],
                                run_time=run_time,
                                boundary_spec=td.BoundarySpec(
                                    x=td.Boundary.periodic(),
                                    y=td.Boundary.periodic(),
                                    z=td.Boundary.pml(num_layers=20)
                                ))
        return sim

    def create_sim(self, width, inverse):   
        import grcwa
        #grcwa.set_backend('autograd')  # important!!
        grcwa.set_backend('numpy')
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
        ep_sub = self.GP.n_sub**2# dielectric for layer 1 (uniform)
        epH = self.GP.n_wg**2
        epL = 1
        epN = 1.  # dielectric for layer N (uniform)

        thick0 = 1. # thickness for vacuum layer 1
        thickp = wh # thickness of patterned layer (half wavelength)
        thickN = 1.
        pillar_eps = np.ones((Nx, Ny)) * epL
        air_eps = np.ones((Nx, Ny))
        sub_eps = np.ones((Nx, Ny)) * ep_sub
        delta_x = Period / Nx
        width_res = int(round(width / delta_x))
        start = int(round((Nx - width_res)/2))
        if self.dim == 1:
            pillar_eps[:, start:start + width_res] = epH
        elif self.dim == 2:
            pillar_eps[start:start + width_res, start:start + width_res] = epH
        else:
            raise Exception("dim invalid.")
        # planewave excitation
        #p is along x direction.#the first dim.
        planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}
        freqcmp = freq*(1+1j/2/Qabs)
        ######### setting up RCWA
        obj = grcwa.obj(nG,L1,L2,freqcmp,theta,phi,verbose=0)
        # input layer information
        if inverse:
            obj.Add_LayerUniform(thickN,epN)
            obj.Add_LayerGrid(thickp,Nx,Ny)
            obj.Add_LayerGrid(wavelength,Nx,Ny)
            obj.Add_LayerUniform(thick0,ep_sub)
            epgrid = np.concatenate((pillar_eps.flatten(), sub_eps.flatten()))
        else: 
            obj.Add_LayerUniform(thick0,ep_sub)
            obj.Add_LayerGrid(thickp,Nx,Ny)
            obj.Add_LayerGrid(wavelength,Nx,Ny)
            obj.Add_LayerUniform(thickN,epN)
            epgrid = np.concatenate((pillar_eps.flatten(), air_eps.flatten()))
        obj.Init_Setup()
        obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)    
        obj.GridLayer_geteps(epgrid)
        #R,T= obj.RT_Solve(normalize=1)
        field = obj.Solve_FieldOnGrid(2,z_offset = wavelength/2)
        return obj, pillar_eps.reshape(Ny, Nx), field

    def gen_lib(self, vis = True, backend = 'grcwa', step_size = 0.001, inverse = False):
        '''
            inverse: if inverse is true, the light is incident from air to waveguide then to substrate, else: the light is from substrate to waveguide then  to air.
        
        '''
        widths = np.arange(self.GP.h_min, self.GP.h_max + step_size, step_size)
        phases = []
        amps = []
        if backend == 'tidy3d':
            from tidy3d import web
            sims = {f'width:{np.round(width, 2)}': self.create_sim_tidy3d(width, inverse) for width in widths}
            batch = web.Batch(simulations=sims)
            path_dir = self.GP.path + 'tidy3d_PBA_lib/'
            if not os.path.exists(path_dir):
                os.mkdir(path_dir)
            batch_results = batch.run(path_dir=path_dir)
            for _, sim_data in batch_results.items():
                ph, amp = get_phase_and_amp_tidy3d(sim_data)
                amps.append(amp)
                phases.append(ph)     
        elif backend == 'grcwa':
            for w in tqdm(widths):
                _, _, field = self.create_sim(w, inverse)
                ph = get_phase(field)
                amp = get_amp(field)
                amps.append(amp)
                phases.append(ph)
        phases = np.array(phases)
        amps = np.array(amps)
        # ccoeffs = np.polynomial.chebyshev.chebfit(widths, phases, deg = 3)
        # fcheb = np.polynomial.Chebyshev(ccoeffs)
        # widths_finer = np.linspace(widths.min(), widths.max(), 100)
        # phases_finer = fcheb(widths_finer)
        if vis:
            plt.figure()
            plt.plot(widths, phases)
            #plt.plot(widths_finer, phases_finer)
            plt.xlabel("Waveguide width [um]")
            plt.ylabel("Phase")
            if inverse:
                plt.savefig(self.GP.path + "PBA_phase_vs_width_inverse.png")
            else:
                plt.savefig(self.GP.path + "PBA_phase_vs_width.png")
            plt.show()
            
            plt.figure()
            plt.plot(widths, amps)
            #plt.plot(widths_finer, phases_finer)
            plt.xlabel("Waveguide width [um]")
            plt.ylabel("Amp")
            if inverse:
                plt.savefig(self.GP.path + "PBA_amp_vs_width_inverse.png")
            else:
                plt.savefig(self.GP.path + "PBA_amp_vs_width.png")
            plt.show()
        L = phases.shape[0]
        width_phase_map = np.zeros((3, L))
        width_phase_map[0] = widths
        width_phase_map[1] = phases
        width_phase_map[2] = amps
        if inverse:
            np.save(self.GP.path + "rcwa_width_phase_map_inverse.npy", width_phase_map)
        else:
            np.save(self.GP.path + "rcwa_width_phase_map.npy", width_phase_map)
        print("PBA width phase map saved.")
        self.width_phase_map = width_phase_map
        return None

    def fit(self, layers = 6, nodes = 64, steps = 1000, lr = 0.001, vis = True, load = True, inverse = False):
        if load:
            if inverse:
                self.width_phase_map = np.load(self.GP.path + "rcwa_width_phase_map_inverse.npy")
            else:
                self.width_phase_map = np.load(self.GP.path + "rcwa_width_phase_map.npy")
        else:
            if self.width_phase_map is None:
                self.gen_lib(inverse=inverse)
        X, Y = self.width_phase_map[0], self.width_phase_map[1]
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        self.model = Model(1, 1, layers= layers, nodes = nodes)
        batch_size = 512
        Y_pred = train(self.model, X, Y, steps, lr, batch_size)
        if inverse:
            torch.save(self.model.state_dict(), self.GP.path + "fitting_PBA_state_dict_inverse")
        else:
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
    
    def design_lens(self, N, focal_length, load = False, vis = True, quarter = False):
        if load:
            self.width_phase_map = np.load(self.GP.path + "rcwa_width_phase_map.npy")
        else:
            if self.width_phase_map is None:
                self.gen_lib()
        if self.dim == 1:
            x_lens, lens = lens_1D(N, self.GP.period, focal_length, self.GP.k)
        elif self.dim == 2:
            if quarter:
                N2 = 2 * N
                x_lens, lens = lens_2D(N2, self.GP.period, focal_length, self.GP.k)
                x_lens = x_lens[N:]
                lens = lens[N:,N:]
            else:
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
        return lens_phase, widths_map

    def design_deflector(self, N, degree, load = False, vis = True):
        if load:
            self.width_phase_map = np.load(self.GP.path + "rcwa_width_phase_map.npy")
        else:
            if self.width_phase_map is None:
                self.gen_lib()
        if self.dim == 1:
            x_lens, lens = deflector_1D(N, self.GP.period, degree, self.GP.k)
        elif self.dim == 2:
                x_lens, lens = deflector_2D(N, self.GP.period, degree, self.GP.k)
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
        return lens_phase, widths_map
    
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

    def width_to_amp(self, widths, dx, load = False, vis = True):
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
        amps = gen_amp_from_width(self.width_phase_map, widths)
        if vis:
            x_lens = (np.arange(widths.shape[0]) - (widths.shape[0] - 1)/2) * dx
            if len(widths.shape) == 2:
                fig, axs = plt.subplots(1, 2, figsize = (12, 6))
                plot1 = axs[0].imshow(amps, cmap = 'magma', extent = (x_lens.min(), x_lens.max(),x_lens.min(), x_lens.max()))
                plt.colorbar(plot1, ax = axs[0])
                plot2 = axs[1].imshow(widths, cmap = 'magma', extent = (x_lens.min(), x_lens.max(),x_lens.min(), x_lens.max()))
                plt.colorbar(plot2, ax = axs[1])
                axs[0].set_title("Lens amplitude")
                axs[0].set_xlabel("Position [um]")
                axs[0].set_ylabel("Position [um]")
                axs[1].set_title("Lens widths")
                axs[1].set_xlabel("Position [um]")
                axs[1].set_ylabel("Position [um]")
                plt.show()
            else:
                fig, axs = plt.subplots(2, 1, figsize = (12, 12))
                axs[0].plot(x_lens, amps)
                axs[1].plot(x_lens, widths)
                axs[0].set_title("Lens amplitude")
                axs[0].set_xlabel("Position [um]")
                axs[1].set_title("Lens widths")
                axs[1].set_xlabel("Position [um]")
                plt.show()
        return amps
      
def gen_width_from_phase(width_phase_map, target_phase_profile):
    phases = width_phase_map[1]
    widths = width_phase_map[0]
    phases = phases.reshape(1,-1)
    target_shape = target_phase_profile.shape
    target_phase_profile = target_phase_profile.reshape(-1,1)
    #diff = np.abs(target_phase_profile - phases)
    diff = np.abs(np.exp(1j*target_phase_profile) - np.exp(1j * phases))
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

def gen_amp_from_width(width_phase_map, width_profile):
    amps = width_phase_map[2]
    widths = width_phase_map[0]
    widths = widths.reshape(1,-1)
    shape = width_profile.shape
    width_profile = width_profile.reshape(-1,1)
    diff = np.abs(width_profile - widths)
    indexes = np.argmin(diff, axis = -1)
    amps_map = np.take(amps, indexes)
    amps_map = amps_map.reshape(shape)
    return amps_map
# def get_phase(field):
#     Ey = field[0][0]
#     Ey_zero = Ey.mean()
#     return np.angle(Ey_zero)

def get_phase(field):
    #which polarization to take is depends on the incident light. Here, the incident polar is along x direction.
    E = field[0][0]
    phase = np.angle(E)
    shape = E.shape[0]
    center = shape//2
    return phase[center, center]

def get_amp(field):
    E = field[0][0]
    amp = np.abs(E)
    shape = E.shape[0]
    center = shape//2
    return amp[center, center] 

def get_phase_and_amp_tidy3d(sim_data):
    E = sim_data['freq'].Ey.values
    E = E[:,:,0,0]
    phase = np.angle(E)
    amp = np.abs(E)
    size1 = phase.shape[0]
    size2 = phase.shape[1]
    center_phase = phase[size1//2, size2//2]
    center_amp = amp[size1//2, size2//2]
    return center_phase, center_amp