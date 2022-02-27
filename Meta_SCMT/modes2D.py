'''
support modes <=2, but modes = 2 not tested.
'''
import warnings
from cv2 import exp
import numpy as np
from .utils import h2index
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
# tidy3D import
import tidy3d as td
from tidy3d import web
from tqdm import tqdm
import warnings

class Gen_modes2D():
    def __init__(self, GP):
        self.GP = GP
        self.H = np.arange(GP.h_min, GP.h_max + GP.dh, GP.dh)
        self.modes_lib = None
        self.batch = None
        self.batch_path = None
        if self.GP.modes > 2:
            raise Exception("gen modes 2D only support modes <= 2.")
    def count_modes(self,):
        if self.modes_lib == None:
            raise Exception("gen modes first!")
        cnts = np.zeros((self.GP.modes,))
        for key in self.modes_lib.keys():
            for m in range(self.GP.modes):
                neff = self.modes_lib[key][m]['neff']
                if neff > self.GP.n0:
                    cnts[m] += 1
        print("total keys: ", len(self.modes_lib.keys()))
        print("number of non zero modes: ", cnts)
        return None

    def local_preview(self,width):
        GP = self.GP
        sim = create_sim(width, GP.lam, GP.n_wg, GP.res, GP.period, GP.Knn, GP.modes)
        fig = plt.figure(figsize=(11, 4))
        gs = mpl.gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.4])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        sim.viz_eps_2D(normal='y', position=0.6/2, ax=ax1, monitor_alpha=0.9, source_alpha=0.9)
        sim.viz_eps_2D(normal='z', ax=ax2)
        plt.show()
        # Visualize the modes. The mode computation is called internally.
        sim.compute_modes(sim.sources[0], Nmodes=GP.modes * 2)
        sim.viz_modes(sim.sources[0], cbar=True)
        plt.show()
    
    def upload(self, batch_path, base_dir):
        GP = self.GP
        # submit all jobs
        step = GP.period/GP.res
        self.sim_res = int(round(1/step))
        sims = [create_sim(h, GP.lam, GP.n_wg, self.sim_res, GP.period, GP.Knn, GP.modes) for h in self.H]
        batch = web.Batch(sims, base_dir=GP.path + base_dir)
        batch.save(GP.path + batch_path)
        self.batch = batch
        return None
    def monitor(self, batch_path = None):
        if batch_path:
            batch_path = self.GP.path + batch_path
            if os.path.exists(batch_path):
                self.batch = web.Batch.load_from_file(batch_path)
            else:
                raise Exception("batch file not saved. Should run upload first.")
        self.batch.monitor()
        return None
    
    def gen(self,load = False, batch_path = None):
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
            #consistency check
            total_hs = (self.GP.h_max - self.GP.h_min + self.GP.dh)//self.GP.dh + 1
            load_total_hs = len(modes_lib.keys())
            if total_hs != load_total_hs:
                print("expected total waveguides:" + str(total_hs) + "loaded:" + str(load_total_hs))
                #raise Exception('You indeed change the physical setup without regenerate the modes!')
            self.modes_lib = modes_lib
        else:
            if batch_path:
                batch_path = self.GP.path + batch_path
                self.batch = web.Batch.load_from_file(batch_path)
            GP = self.GP
            # get results from all jobs
            warnings.warn("only load the results once you monitor that the simulation run on server is done.")
            sims_loaded = self.batch.load_results()
            none_sims = []
            for i, sim in enumerate(sims_loaded):
                if sim == None:
                    none_sims.append(i)
            if len(none_sims) > 0:
                raise Exception("for these sims", none_sims, "tidy3d load went wrong. manually download the results from website to", self.base_dir, "and run gen again.")
            #visual the fields and saved in sim_cache/show_fields.
            root_path = GP.path + "show_fields/"
            if not os.path.exists(root_path):
                os.mkdir(root_path)
            for i, sim in enumerate(sims_loaded):
                path = root_path + str(i) + ".png"
                print_field(sim, path)
            modes_lib = {}
            f_size = 2 * (GP.Knn + 1) * GP.res
            zero_field = np.zeros((f_size, f_size))
            for i, h in tqdm(enumerate(self.H)):
                h_index = h2index(h, GP.dh)
                modes_lib[h_index] = {}
                for n_mode in range(GP.modes):
                    modes_lib[h_index][n_mode] = {}
                    neff, _, Ey, Hx = get_field_mode(sims_loaded[i], n_mode)
                    Ey = self.resize_field(Ey)
                    Hx = self.resize_field(Hx)
                    if neff > GP.n0 + 0.1:
                        modes_lib[h_index][n_mode]['neff'] = neff
                        normalization = np.sqrt(- 2 * np.sum(Ey * Hx) * GP.dx**2)
                        modes_lib[h_index][n_mode]['Ey'] = Ey / normalization
                        modes_lib[h_index][n_mode]['Hx'] = Hx / normalization
                    else:
                        modes_lib[h_index][n_mode]['neff'] = GP.n0
                        modes_lib[h_index][n_mode]['Ey'] = zero_field
                        modes_lib[h_index][n_mode]['Hx'] = zero_field
            self.modes_lib =  modes_lib
            np.save(load_path, self.modes_lib)
            print("generated modes lib saved at:" + load_path)
        return None

    def resize_field(self, field):
        expect_size = 2 * (self.GP.Knn + 1) * self.GP.res
        out_f = np.zeros((expect_size, expect_size), dtype= field.dtype)
        c0 = int(field.shape[0]//2)
        c1 = int(field.shape[1]//2)
        r0 = min(int(expect_size//2), field.shape[0]//2)
        r1 = min(int(expect_size//2), field.shape[1]//2)
        c_out = expect_size//2
        out_f[c_out - r0: c_out + r0, c_out - r1: c_out + r1] = field[c0 - r0: c0 + r0, c1 - r1: c1 + r1]
        return out_f
    
    def vis_field(self,H):
        '''
            H an list of wg width you want to plot.
        '''
        if self.modes_lib == None:
            raise Exception("gen modes first!")
        for h in H:
            index = h2index(h, self.GP.dh)
            for n_mode in range(self.GP.modes):
                fig, axs = plt.subplots(1, 2, figsize = (12, 6))
                Ey = self.modes_lib[index][n_mode]['Ey']
                neff = self.modes_lib[index][n_mode]['neff']
                L = "h:" + str(round(h,3)) + "neff" + str(neff) + "mode:" + str(n_mode) + "Ey"
                plot0 = axs[0].imshow(np.real(Ey))
                axs[0].set_title(L)
                plt.colorbar(plot0, ax = axs[0])
                Hx = self.modes_lib[index][n_mode]['Hx']
                L = "h:" + str(round(h,3)) + "neff" + str(neff)  + "mode:" + str(n_mode) + "Hx"
                plot1 = axs[1].imshow(np.real(Hx))
                axs[1].set_title(L)
                plt.colorbar(plot1, ax = axs[1])
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
    
def create_sim(width, wavelength, n1, resolution, period, Knn, modes):
    # Unit length is micron.
    wg_size = width
    # Free-space wavelength (in um) and frequency (in Hz)
    lambda0 = wavelength
    freq0 = td.C_0/lambda0
    fwidth = freq0/10
    run_time = 20/fwidth
    # Simulation size inside the PML along propagation direction
    sim_length = 1
    # PML layers
    Npml = 15
    # Simulation domain size and total run time
    sim_width = 2 * period * (Knn + 1) + 1
    sim_size = [sim_width, sim_width, sim_length]
    # Waveguide and substrate materials
    mat_wg = td.Medium(epsilon=n1**2)
    # Waveguide
    waveguide = td.Box(
        center=[0, 0, 0],
        size=[wg_size, wg_size, 100],
        material=mat_wg)

    # Modal source
    src_pos = -sim_size[-1]/2 + 0.2
    msource = td.ModeSource(
        center=[0, 0, src_pos],
        size=[sim_width - 1, sim_width - 1, 0],
        source_time = td.GaussianPulse(
            frequency=freq0,
            fwidth=fwidth),
        direction='forward')
    # Modal monitor at a range of frequencies
    mode_mnt = td.ModeMonitor(
        center=[0,0,src_pos + 0.5],
        size=[sim_width - 1, sim_width - 1, 0],
        freqs=freq0,
        Nmodes=modes * 2)
    # Simulation
    sim = td.Simulation(
        size=sim_size,
        resolution=resolution,
        structures=[waveguide],
        sources=[msource],
        monitors=[mode_mnt],
        run_time=run_time/10000,
        pml_layers=[Npml]*3)
    sim.set_mode(msource, mode_ind=0)
    return sim

def get_field_mode(sim, mode):
    if mode == 0:
        return get_field_mode1(sim)
    if mode == 1:
        return get_field_mode2(sim)
    else:
        raise Exception("only modes <= 2 is supported, for Modes2D module.")
    
def get_field_mode1(sim,):
    mode_mnt = sim.monitors[0]
    neff = sim.data(mode_mnt)['modes'][0][0].__dict__['neff']
    keff = sim.data(mode_mnt)['modes'][0][0].__dict__['keff'] 
    Ey0 = sim.data(mode_mnt)['modes'][0][0].__dict__['E'][1]
    Ey1 = sim.data(mode_mnt)['modes'][0][1].__dict__['E'][1]
    if np.abs(Ey0).mean() > np.abs(Ey1).mean():
        Ey = Ey0
        Hx = sim.data(mode_mnt)['modes'][0][0].__dict__['H'][0]
    else:
        Ey = Ey1
        Hx = sim.data(mode_mnt)['modes'][0][1].__dict__['H'][0]
    Ey = Ey.T
    Hx = Hx.T
    return neff, keff, Ey, Hx

def get_field_mode2(sim):
    mode_mnt = sim.monitors[0]
    neff = sim.data(mode_mnt)['modes'][0][2].__dict__['neff']
    keff = sim.data(mode_mnt)['modes'][0][2].__dict__['keff'] 
    Ey = sim.data(mode_mnt)['modes'][0][2].__dict__['E'][1]
    Hx = sim.data(mode_mnt)['modes'][0][2].__dict__['H'][1]
    Ey = Ey.T
    Hx = Hx.T
    return neff, keff, Ey, Hx  

def print_field(sim, path):
    mode_mnt = sim.monitors[0]
    sim.viz_modes(mode_mnt, freq_ind=0, cbar=True)
    plt.savefig(path)
    plt.close()