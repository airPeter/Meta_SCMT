# standard python imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import warnings
from .SCMT_model_2D import freespace_layer
import torch
from tqdm import tqdm

class Fullwave_2D():
    def __init__(self, GP) -> None:
        self.GP =GP
        self.sim = None
        
    def init_sim(self, prop_dis, N, hs, res = None, theta = 0, empty = False):
        '''
        input:
            hs: the 2d array of waveguides need to be simulated. shape [N, N]
            theta: if theta != 0, using gaussian beam, else using plane wave.
        '''
        # tidy3D lazy import
        import tidy3d as td
        warnings.warn("Fullwave is expensive and slow. You can set the prop_dis = 0, and use near_to_far to get the far field. Only do fullwave on small devices. And low resolution can be inaccurate.")
        if res == None:
            self.res = int(round(1 / self.GP.dh))
        else:
            self.res = res
        print("Fullwave resolution:", str(self.res))
        self.N = N
        self.prop_dis = prop_dis
        hs = hs.reshape(self.N, self.N)
        # Simulation domain size (in micron)
        z_size = self.GP.wh + 2 + prop_dis
        x_size = (N + 5) * self.GP.period
        y_size = x_size
        sim_size = [x_size, y_size, z_size]
        # Central frequency and bandwidth of pulsed excitation, in Hz
        fcen = td.constants.C_0 / self.GP.lam
        fwidth = fcen/10
        # Number of PML layers to use along each of the three directions.
        pml_layers = [15, 15, 15]
        # Total time to run in seconds
        run_time = max(50/fwidth, 5 * N * self.GP.period / td.constants.C_0)
        print("total running time:", run_time)

        # Lossless dielectric
        material1 = td.Medium(epsilon=self.GP.n_wg**2)
        waveguides = []
        z_plane = -z_size/2 + 1
        X = (np.arange(N) - N//2) * self.GP.period
        Y = X[::-1]
        positions = []
        if not empty:
            for i in range(N):
                for j in range(N):
                    width = hs[i, j]
                    x = X[j]
                    y = Y[i]
                    positions.append([float(x), float(y)])
                    temp_wg = td.Box(center=[x, y, z_plane + self.GP.wh/2], size=[width, width, self.GP.wh], material=material1)
                    waveguides.append(temp_wg)
            if self.GP.n_sub != 1:
                material2 = td.Medium(epsilon=self.GP.n_sub**2)
                sub = td.Box(center=[0, 0, -z_size/2], size=[x_size*2, y_size*2, 2], material=material2)
                waveguides.append(sub)
            positions = np.array(positions)
            self.hs_with_pos = np.c_[hs.reshape((-1,1)), positions]

        if theta == 0:
            source = td.PlaneWave(
                injection_axis='+z',
                position=-z_size/2 + 0.5,
                source_time = td.GaussianPulse(
                    frequency=fcen,
                    fwidth=fwidth),
                polarization='y')
            print(source)
        else:
            source = td.GaussianBeam(
                normal='z',
                center=[0, 0, -z_size/2 + 0.5],
                source_time=td.GaussianPulse(fcen, fwidth),
                angle_theta=theta,
                angle_phi=0,
                direction='forward',
                waist_radius=x_size * 10,
                pol_angle=np.pi/2) #S polarization.
            print("Using Gaussian beam. the waist radius usually should much smaller than the sim size.")
        self.monitors = []
        self.monitors.append(td.FreqMonitor(center=[0, 0, z_plane + self.GP.wh + self.GP.lam/2], size=[x_size, y_size, 0], freqs=[fcen]))
        if prop_dis != 0:
            self.monitors.append(td.FreqMonitor(center=[0, 0, z_plane + self.GP.wh + prop_dis], size=[x_size, y_size, 0], freqs=[fcen]))
        self.monitors.append(td.FreqMonitor(center=[0, 0, 0], size=[0, x_size, z_size], freqs=[fcen]))

        # Initialize simulation
        self.sim = td.Simulation(size=sim_size,
                            resolution=self.res,
                            structures=waveguides,
                            sources=[source],
                            monitors=self.monitors,
                            run_time=run_time,
                            pml_layers=pml_layers)    
        _, ax = plt.subplots(1, 3, figsize=(12, 3))
        self.sim.viz_mat_2D(normal='z', position=z_plane, ax=ax[0])
        self.sim.viz_mat_2D(normal='y', ax=ax[1], monitor_alpha=0)
        self.sim.viz_mat_2D(normal='x', ax=ax[2], source_alpha=0.9)
        plt.show()
        return None   
    
    def upload(self, task_name):
        # tidy3D lazy import
        from tidy3d import web
        self.task_name = task_name     
        self.project = web.new_project(self.sim.export(), task_name=task_name)
        web.monitor_project(self.project['taskId'])
        return None
    
    def download(self, data_path):
        # tidy3D lazy import
        from tidy3d import web
        web.download_results(self.project['taskId'], target_folder=data_path + self.task_name)
        # Show the output of the log file
        with open(data_path + self.task_name + "/tidy3d.log") as f:
            print(f.read())
        if self.sim == None:
            raise Exception("init sim first, then you can download data.")
            #self.sim = td.Simulation.import_json(data_path + self.task_name + "/simulation.json")
        self.sim.load_results(data_path + self.task_name + '/monitor_data.hdf5')
        return None
    
    def near_to_far(self, En, prop_dis):
        dx = self.GP.period / self.GP.out_res
        freelayer1 = freespace_layer(prop_dis - self.GP.lam/2, self.GP.lam, En.shape[0], dx)
        En = torch.tensor(En, dtype = torch.complex64)
        with torch.no_grad():
            Ef = freelayer1(En)
        out_Ef = Ef.cpu().numpy()
        return out_Ef
    
    def near_to_far_tidy3d(self, prop_dis, far_size):
        '''
            seems far off the ground truth.
        '''
        import tidy3d as td
        monitors = self.sim.monitors
        monitor_near = monitors[0]
        monitor_data = self.sim.data(monitor_near)
        n2f = td.Near2Far(monitor_data)
        # points to project to
        dx = self.sim.grid.mesh_step[0]
        xs_far = np.arange(-far_size/2, far_size/2, dx)
        ys_far = np.arange(-far_size/2, far_size/2, dx)
        num_far = xs_far.size
        # initialize the far field values
        E_far = np.zeros((3, num_far, num_far), dtype=complex)

        # loop through points in the output plane
        for i in tqdm(range(num_far)):
            x = xs_far[i]
            for j in range(num_far):
                y = ys_far[j]
                # compute and store the outputs from projection function at the focal plane
                E, H = n2f.get_fields_cartesian(x, y, prop_dis - self.GP.lam/2)
                E_far[:, i, j] = E
        E_far = E_far[1]
        return E_far
        
    def vis_monitor(self,path = None, tidy3d_viz = True):
        if path:
            if self.sim == None:
                raise Exception("init sim first, then you can download data.")
            self.sim.load_results(path + 'monitor_data.hdf5')
        monitors = self.sim.monitors
        if tidy3d_viz:
            _, ax = plt.subplots(1, 3, figsize=(18, 6))
            self.sim.viz_field_2D(monitors[0], ax=ax[0], cbar=True, comp='y', val='abs')
            self.sim.viz_field_2D(monitors[1], ax=ax[1], cbar=True, comp='y', val='abs')
            if len(monitors) == 3:
                self.sim.viz_field_2D(monitors[2], ax=ax[2], cbar=True, comp='y', val='abs')
        step1 = self.sim.grid.mesh_step[0]
        step2 = self.GP.period / self.GP.out_res
        mdata = self.sim.data(monitors[0])
        Ey_near = mdata['E'][1,:,:,0,0]
        Ey_near_FW = Ey_near
        Ey_near = resize_2d(Ey_near, step1, step2)
        phy_size_x = Ey_near.shape[1] * step2
        phy_size_y = Ey_near.shape[0] * step2
        show_field(Ey_near, phy_size_x, phy_size_y)
        if len(monitors) == 3:
            mdata = self.sim.data(monitors[1])
            Ey_far = mdata['E'][1,:,:,0,0] 
            Ey_far_FW = Ey_far
            Ey_far = resize_2d(Ey_far, step1, step2) 
            I_far = np.abs(Ey_far)**2
            show_intensity(I_far, phy_size_x, phy_size_y)     
        else:
            Ey_far = None
            Ey_far_FW = None
        print("return field: Ey_near, Ey_near_FW, Ey_far, Ey_far_FW. Ey_near is downsampled Ey_near_FW, so that resolution is same with cmt model.")
        return Ey_near, Ey_near_FW, Ey_far, Ey_far_FW
 
    def focal_efficiency(self, E_in, E_focal, NA):
        B = self.GP.out_res * 3
        I_in = np.abs(E_in[B:-B, B:-B])**2
        c = E_focal.shape[0]//2 - self.GP.out_res//2
        r = 6 * self.GP.out_res
        I_focal = np.abs(E_focal[c - r: c + r, c - r: c + r])**2
        self.FWHM(I_focal[r], NA)
        phy_size = 12 * self.GP.period
        show_intensity(I_focal, phy_size, phy_size)
        focal_efficiency = I_focal.sum() / I_in.sum() * 100
        print(f"focal efficiency: {focal_efficiency:2f}%.")
        return None
    
    def FWHM(self, I, NA):
        I_HM = I.max()/2
        idx_h = I.size//2
        I_diff = np.abs(I[:idx_h] - I_HM)
        idx_l = np.argmin(I_diff)
        I_diff = np.abs(I[idx_h:] - I_HM)
        idx_r = np.argmin(I_diff) 
        dx = self.GP.period / self.GP.out_res
        fwhm = (idx_r + idx_h - idx_l) * dx   
        print(f"Full width half maximum: {fwhm:.3f}")   
        theo_fwhm = self.GP.lam / 2 / NA
        print(f"Full width half maximum of diffraction limit: {theo_fwhm:.3f}")
        
def show_field(E, phy_size_x, phy_size_y):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plot0 = ax[0].imshow(np.angle(E), cmap = 'magma', origin='lower', extent = (-phy_size_x/2, phy_size_x/2, -phy_size_y/2, phy_size_y/2))
    ax[0].set_xlabel("Position [um]")
    ax[0].set_ylabel("Position [um]")
    ax[0].set_title("Phase")
    plt.colorbar(plot0, ax = ax[0])
    plot0 = ax[1].imshow(np.abs(E), cmap = 'magma', origin='lower', extent = (-phy_size_x/2, phy_size_x/2, -phy_size_y/2, phy_size_y/2))
    ax[1].set_xlabel("Position [um]")
    ax[1].set_ylabel("Position [um]")
    ax[1].set_title("Amplitude")
    plt.colorbar(plot0, ax = ax[1])
    plt.show()
    return None

def show_intensity(I, phy_size_x, phy_size_y):
    plt.figure()
    plt.imshow(I, cmap = 'magma', origin='lower', extent = (-phy_size_x/2, phy_size_x/2, -phy_size_y/2, phy_size_y/2))
    plt.xlabel("Position [um]")
    plt.ylabel("Position [um]")
    plt.colorbar()
    plt.title("Intensity")
    plt.show()
    
    
def resize_2d(field, step1, step2):
    '''
      tooooooooo slow!
    input:
        field 2D data needed to be resampled.
        step1, current step size.
        step2, output step size.
    output:
        a dict, that ['Ey'] , ['X'], ['Y'], data, and coordinate.
    '''
    phy_size_x = field.shape[1] * step1
    phy_size_y = field.shape[0] * step1
    
    x = np.linspace(0, phy_size_x, num = field.shape[1])
    y = np.linspace(0, phy_size_y, num = field.shape[0])

    f_real = interpolate.interp2d(x, y, np.real(field), kind='linear')
    f_imag = interpolate.interp2d(x, y, np.imag(field), kind='linear')
    
    x = np.arange(0, phy_size_x, step2)
    y = np.arange(0, phy_size_y, step2)
    out_real = f_real(x, y)
    out_img = f_imag(x, y)
    out_field = out_real + 1j * out_img
    # out = {}
    # out['Ey'] = out_field
    # out['x'] = x
    # out['y'] = y
    return out_field



    

