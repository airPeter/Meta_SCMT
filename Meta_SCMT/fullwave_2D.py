# standard python imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import warnings
from .SCMT_model_2D import freespace_layer
import torch
from tqdm import tqdm
from .cmap import paper_cmap
from .ideal_meta import Ideal_meta
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
        if hs.max() > self.GP.h_max:
            warnings.warn("initial widths larger than h_max, bad initial widths for waveguides.")
        if hs.min() < self.GP.h_min:
            warnings.warn("initial widths smaller than h_min, bad initial widths for waveguides.")
        if res == None:
            self.res = int(round(1 / self.GP.dh))
        else:
            self.res = res
        print("Fullwave resolution:", str(self.res))
        self.N = N
        self.dl = 1 / self.res
        self.prop_dis = prop_dis
        hs = hs.reshape(self.N, self.N)
        # Simulation domain size (in micron)
        spacing = 1 * self.GP.lam
        z_size = self.GP.wh + 2 * spacing + prop_dis * 1.1
        x_size = (N) * self.GP.period
        self.x_size = x_size
        xh = N * self.GP.period
        NA = np.sin(xh / np.sqrt(xh**2 + prop_dis**2))
        print(f"numerical aperture: {NA:.2f}")
        self.min_focal_spot = self.GP.lam / 2 / NA
        self.efficiency_length = 6 * self.min_focal_spot
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
        z_plane = -z_size/2 + spacing
        X = (np.arange(N) - N/2) * self.GP.period + self.GP.period/2
        Y = X[::-1]
        positions = []
        if not empty:
            if self.GP.n_sub != 1:
                material2 = td.Medium(epsilon=self.GP.n_sub**2)
                sub = td.Box(center=[0, 0, -z_size + spacing], size=[td.inf, td.inf, z_size], material=material2)
                waveguides.append(sub)
            for i in range(N):
                for j in range(N):
                    width = hs[i, j]
                    x = X[j]
                    y = Y[i]
                    positions.append([float(x), float(y)])
                    temp_wg = td.Box(center=[x, y, z_plane + self.GP.wh/2], size=[width, width, self.GP.wh], material=material1)
                    waveguides.append(temp_wg)
            positions = np.array(positions)
            self.hs_with_pos = np.c_[hs.reshape((-1,1)), positions]

        if theta == 0:
            source = td.PlaneWave(
                injection_axis='+z',
                position=-z_size/2 + 2 * self.dl,
                source_time = td.GaussianPulse(
                    frequency=fcen,
                    fwidth=fwidth),
                polarization='y')
            print(source)
        else:
            source = td.GaussianBeam(
                normal='z',
                center=[0, 0, -z_size/2 + 2 * self.dl],
                source_time=td.GaussianPulse(fcen, fwidth),
                angle_theta=theta,
                angle_phi=0,
                direction='forward',
                waist_radius=x_size * 10,
                pol_angle=np.pi/2) #S polarization.
            print("Using Gaussian beam. the waist radius usually should much smaller than the sim size.")
            
        monitor_axis = td.FreqMonitor(
                        center=[0., 0., 0],
                        size=[0, 0, z_size],
                        freqs=[fcen],
                        store=['E', 'H'],
                        name='axis')
        monitor_focal_scan = td.FreqMonitor(
                        center=[0., 0., -z_size/2 + spacing + self.GP.wh + prop_dis],
                        size=[x_size, 0, 0],
                        freqs=[fcen],
                        store=['E', 'H'],
                        name='focal_plane')
        monitor_in = td.FreqMonitor(
                        center=[0., 0., -z_size/2 + spacing -2*self.dl],
                        size=[x_size, x_size, 0],
                        freqs=[fcen],
                        store=['E', 'H'],
                        name='incident')
        monitor_near = td.FreqMonitor(
                        center=[0., 0., -z_size/2 + spacing + self.GP.wh + spacing/2],
                        size=[x_size, x_size, 0],
                        freqs=[fcen],
                        store=['E', 'H'],
                        name='near_field')
        monitor_xy = td.FreqMonitor(
                        center=[0., 0., -z_size/2 + spacing + self.GP.wh + prop_dis],
                        size=[x_size, x_size, 0],
                        freqs=[fcen],
                        store=['E', 'H'],
                        name='focal_plane')
        monitor_eff = td.FreqMonitor(
                        center=[0., 0., -z_size/2 + spacing + self.GP.wh + prop_dis],
                        size=[self.efficiency_length, self.efficiency_length, 0],
                        freqs=[fcen],
                        store=['E', 'H'],
                        name='focus')
        monitor_CS1 = td.FreqMonitor(
                        center=[0., 0., 0],
                        size=[x_size, 0, z_size],
                        freqs=[fcen],
                        store=['E', 'H'],
                        name='cross_section1')

        self.monitors = [monitor_axis, monitor_focal_scan, monitor_in, monitor_near, monitor_xy, monitor_eff, monitor_CS1]

        # Initialize simulation
        self.sim = td.Simulation(size=sim_size,
                            resolution=self.res,
                            structures=waveguides,
                            sources=[source],
                            monitors=self.monitors,
                            run_time=run_time,
                            pml_layers=pml_layers)    
        _, ax = plt.subplots(1, 2, figsize=(12, 6))
        self.sim.viz_mat_2D(normal='z', position=z_plane, ax=ax[0])
        self.sim.viz_mat_2D(normal='x', ax=ax[1], source_alpha=0.9)
        plt.show()
        return None   
    
    def upload(self, task_name):
        # tidy3D lazy import
        from tidy3d import web
        self.task_name = task_name     
        self.project = web.new_project(self.sim.export(), task_name=task_name)
        web.monitor_project(self.project['taskId'])
        return None
    
    def download(self, data_path, taskId = None, task_name = None):
        # tidy3D lazy import
        from tidy3d import web
        if taskId:
            self.task_name = task_name
            web.download_results(taskId, target_folder=data_path + self.task_name)
        else:
            web.download_results(self.project['taskId'], target_folder=data_path + self.task_name)
        # Show the output of the log file
        with open(data_path + self.task_name + "/tidy3d.log") as f:
            print(f.read())
        if self.sim is None:
            raise Exception("self.sim is None.")
            #self.sim = td.Simulation.import_json(data_path + self.task_name + "/simulation.json")
        self.sim.load_results(data_path + self.task_name + '/monitor_data.hdf5')
        return None
    
    def near_to_far(self, En, prop_dis):
        dx = self.GP.period / self.GP.out_res
        freelayer1 = freespace_layer(prop_dis, self.GP.lam, En.shape[0], dx)
        En = torch.tensor(En, dtype = torch.complex64)
        with torch.no_grad():
            Ef = freelayer1(En)
        out_Ef = Ef.cpu().numpy()
        return out_Ef

    def results_analysis(self,path = None):
        '''
            In theory, the intensity should be the norm of poynting vector. S = E x H.
            However, in freespace the ||S|| is very close to a||E||, a is a constant.
            Basically, Ey dot Hx is very propotionally close to Ey^2. 
            according the Maxwell equation, Hx ~ \partial_z(Ey). This shows that at any point x0, we can aaproximate Ey by
            Ey(x0)exp(1jkz), which means Ey(x, z) ~ Ey(x)exp(1jkz). This is very suprising results.
             
        
        '''
        def FWHM(xs, Is):
            # assume uniform sampling in xs
            dx = np.mean(np.diff(xs))
            hm = np.max(Is) / 2.0
            return dx * np.sum(Is > hm)
        if path:
            if self.sim is None:
                raise Exception("init sim first, then you can download data.")
            self.sim.load_results(path + 'monitor_data.hdf5')
        monitors = self.sim.monitors
        monitor_axis, monitor_focal_scan, monitor_in, monitor_near, monitor_xy, monitor_eff, monitor_CS1 = monitors
        # intensity along focal plane
        data_focus = self.sim.data(monitor_focal_scan)
        E_focus = np.squeeze(data_focus['E'])
        I_focus = np.sum(np.square(np.abs(E_focus)), axis=0)
        xs = data_focus['xmesh']
        fwhm = FWHM(xs, I_focus)
        print(f'fwhm = {fwhm:.4f} um, {(fwhm / self.GP.lam):.2f} $\lambda$')
        #theo_fwhm = 1.025 * self.GP.lam * self.prop_dis / self.x_size
        
        data_far = self.sim.data(monitor_xy)
        E_far = np.squeeze(data_far['E'])
        I_far = np.sum(np.square(np.abs(E_far)), axis=0)
        xs_far = data_far['xmesh']
        ys_far = data_far['ymesh']
        ideal_meta = Ideal_meta(self.GP)
        ideal_meta.model_init(self.N, self.prop_dis, lens = True)
        I_ideal = ideal_meta.forward()
        I_ideal = resize_2d_intensity(I_ideal, ideal_meta.dx, xs_far, ys_far)
        power_ideal = np.sum(I_ideal)
        r = int(round(self.efficiency_length / np.mean(np.diff(xs_far)) / 2))
        c = I_ideal.shape[0]//2
        I_focus_ideal = I_ideal[c - r: c + r, c - r: c + r]
        power_ideal_focus = I_focus_ideal.sum()
        print(f'Ideal focal area power/total_far_field_power = {power_ideal_focus/ power_ideal * 100:.2f}%')
        I_far_normalized = I_far / np.sum(I_far) * power_ideal
        strehl_ratio = np.max(I_far_normalized) / np.max(I_ideal)
        I_ideal_1D = I_ideal[I_ideal.shape[0]//2]
        I_ideal_normalized_1D = I_ideal_1D/I_ideal_1D.max()
        I_far_normalized_1D = I_far_normalized[I_far_normalized.shape[0]//2]
        I_far_normalized_1D = I_far_normalized_1D/I_ideal_1D.max()
        
        #diff_lim = airy(xs)
        fwhm_airy = FWHM(xs_far, I_ideal_1D)
        print(f'fwhm_airy = {fwhm_airy:.4f} um,  {(fwhm_airy / self.GP.lam):.2f} $\lambda$')

        plt.plot(xs_far / self.GP.lam, I_far_normalized_1D, label='measured')
        plt.plot(xs_far / self.GP.lam, I_ideal_normalized_1D, label='diffraction limited')
        plt.xlim([-2, 2])
        plt.legend()
        plt.title(f'FWHM = {(fwhm / self.GP.lam):.4f} $\lambda$, {(fwhm*1000):.2f} nm , Strehl ratio = {strehl_ratio:.4f}')
        plt.xlabel('axis position ($\lambda_0$)')
        plt.ylabel('intensity (normalized)')
        plt.show()     

        power_in = self.sim.flux(monitor_in)[0][0]
        power_near = self.sim.flux(monitor_near)[0][0]
        power_far = self.sim.flux(monitor_xy)[0][0]
        power_focus = self.sim.flux(monitor_eff)[0][0]

        eff_trans = power_near / power_in
        eff_far = power_far / power_in
        eff_focus = power_focus / power_in

        print(f'transmission efficiency = {(eff_trans*100):.2f}%')
        print(f'far field efficiency = {(eff_far*100):.2f}%')
        print(f'focusing efficiency = {(eff_focus*100):.2f}%')
        print(f'focal area power/total_far_field_power = {power_focus / power_far * 100:.2f}%')
    def vis_monitor(self,path = None, tidy3d_viz = True):
        
        if path:
            if self.sim is None:
                raise Exception("init sim first, then you can download data.")
            self.sim.load_results(path + 'monitor_data.hdf5')
        monitors = self.sim.monitors
        monitor_axis, monitor_focal_scan, monitor_in, monitor_near, monitor_xy, monitor_eff, monitor_CS1= monitors
        
        if tidy3d_viz:
            _, ax = plt.subplots(1, 2, figsize=(12, 6))
            im = self.sim.viz_field_2D(monitor_CS1, ax=ax[0], cbar=True, comp='y', val='int')
            im.set_cmap(paper_cmap)
            im = self.sim.viz_field_2D(monitor_eff, ax=ax[1], cbar=True, comp='y', val='int')
            im.set_cmap(paper_cmap)
            plt.show()
            
        step1 = self.sim.grid.mesh_step[0]
        step2 = self.GP.period / self.GP.out_res
        # mdata = self.sim.data(monitor_in)
        # Ey_in = mdata['E'][1,:,:,0,0]
        # Ey_in_FW = Ey_in
        # Ey_in = resize_2d(Ey_in, step1, step2)
        # phy_size_x = Ey_in.shape[1] * step2
        # phy_size_y = Ey_in.shape[0] * step2
        # show_field(Ey_in, phy_size_x, phy_size_y)
        mdata = self.sim.data(monitor_near)
        Ey_near = mdata['E'][1,:,:,0,0]
        Ey_near_FW = Ey_near
        Ey_near = resize_2d(Ey_near, step1, step2)
        phy_size_x = Ey_near.shape[1] * step2
        phy_size_y = Ey_near.shape[0] * step2
        show_field(Ey_near, phy_size_x, phy_size_y)

        mdata = self.sim.data(monitor_xy)
        Ey_far = mdata['E'][1,:,:,0,0] 
        Ey_far_FW = Ey_far
        Ey_far = resize_2d(Ey_far, step1, step2) 
        I_far = np.abs(Ey_far)**2
        show_intensity(I_far, phy_size_x, phy_size_y)     

        mdata = self.sim.data(monitor_eff)
        Ey_eff = mdata['E'][1,:,:,0,0] 
        I_eff = np.abs(Ey_eff)**2
        show_intensity(I_eff, self.efficiency_length, self.efficiency_length)                
        print("return field: Ey_near, Ey_near_FW, Ey_far, Ey_far_FW. Ey_near is downsampled Ey_near_FW, so that resolution is same with cmt model.")
        return Ey_near, Ey_near_FW, Ey_far, Ey_far_FW
 
    def focal_efficiency(self, E_in, E_focal, NA):
    
        def FWHM(Is, dx, NA, lam):
            hm = np.max(Is) / 2.0
            fwhm = dx * np.sum(Is > hm)
            print(f"Full width half maximum: {fwhm:.3f}")   
            theo_fwhm = lam / 2 / NA
            print(f"Full width half maximum of diffraction limit: {theo_fwhm:.3f}")

        I_in = np.abs(E_in)**2
        c = E_focal.shape[0]//2
        dx = self.GP.period/self.GP.out_res
        r = int(round(self.efficiency_length / dx / 2))
        I_focal = np.abs(E_focal[c - r: c + r, c - r: c + r])**2
        self.FWHM(I_focal[r], dx, NA, self.GP.lam)
        phy_size = 12 * self.GP.period
        show_intensity(I_focal, phy_size, phy_size)
        focal_efficiency = I_focal.sum() / I_in.sum() * 100
        print(f"focal efficiency: {focal_efficiency:2f}%.")
        return None
    
    # def FWHM(self, I, NA):
    #     I_HM = I.max()/2
    #     idx_h = I.size//2
    #     I_diff = np.abs(I[:idx_h] - I_HM)
    #     idx_l = np.argmin(I_diff)
    #     I_diff = np.abs(I[idx_h:] - I_HM)
    #     idx_r = np.argmin(I_diff) 
    #     dx = self.GP.period / self.GP.out_res
    #     fwhm = (idx_r + idx_h - idx_l) * dx   
    #     print(f"Full width half maximum: {fwhm:.3f}")   
    #     theo_fwhm = self.GP.lam / 2 / NA
    #     print(f"Full width half maximum of diffraction limit: {theo_fwhm:.3f}")

   
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

def resize_2d_intensity(field, step1, xs_far, ys_far):
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
    
    x = np.linspace(-phy_size_x/2, phy_size_x/2, num = field.shape[1])
    y = np.linspace(-phy_size_y/2, phy_size_y/2, num = field.shape[0])

    f_real = interpolate.interp2d(x, y, field, kind='linear')
    
    # x = np.arange(0, phy_size_x, step2)
    # y = np.arange(0, phy_size_y, step2)
    out = f_real(xs_far, ys_far)
    return out
    
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



    

