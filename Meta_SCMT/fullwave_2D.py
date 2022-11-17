# standard python imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import warnings
from .SCMT_utils.SCMT_model_2D import freespace_layer
import torch
from tqdm import tqdm
from .cmap import paper_cmap
from .ideal_meta import Ideal_meta
from typing import Optional, Tuple
import os

class Fullwave_2D():
    def __init__(self, GP) -> None:
        self.GP = GP
        self.sim = None

    def init_sim(self, prop_dis: float, N: int, hs: float, res: Optional[float] = None, theta: float = 0, empty: bool = False) -> None:
        '''
        input:
            hs: the 2d array of waveguides need to be simulated. shape [N, N]
            theta: if theta != 0, using gaussian beam, else using plane wave.
        '''
        # tidy3D lazy import
        import tidy3d as td
        warnings.warn("Fullwave is expensive and slow. You can set the prop_dis = 0, and use near_to_far to get the far field. Only do fullwave on small devices. And low resolution can be inaccurate.")
        if hs.max() > self.GP.h_max:
            warnings.warn(
                "initial widths larger than h_max, bad initial widths for waveguides.")
        if hs.min() < self.GP.h_min:
            warnings.warn(
                "initial widths smaller than h_min, bad initial widths for waveguides.")
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
        x_size = (N + 5) * self.GP.period
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
        self.fcen = fcen
        fwidth = fcen/10

        # Total time to run in seconds
        run_time = max(50/fwidth, 5 * N * self.GP.period / td.constants.C_0)
        print("total running time:", run_time)

        # Lossless dielectric
        material1 = td.Medium(permittivity=self.GP.n_wg**2)
        waveguides = []
        z_plane = -z_size/2 + spacing
        X = (np.arange(N) - N/2) * self.GP.period + self.GP.period/2
        Y = X[::-1]
        positions = []
        if not empty:
            if self.GP.n_sub != 1:
                material2 = td.Medium(permittivity=self.GP.n_sub**2)
                sub = td.Structure(
                    geometry = td.Box(center=[0, 0, -z_size + spacing],
                            size=[td.inf, td.inf, z_size]),
                    medium = material2,
                    name = 'substrate')
                waveguides.append(sub)
            for i in range(N):
                for j in range(N):
                    width = hs[i, j]
                    x = X[j]
                    y = Y[i]
                    positions.append([float(x), float(y)])
                    temp_wg = td.Structure(
                        geometry = td.Box(center=[
                                        x, y, z_plane + self.GP.wh/2],
                                    size=[width, width, self.GP.wh]),
                        medium = material1,
                        name = f'wg: {i}_{j}')
                    waveguides.append(temp_wg)
            positions = np.array(positions)
            self.hs_with_pos = np.c_[hs.reshape((-1, 1)), positions]
        gaussian = td.GaussianPulse(freq0=fcen, fwidth=fwidth, phase=0)

        source = td.PlaneWave(
            source_time=gaussian,
            size=(td.inf, td.inf, 0),
            center=(0,0,-z_size/2 + spacing/2),
            direction='+',
            pol_angle = np.pi/2, #Ey polarization.
            angle_theta = theta
        )
        print(source)

        monitor_back_flux = td.FluxMonitor(
            center=[0., 0., -z_size/2 + 2*self.dl],
            size=[x_size, x_size, 0],
            freqs=[fcen],
            name='back_flux')
        
        monitor_in_flux = td.FluxMonitor(
            center=[0., 0., -z_size/2 + spacing - 2*self.dl],
            size=[x_size, x_size, 0],
            freqs=[fcen],
            name='in_flux')
        monitor_near_flux = td.FluxMonitor(
            center=[0., 0., -z_size/2 + spacing + self.GP.wh + spacing/2],
            size=[x_size, x_size, 0],
            freqs=[fcen],
            name='near_flux')
        monitor_xy_flux = td.FluxMonitor(
            center=[0., 0., -z_size/2 + spacing + self.GP.wh + prop_dis],
            size=[x_size, x_size, 0],
            freqs=[fcen],
            name='far_flux')
        monitor_eff_flux = td.FluxMonitor(
            center=[0., 0., -z_size/2 + spacing + self.GP.wh + prop_dis],
            size=[self.efficiency_length, self.efficiency_length, 0],
            freqs=[fcen],
            name='focus_flux')       
        
        monitor_near = td.FieldMonitor(
            center=[0., 0., -z_size/2 + spacing + self.GP.wh + spacing/2],
            size=[x_size, x_size, 0],
            freqs=[fcen],
            fields=('Ey',),
            name='near')
        monitor_xy = td.FieldMonitor(
            center=[0., 0., -z_size/2 + spacing + self.GP.wh + prop_dis],
            size=[x_size, x_size, 0],
            freqs=[fcen],
            fields=('Ey',),
            name='far')
        monitor_eff = td.FieldMonitor(
            center=[0., 0., -z_size/2 + spacing + self.GP.wh + prop_dis],
            size=[self.efficiency_length, self.efficiency_length, 0],
            freqs=[fcen],
            fields=('Ey',),
            name='focus')
        monitor_CS1 = td.FieldMonitor(
            center=[0., 0., 0],
            size=[x_size, 0, z_size],
            freqs=[fcen],
            fields=('Ey',),
            name='CS1')

        self.monitors = [monitor_back_flux, monitor_in_flux, monitor_xy_flux, monitor_eff_flux, monitor_near_flux,
                         monitor_near, monitor_xy, monitor_eff, monitor_CS1]
        
        grid_x = td.UniformGrid(dl=1/res)
        grid_y = td.UniformGrid(dl=1/res)
        grid_z = td.AutoGrid(min_steps_per_wvl=int(self.GP.lam * res))
        grid_spec = td.GridSpec(wavelength=self.GP.lam, grid_x=grid_x, grid_y=grid_y, grid_z=grid_z)
        
        self.sim = td.Simulation(size=sim_size,
                                grid_spec=grid_spec,
                                structures=waveguides,
                                sources=[source],
                                monitors=self.monitors,
                                run_time=run_time,
                                boundary_spec=td.BoundarySpec(
                                    x=td.Boundary.absorber(),
                                    y=td.Boundary.absorber(),
                                    z=td.Boundary.pml(num_layers=15)
                                ))
        _, ax = plt.subplots(1, 2, figsize=(12, 6))
        self.sim.plot(x=0, ax=ax[0])
        self.sim.plot(z=z_plane  + self.GP.wh/2, ax=ax[1], source_alpha=0.9)      
        plt.show()
        return None

    def upload(self, task_name: str) -> None:
        # tidy3D lazy import
        from tidy3d import web
        self.task_name = task_name
        self.task_id = web.upload(simulation=self.sim, task_name=task_name)
        web.start(self.task_id)
        web.monitor(self.task_id)
        return None

    def download(self, data_path: str, taskId: Optional[str] = None, task_name: Optional[str] = None) -> None:
        # tidy3D lazy import
        # Show the output of the log file
        from tidy3d import web
        if taskId:
            self.task_id = taskId
        replace = True
        if os.path.exists(data_path + self.task_name  + '/monitor_data.hdf5'):
            replace = False
        print('load data from sever: ', replace)
        self.sim_data = web.load(
            self.task_id, path=data_path + self.task_name  + '/monitor_data.hdf5', replace_existing=replace)
        return None

    def near_to_far(self, En: np.ndarray, prop_dis: float) -> np.ndarray:
        dx = self.GP.period / self.GP.out_res
        freelayer1 = freespace_layer(prop_dis, self.GP.lam, En.shape[0], dx)
        En = torch.tensor(En, dtype=torch.complex64)
        with torch.no_grad():
            Ef = freelayer1(En)
        out_Ef = Ef.cpu().numpy()
        return out_Ef

    def results_analysis(self, path: Optional[str] = None) -> None:
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
            from tidy3d import SimulationData
            self.sim_data = SimulationData.from_file(path)
        else:
            if not hasattr(self, 'sim_data'):
                raise Exception("Specify path argument to load sim data.")

        E_far = np.squeeze(self.sim_data['far'].Ey).T
        I_far = np.square(np.abs(E_far))
        xs_far = self.sim_data['far'].Ey.coords['x'].values
        ys_far = self.sim_data['far'].Ey.coords['y'].values
        ideal_meta = Ideal_meta(self.GP)
        ideal_meta.model_init(self.N, self.prop_dis, lens=True)
        I_ideal = ideal_meta.forward()
        I_ideal = resize_2d_intensity(I_ideal, ideal_meta.dx, xs_far, ys_far)
        power_ideal = np.sum(I_ideal)
        r = int(round(self.efficiency_length / np.mean(np.diff(xs_far)) / 2))
        c = I_ideal.shape[0]//2
        I_focus_ideal = I_ideal[c - r: c + r, c - r: c + r]
        power_ideal_focus = I_focus_ideal.sum()
        print(
            f'Ideal focal area power/total_far_field_power = {power_ideal_focus/ power_ideal * 100:.2f}%')
        I_far_normalized = I_far / np.sum(I_far) * power_ideal
        strehl_ratio = np.max(I_far_normalized) / np.max(I_ideal)
        I_ideal_1D = I_ideal[I_ideal.shape[0]//2]
        I_ideal_normalized_1D = I_ideal_1D/I_ideal_1D.max()
        I_far_normalized_1D = I_far_normalized[I_far_normalized.shape[0]//2]
        I_far_normalized_1D = I_far_normalized_1D/I_ideal_1D.max()

        fwhm_airy = FWHM(xs_far, I_ideal_1D)
        print(
            f'fwhm_airy = {fwhm_airy:.4f} um,  {(fwhm_airy / self.GP.lam):.2f} $\lambda$')

        fwhm = FWHM(xs_far, I_far_normalized_1D)
        print(f'fwhm = {fwhm:.4f} um,  {(fwhm / self.GP.lam):.2f} $\lambda$')
        fig = plt.figure()
        plt.plot(xs_far / self.GP.lam, I_far_normalized_1D, label='measured')
        plt.plot(xs_far / self.GP.lam, I_ideal_normalized_1D,
                 label='diffraction limited')
        plt.xlim([-2, 2])
        plt.legend()
        plt.title(
            f'FWHM = {(fwhm / self.GP.lam):.4f} $\lambda$, {(fwhm*1000):.2f} nm , Strehl ratio = {strehl_ratio:.4f}')
        plt.xlabel('axis position ($\lambda_0$)')
        plt.ylabel('intensity (normalized)')
        plt.show()
        
        power_back = self.sim_data['back_flux'].flux[0]
        power_in = self.sim_data['in_flux'].flux[0]- power_back
        power_near = self.sim_data['near_flux'].flux[0]
        power_far = self.sim_data['far_flux'].flux[0]
        power_focus = self.sim_data['focus_flux'].flux[0]
        
        eff_trans = power_near / power_in
        eff_far = power_far / power_in
        eff_focus = power_focus / power_in

        print(f'transmission efficiency = {(eff_trans*100):.2f}%')
        print(f'far field efficiency = {(eff_far*100):.2f}%')
        print(f'focusing efficiency = {(eff_focus*100):.2f}%')
        print(
            f'focal area power/total_far_field_power = {power_focus / power_far * 100:.2f}%')
        return fig
    
    def vis_monitor(self, path: Optional[str] = None, tidy3d_viz: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if path:
            if self.sim is None:
                raise Exception("init sim first, then you can download data.")
            from tidy3d import SimulationData
            self.sim_data = SimulationData.from_file(path)
        else:
            if not hasattr(self, 'sim_data'):
                raise Exception("Specify path argument to load sim data.")

        if tidy3d_viz:
            _, ax = plt.subplots(1, 3, figsize=(18, 6))
            self.sim_data.plot_field('CS1', 'Ey', val = 'abs', ax = ax[0])
            self.sim_data.plot_field('focus', 'Ey', val = 'abs', ax = ax[1])
            self.sim_data.plot_field('near', 'Ey', val = 'abs', ax = ax[2])
            plt.show()

        Ey_far = np.squeeze(self.sim_data['far'].Ey).T
        step1 = self.sim_data['far'].monitor.size[0] / Ey_far.shape[1]
        step2 = self.GP.period / self.GP.out_res

        Ey_near = np.squeeze(self.sim_data['near'].Ey).T
        Ey_near_FW = Ey_near
        Ey_near = resize_2d(Ey_near, step1, step2)
        phy_size_x = Ey_near.shape[1] * step2
        phy_size_y = Ey_near.shape[0] * step2
        show_field(Ey_near, phy_size_x, phy_size_y)

        Ey_far_FW = Ey_far
        Ey_far = resize_2d(Ey_far, step1, step2)
        I_far = np.abs(Ey_far)**2
        show_intensity(I_far, phy_size_x, phy_size_y)

        Ey_eff = np.squeeze(self.sim_data['focus'].Ey).T
        I_eff = np.abs(Ey_eff)**2
        show_intensity(I_eff, self.efficiency_length, self.efficiency_length)
        print("return field: Ey_near, Ey_near_FW, Ey_far, Ey_far_FW. Ey_near is downsampled Ey_near_FW, so that resolution is same with cmt model.")
        return Ey_near, Ey_near_FW, Ey_far, Ey_far_FW

    def focal_efficiency(self, E_in: np.ndarray, E_focal: np.ndarray, NA: float):

        def FWHM(Is, dx, NA, lam):
            hm = np.max(Is) / 2.0
            fwhm = dx * np.sum(Is > hm)
            print(f"Full width half maximum: {fwhm:.3f}")
            theo_fwhm = lam / 2 / NA
            print(
                f"Full width half maximum of diffraction limit: {theo_fwhm:.3f}")

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


def show_field(E: np.ndarray, phy_size_x: float, phy_size_y: float) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plot0 = ax[0].imshow(np.angle(E), cmap='magma', origin='lower',
                         extent=(-phy_size_x/2, phy_size_x/2, -phy_size_y/2, phy_size_y/2))
    ax[0].set_xlabel("Position [um]")
    ax[0].set_ylabel("Position [um]")
    ax[0].set_title("Phase")
    plt.colorbar(plot0, ax=ax[0])
    plot0 = ax[1].imshow(np.abs(E), cmap='magma', origin='lower',
                         extent=(-phy_size_x/2, phy_size_x/2, -phy_size_y/2, phy_size_y/2))
    ax[1].set_xlabel("Position [um]")
    ax[1].set_ylabel("Position [um]")
    ax[1].set_title("Amplitude")
    plt.colorbar(plot0, ax=ax[1])
    plt.show()
    return None


def show_intensity(I: np.ndarray, phy_size_x: float, phy_size_y: float) -> None:
    plt.figure()
    plt.imshow(I, cmap='magma', origin='lower', extent=(-phy_size_x /
                                                        2, phy_size_x/2, -phy_size_y/2, phy_size_y/2))
    plt.xlabel("Position [um]")
    plt.ylabel("Position [um]")
    plt.colorbar()
    plt.title("Intensity")
    plt.show()


def resize_2d_intensity(field: np.ndarray, step1: float, xs_far: np.ndarray, ys_far: np.ndarray) -> np.ndarray:
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

    x = np.linspace(-phy_size_x/2, phy_size_x/2, num=field.shape[1])
    y = np.linspace(-phy_size_y/2, phy_size_y/2, num=field.shape[0])

    f_real = interpolate.interp2d(x, y, field, kind='linear')

    # x = np.arange(0, phy_size_x, step2)
    # y = np.arange(0, phy_size_y, step2)
    out = f_real(xs_far, ys_far)
    return out


def resize_2d(field: np.ndarray, step1: float, step2: float) -> np.ndarray:
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

    x = np.linspace(0, phy_size_x, num=field.shape[1])
    y = np.linspace(0, phy_size_y, num=field.shape[0])

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
