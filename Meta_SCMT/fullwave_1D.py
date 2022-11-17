# standard python imports
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
import cmath
import time
from .ideal_meta_1D import Ideal_meta
from typing import Optional, Dict, Tuple, Union, List
import os

class Fullwave_1D():
    def __init__(self, GP) -> None:
        self.GP = GP
        self.sim = None

    def init_sim(self, prop_dis: float, N: int, hs: np.ndarray, res: Optional[int] = None, theta: float = 0, empty: bool = False, backend: str = 'meep', vis_path: Optional[str] = None) -> None:
        self.backend = backend
        self.vis_path = vis_path
        if backend == 'meep':
            self.meep_init_sim(prop_dis, N, hs, res, theta, empty)
        elif backend == 'tidy3d':
            if theta != 0:
                warnings.warn(
                    "should use meep for theta!=0. For the tidy3d, we use gaussian beam with a super large waist, which is very dirty.")
            self.tidy3d_init_sim(prop_dis, N, hs, res, theta, empty)

    def meep_init_sim(self, prop_dis: float, N: int, hs: np.ndarray, res: Optional[int] = None, theta: float = 0, empty: bool = False) -> None:
        '''
        input:
            theta: [rad]
        '''
        # tidy3D import
        import meep as mp
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
        self.out_res = int(round(1 / self.GP.dx))
        self.N = N
        self.prop_dis = prop_dis
        xh = N * self.GP.period
        NA = np.sin(xh / np.sqrt(xh**2 + prop_dis**2))
        print(f"numerical aperture: {NA:.2f}")
        self.min_focal_spot = self.GP.lam / 2 / NA
        self.efficiency_length = 6 * self.min_focal_spot
        # Simulation domain size (in micron)
        dpml = 1
        x_size = (self.N) * self.GP.period + 2 * dpml
        y_size = 2 * self.GP.lam + self.GP.wh + self.prop_dis + 2 * dpml
        print(f"total_sim size x: {x_size:.2f}, y:{y_size:.2f}")
        y_plane = - y_size/2 + dpml + self.GP.lam
        cell_size = mp.Vector3(x_size, y_size)
        # Central frequency and bandwidth of pulsed excitation, in Hz
        fcen = 1 / self.GP.lam
        pml_layers = [mp.PML(dpml)]
        nonpml_vol = mp.Volume(mp.Vector3(), size=mp.Vector3(
            x_size-2*dpml, y_size-2*dpml))

        geometry = []
        positions = []
        if self.GP.n_sub != 1:
            sub = mp.Block(mp.Vector3(x_size, dpml + self.GP.lam, mp.inf),
                           center=mp.Vector3(
                               0, (-y_size/2 + (dpml + self.GP.lam)/2)),
                           material=mp.Medium(epsilon=self.GP.n_wg**2))
            geometry.append(sub)

        if not empty:
            X = (np.arange(N) - (N - 1)/2) * self.GP.period
            for i in range(N):
                width = hs[i]
                x = X[i]
                positions.append(float(x))
                geometry.append(mp.Block(mp.Vector3(width, self.GP.wh, mp.inf),
                                         center=mp.Vector3(
                                             x, y_plane + self.GP.wh/2),
                                         material=mp.Medium(epsilon=self.GP.n_wg**2)))

            positions = np.array(positions)
            self.hs_with_pos = np.c_[hs.reshape((-1, 1)), positions]

        # k (in source medium) with correct length (plane of incidence: XY)
        k_rotate = mp.Vector3(0, fcen * 2 * math.pi,
                              0).rotate(mp.Vector3(z=1), -theta)

        def pw_amp(k, x0):
            def _pw_amp(x):
                return cmath.exp(1j * k.dot(x+x0))
            return _pw_amp
        src_pt = mp.Vector3(0, y_plane - self.GP.lam/2)
        src = [mp.Source(mp.GaussianSource(fcen, fwidth=fcen/10),
                         component=mp.Ez,
                         center=src_pt,
                         size=mp.Vector3(x_size, 0),
                         amp_func=pw_amp(k_rotate, src_pt))]

        self.sim = mp.Simulation(cell_size=cell_size,
                                 geometry=geometry,
                                 sources=src,
                                 # set the Block-periodic boundary condition.
                                 k_point=k_rotate,
                                 resolution=res,
                                 force_complex_fields=True,
                                 boundary_layers=pml_layers)

        #cell_vol = mp.Volume(mp.Vector3(), size=cell_size)
        self.dft_obj = self.sim.add_dft_fields(
            [mp.Ez], fcen, 0, 1, where=nonpml_vol)
        self.sim.init_sim()
        self.stop_condition_func = mp.stop_when_fields_decayed(
            dt=prop_dis, c=mp.Ez, pt=mp.Vector3(0, y_plane + self.GP.wh + prop_dis), decay_by=1e-5)

        self.eps_data = self.sim.get_array(
            vol=nonpml_vol, component=mp.Dielectric)
        self.eps_data = self.eps_data.transpose()
        plt.figure()
        plt.imshow(self.eps_data,  origin='lower', cmap='binary')
        if self.vis_path is None:
            plt.show()
        else:
            plt.savefig(self.vis_path + "structure.png")
        return None

    def run(self) -> None:
        if self.backend != 'meep':
            raise Exception("only call fullwave.run() for meep backend.\
                            for tidy3d call .upload() and .download().")
        start_time = time.time()
        self.sim.run(until_after_sources=self.stop_condition_func)
        print(f"running time : {time.time() - start_time:.2f}")
        return None

    def vis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.backend != 'meep':
            raise Exception("only call fullwave.vis() for meep backend.\
                            for tidy3d call .vis_monitor().")
        import meep as mp
        ez_data = self.sim.get_dft_array(self.dft_obj, mp.Ez, 0)
        ez_data = ez_data.transpose()
        Iz_data = np.abs(ez_data)**2
        out_phy_size = (self.N) * self.GP.period
        step1 = 1/self.res
        phy_size_x = Iz_data.shape[1] * step1
        phy_size_y = Iz_data.shape[0] * step1
        index_near = int(round((self.GP.lam + self.GP.wh)/step1))
        index_far = int(
            round((self.GP.lam + self.GP.wh + self.prop_dis)/step1))
        index_in = int(round((self.GP.lam/3 * 2)/step1))
        Ey_near = ez_data[index_near, :]
        Ey_far = ez_data[index_far, :]
        Ey_in = ez_data[index_in, :]
        num_steps2 = (self.N) * self.GP.res
        xp = np.linspace(-phy_size_x/2, phy_size_x/2, num=ez_data.shape[1])
        x = np.linspace(-out_phy_size/2, out_phy_size/2, num_steps2)
        data_near = resize_1d(Ey_near, x, xp)
        data_far = resize_1d(Ey_far, x, xp)
        data_in = resize_1d(Ey_in, x, xp)
        plt.figure(figsize=(12, 6))
        plt.imshow(self.eps_data,  origin='lower', cmap='binary',
                   extent=(-phy_size_x/2, phy_size_x/2, -phy_size_y/2, phy_size_y/2))
        plt.imshow(Iz_data, origin='lower', cmap='magma', extent=(-phy_size_x /
                                                                  2, phy_size_x/2, -phy_size_y/2, phy_size_y/2), alpha=0.9)
        plt.xlabel("Position [um]")
        plt.ylabel("Position [um]")
        plt.colorbar()
        plt.title("Intensity.")
        if self.vis_path is None:
            plt.show()
        else:
            plt.savefig(self.vis_path + "Iz.png")

        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        #self.sim.viz_field_2D(monitors[0], ax=ax[0], cbar=True, comp='y', val='abs')
        ax[0].plot(data_near['x'], np.angle(
            data_near['Ey']), label='near field phase')
        ax[0].set_xlabel("Position [um]")
        ax[0].legend()
        ax[1].plot(data_far['x'], np.abs(data_far['Ey'])
                   ** 2, label='far field Intensity')
        ax[1].set_xlabel("Position [um]")
        ax[1].legend()
        ax[2].plot(data_in['x'], np.abs(data_in['Ey'])
                   ** 2, label='input intensity')
        ax[2].set_xlabel("Position [um]")
        ax[2].legend()
        if self.vis_path is None:
            plt.show()
        else:
            plt.savefig(self.vis_path + "near_and_far_field.png")
        return ez_data, data_in, data_near, data_far

    def tidy3d_init_sim(self, prop_dis: float, N: int, hs: np.ndarray, res: Optional[int] = None, theta: float = 0, empty: bool = False) -> None:
        # tidy3D import
        import tidy3d as td
        warnings.warn(
            "Fullwave is expensive and slow. Only do fullwave on small devices. And low resolution can be inaccurate.")
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
        self.out_res = int(round(1 / self.GP.dx))
        self.N = N
        self.prop_dis = prop_dis
        xh = N * self.GP.period
        NA = np.sin(xh / np.sqrt(xh**2 + prop_dis**2))
        print(f"numerical aperture: {NA:.2f}")
        self.min_focal_spot = self.GP.lam / 2 / NA
        self.efficiency_length = 6 * self.min_focal_spot
        # Simulation domain size (in micron)
        z_size = self.GP.wh + 2 + prop_dis
        x_wgs = (N + 5) * self.GP.period
        #x_aper = N * self.GP.period
        x_aper = 0
        x_size = x_wgs + 2 * x_aper
        sim_size = [x_size, 0, z_size]
        # Central frequency and bandwidth of pulsed excitation, in Hz
        fcen = td.constants.C_0 / self.GP.lam
        fwidth = fcen/10
        # Total time to run in seconds
        run_time = max(50/fwidth, 5 * N * self.GP.period / td.constants.C_0)
        print("total running time:", run_time)

        # Lossless dielectric
        material1 = td.Medium(permittivity=self.GP.n_wg**2)
        waveguides = []
        z_plane = -z_size/2 + 1
        X = (np.arange(N) - (N - 1)/2) * self.GP.period
        positions = []
        if self.GP.n_sub != 1:
            material2 = td.Medium(permittivity=self.GP.n_sub**2)
            sub = td.Structure(
                geometry = td.Box(center=[0, 0, -z_size/2],
                         size=[x_size*2, td.inf, 2]),
                medium = material2,
                name = 'substrate')
            waveguides.append(sub)
        # gold = td.material_library.Au()
        # aper1 = td.Box(center=[-(x_aper + x_wgs)/2 - x_aper, 0, z_plane + self.GP.wh/2], size=[2 * x_aper, y_size*2, self.GP.wh], material=gold)
        # waveguides.append(aper1)
        # aper2 = td.Box(center=[(x_aper + x_wgs)/2 + x_aper, 0, z_plane + self.GP.wh/2], size=[2 * x_aper, y_size*2, self.GP.wh], material=gold)
        # waveguides.append(aper2)
        if not empty:
            for i in range(N):
                width = hs[i]
                x = X[i]
                positions.append(float(x))
                temp_wg = td.Structure(
                    geometry = td.Box(center=[x, 0, z_plane + self.GP.wh/2],
                                 size=[width, td.inf, self.GP.wh]),
                    medium = material1,
                    name = f'wg: {i}')
                waveguides.append(temp_wg)
                # temp_wg = td.Box(center=[x - x_wgs, 0, z_plane + self.GP.wh/2], size=[width, y_size*2, self.GP.wh], material=material1)
                # waveguides.append(temp_wg)
                # temp_wg = td.Box(center=[x + x_wgs, 0, z_plane + self.GP.wh/2], size=[width, y_size*2, self.GP.wh], material=material1)
                # waveguides.append(temp_wg)

        positions = np.array(positions)
        self.hs_with_pos = np.c_[hs.reshape((-1, 1)), positions]

        gaussian = td.GaussianPulse(freq0=fcen, fwidth=fwidth, phase=0)
        # gaussian_beam = td.GaussianBeam(
        #     center=[0, 0, -z_size/2 + 0.5],
        #     source_time=gaussian,
        #     angle_theta=theta,
        #     angle_phi=0,
        #     direction='forward',
        #     waist_radius=x_wgs * 10,
        #     pol_angle=np.pi/2)  # S polarization.
        psource = td.PlaneWave(
            source_time=gaussian,
            size=(td.inf, td.inf, 0),
            center=(0,0,-z_size/2 + 0.5),
            direction='+',
            pol_angle = np.pi/2, #Ey polarization.
        )
        print(psource)
        # x-z plane monitor.
        monitor_xz = td.FieldMonitor(center=[0, 0, 0], size=[
                                    x_size, 0, z_size], freqs=[fcen], name = 'xz')
        # focal plane monitor.
        monitor_far0 = td.FieldMonitor(
            center=[0, 0, z_plane + self.GP.wh + prop_dis], size=[x_wgs, 0, 0], freqs=[fcen], name = 'far')
        
        monitor_far1 = td.FluxMonitor(
            center=[0, 0, z_plane + self.GP.wh + prop_dis], size=[x_wgs, td.inf, 0], freqs=[fcen], name = 'far_flux')
        monitor_near = td.FluxMonitor(center=[
                                      0, 0, z_plane + self.GP.wh + self.GP.lam/2], size=[x_wgs, td.inf, 0], freqs=[fcen], name = 'near_flux')
        monitor_in = td.FluxMonitor(
            center=[0, 0, -z_size/2 + 0.7], size=[x_wgs, td.inf, 0], freqs=[fcen], name = 'in_flux')
        monitor_focus = td.FluxMonitor(center=[0, 0, z_plane + self.GP.wh + prop_dis], size=[
                                       self.efficiency_length, td.inf, 0], freqs=[fcen], name = 'focus_flux')
        monitor_back = td.FluxMonitor(
            center=[0, 0, -z_size/2 + 0.2], size=[x_wgs, td.inf, 0], freqs=[fcen], name = 'back_flux')
        grid_x = td.UniformGrid(dl=1/res)
        # in z, use an automatic nonuniform mesh with the wavelength being the "unit length"
        grid_z = td.AutoGrid(min_steps_per_wvl=int(self.GP.lam * res))
        #grid_z = td.UniformGrid(dl=1/res)
        grid_spec = td.GridSpec(wavelength=self.GP.lam, grid_x=grid_x, grid_z=grid_z)
        # Initialize simulation
        self.sim = td.Simulation(size=sim_size,
                                grid_spec=grid_spec,
                                structures=waveguides,
                                sources=[psource],
                                monitors=[monitor_xz, monitor_in, monitor_near,
                                        monitor_far0, monitor_far1,  monitor_focus, monitor_back],
                                run_time=run_time,
                                boundary_spec=td.BoundarySpec(
                                    x=td.Boundary.absorber(),
                                    y=td.Boundary.absorber(),
                                    z=td.Boundary.pml(num_layers=15)
                                ))
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
        self.sim.plot(y=0, ax=ax)
        if self.vis_path is None:
            plt.show()
        else:
            plt.savefig(self.vis_path + "structure.png")
        return None

    def upload(self, task_name: str) -> None:
        if self.backend != 'tidy3d':
            raise Exception("call fullwave.run() for meep backend.\
                            for tidy3d call .upload() and .download().")
        # tidy3D import
        from tidy3d import web
        self.task_name = task_name
        self.task_id = web.upload(simulation=self.sim, task_name=task_name)
        web.start(self.task_id)
        web.monitor(self.task_id)
        return None

    def download(self, data_path: str) -> None:
        if self.backend != 'tidy3d':
            raise Exception("call fullwave.run() for meep backend.\
                            for tidy3d call .upload() and .download().")
        # Show the output of the log file
        # if self.sim == None:
        #     raise Exception("init sim first, then you can download data.")
            #self.sim = td.Simulation.import_json(data_path + self.task_name + "/simulation.json")
        from tidy3d import web
        replace = True
        if os.path.exists(data_path + self.task_name  + '/monitor_data.hdf5'):
            replace = False
        print('load data from sever: ', replace)
        self.sim_data = web.load(
            self.task_id, path=data_path + self.task_name  + '/monitor_data.hdf5', replace_existing=replace)
        return None

    def vis_monitor(self, path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.backend != 'tidy3d':
            raise Exception("only call fullwave.vis() for meep backend.\
                            for tidy3d call .vis_monitor().")
        if path:
            if self.sim is None:
                raise Exception("init sim first, then you can download data.")
            from tidy3d import SimulationData
            self.sim_data = SimulationData.from_file(path)
        else:
            if not hasattr(self, 'sim_data'):
                raise Exception("Specify path argument to load sim data.")
            
        Ey_xz_raw = np.squeeze(self.sim_data['xz'].Ey).T
        out_phy_size = (self.N) * self.GP.period
        step1 = self.sim_data['xz'].monitor.size[0] / Ey_xz_raw.shape[1]
        # r = int(round(x_out_size / step1/2))
        # c = Ey_xz_raw.shape[1]//2
        # print(r, c)
        # Ey_xz_raw = Ey_xz_raw[:, c - r: c + r]
        phy_size_x = Ey_xz_raw.shape[1] * step1
        phy_size_y = Ey_xz_raw.shape[0] * step1
        index_near = int(round((1 + self.GP.wh)/step1))
        index_far = int(round((1 + self.GP.wh + self.prop_dis)/step1))
        index_in = int(round((0.2)/step1))
        Ey_near = Ey_xz_raw[index_near, :]
        Ey_far = Ey_xz_raw[index_far, :]
        Ey_in = Ey_xz_raw[index_in, :]
        num_steps2 = (self.N) * self.GP.res
        xp = np.linspace(-phy_size_x/2, phy_size_x/2, num=Ey_xz_raw.shape[1])
        x = np.linspace(-out_phy_size/2, out_phy_size/2, num_steps2)
        data_near = resize_1d(Ey_near, x, xp)
        data_far = resize_1d(Ey_far, x, xp)
        data_in = resize_1d(Ey_in, x, xp)
        plt.figure(figsize=(12, 6))
        plt.imshow(np.abs(Ey_xz_raw), origin='lower',
                   extent=(-phy_size_x/2, phy_size_x/2, -phy_size_y/2, phy_size_y/2))
        plt.xlabel("Position [um]")
        plt.ylabel("Position [um]")
        plt.colorbar()
        plt.title("abs(field)")
        if self.vis_path is None:
            plt.show()
        else:
            plt.savefig(self.vis_path + "Iz.png")

        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        #self.sim.viz_field_2D(monitors[0], ax=ax[0], cbar=True, comp='y', val='abs')
        ax[0].plot(data_near['x'], np.angle(
            data_near['Ey']), label='near field phase')
        ax[0].set_xlabel("Position [um]")
        ax[0].legend()
        ax[1].plot(data_far['x'], np.abs(data_far['Ey'])
                   ** 2, label='far field Intensity')
        ax[1].set_xlabel("Position [um]")
        ax[1].legend()
        ax[2].plot(data_in['x'], np.abs(data_in['Ey'])
                   ** 2, label='input intensity')
        ax[2].set_xlabel("Position [um]")
        ax[2].legend()
        if self.vis_path is None:
            plt.show()
        else:
            plt.savefig(self.vis_path + "near_and_far_field.png")
        return Ey_xz_raw, data_in, data_near, data_far

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

        xs_far = self.sim_data['far'].Ey.coords['x'].values
        E_far = np.squeeze(self.sim_data['far'].Ey)
        I_far = np.square(np.abs(E_far))
        fwhm = FWHM(xs_far, I_far)
        print(f'fwhm = {fwhm:.4f} um, {(fwhm / self.GP.lam):.2f} $\lambda$')
        ideal_meta = Ideal_meta(self.GP)
        ideal_meta.model_init(self.N, self.prop_dis, lens=True)
        xs_ideal, I_ideal = ideal_meta.forward(vis=False)
        I_ideal = np.interp(xs_far, xs_ideal, I_ideal)
        power_ideal = np.sum(I_ideal)
        r = int(round(self.efficiency_length / np.mean(np.diff(xs_far)) / 2))
        c = I_ideal.shape[0]//2
        I_focus_ideal = I_ideal[c - r: c + r]
        power_ideal_focus = I_focus_ideal.sum()
        print(
            f'Ideal focal area power/total_far_field_power = {power_ideal_focus/ power_ideal * 100:.2f}%')
        I_far_normalized = I_far / np.sum(I_far) * power_ideal
        strehl_ratio = np.max(I_far_normalized) / np.max(I_ideal)

        #diff_lim = airy(xs)
        fwhm_airy = FWHM(xs_far, I_ideal)
        print(
            f'fwhm_airy = {fwhm_airy:.4f} um,  {(fwhm_airy / self.GP.lam):.2f} $\lambda$')

        plt.plot(xs_far / self.GP.lam, I_far_normalized, label='measured')
        plt.plot(xs_far / self.GP.lam, I_ideal, label='diffraction limited')
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

    # def results_analysis(self, data_in, data_near, data_far, ideal_far):
    #     def FWHM(dx, Is):
    #         hm = np.max(Is) / 2.0
    #         return dx * np.sum(Is > hm)
    #     I_far = np.abs(data_far['Ey'])**2
    #     fwhm = FWHM(self.GP.dx, I_far)
    #     print(f'fwhm = {fwhm:.4f} um, {(fwhm / self.GP.lam):.2f} $\lambda$')
    #     #theo_fwhm = 1.025 * self.GP.lam * self.prop_dis / self.x_size

    #     I_ideal = np.abs(ideal_far['Ey'])**2
    #     power_ideal = np.sum(I_ideal)
    #     r = int(round(self.efficiency_length / self.GP.dx / 2))
    #     c = I_ideal.shape[0]//2
    #     I_focus_ideal = I_ideal[c - r: c + r]
    #     I_focus = I_far[c - r: c + r]
    #     power_ideal_focus = I_focus_ideal.sum()
    #     print(f'Ideal focal area power/total_far_field_power = {power_ideal_focus/ power_ideal * 100:.2f}%')
    #     I_far_normalized = I_far / np.sum(I_far) * power_ideal
    #     strehl_ratio = np.max(I_far_normalized) / np.max(I_ideal)

    #     #diff_lim = airy(xs)
    #     fwhm_airy = FWHM(self.GP.dx, I_ideal)
    #     print(f'fwhm_airy = {fwhm_airy:.4f} um,  {(fwhm_airy / self.GP.lam):.2f} $\lambda$')

    #     plt.plot(data_far['x'], I_far_normalized, label='measured')
    #     plt.plot(ideal_far['x'], I_ideal, label='diffraction limited')
    #     #plt.xlim([-2, 2])
    #     plt.legend()
    #     plt.title(f'FWHM = {(fwhm / self.GP.lam):.4f} $\lambda$, {(fwhm*1000):.2f} nm , Strehl ratio = {strehl_ratio:.4f}')
    #     plt.xlabel('axis position ($\lambda_0$)')
    #     plt.ylabel('intensity (normalized)')
    #     plt.show()

    #     power_in = np.sum(np.abs(data_in['Ey'])**2)
    #     power_near = np.sum(np.abs(data_near['Ey'])**2)
    #     power_far = np.sum(I_far)
    #     power_focus = np.sum(I_focus)
    #     eff_trans = power_near / power_in
    #     eff_far = power_far / power_in
    #     eff_focus = power_focus / power_in

    #     print(f'transmission efficiency = {(eff_trans*100):.2f}%')
    #     print(f'far field efficiency = {(eff_far*100):.2f}%')
    #     print(f'focusing efficiency = {(eff_focus*100):.2f}%')
    #     print(f'focal area power/total_far_field_power = {power_focus / power_far * 100:.2f}%')


def resize_1d(field: np.ndarray, x: np.ndarray, xp: np.ndarray) -> Dict:
    out_field = np.interp(x, xp, field)
    out = {}
    out['Ey'] = out_field
    out['x'] = x
    return out
