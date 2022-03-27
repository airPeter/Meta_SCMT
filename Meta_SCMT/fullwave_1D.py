# standard python imports
import numpy as np
import matplotlib.pyplot as plt
import warnings
class Fullwave_1D():
    def __init__(self, GP) -> None:
        self.GP =GP
        self.sim = None
        
    def init_sim(self, prop_dis, N, hs, res = None, theta = 0, empty = False):
        # tidy3D import
        import tidy3d as td
        warnings.warn("Fullwave is expensive and slow. Only do fullwave on small devices. And low resolution can be inaccurate.")
        if hs.max() > self.GP.h_max:
            warnings.warn("initial widths larger than h_max, bad initial widths for waveguides.")
        if hs.min() < self.GP.h_min:
            warnings.warn("initial widths smaller than h_min, bad initial widths for waveguides.")
        if res == None:
            self.res = int(round(1 / self.GP.dh))
        else:
            self.res = res
        print("Fullwave resolution:", str(self.res))
        self.out_res = int(round(1 / self.GP.dx))
        self.N = N
        self.prop_dis = prop_dis
        # Simulation domain size (in micron)
        z_size = self.GP.wh + 2 + prop_dis
        x_wgs = N * self.GP.period
        #x_aper = N * self.GP.period
        x_aper = 0
        x_size = x_wgs + 2 * x_aper
        y_size = 1/self.res
        sim_size = [x_size, y_size, z_size]
        # Central frequency and bandwidth of pulsed excitation, in Hz
        fcen = td.constants.C_0 / self.GP.lam
        fwidth = fcen/10
        # Number of PML layers to use along each of the three directions.
        pml_layers = [15, 0, 15]
        # Total time to run in seconds
        run_time = max(50/fwidth, 5 * N * self.GP.period / td.constants.C_0)
        print("total running time:", run_time)

        # Lossless dielectric
        material1 = td.Medium(epsilon=self.GP.n_wg**2)
        waveguides = []
        z_plane = -z_size/2 + 1
        X = (np.arange(N) - (N - 1)/2) * self.GP.period
        positions = []
        if self.GP.n_sub != 1:
            material2 = td.Medium(epsilon=self.GP.n_sub**2)
            sub = td.Box(center=[0, 0, -z_size/2], size=[x_size*2, y_size*2, 2], material=material2)
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
                    temp_wg = td.Box(center=[x, 0, z_plane + self.GP.wh/2], size=[width, y_size*2, self.GP.wh], material=material1)
                    waveguides.append(temp_wg)
                    # temp_wg = td.Box(center=[x - x_wgs, 0, z_plane + self.GP.wh/2], size=[width, y_size*2, self.GP.wh], material=material1)
                    # waveguides.append(temp_wg)
                    # temp_wg = td.Box(center=[x + x_wgs, 0, z_plane + self.GP.wh/2], size=[width, y_size*2, self.GP.wh], material=material1)
                    # waveguides.append(temp_wg)
                    
        positions = np.array(positions)
        self.hs_with_pos = np.c_[hs.reshape((-1,1)), positions]

        # psource = td.PlaneWave(
        #     injection_axis='+z',
        #     position=-z_size/2 + 0.5,
        #     source_time = td.GaussianPulse(
        #         frequency=fcen,
        #         fwidth=fwidth),
        #     polarization='y')
        #print(psource)
        gaussian_beam = td.GaussianBeam(
            normal='z',
            center=[0, 0, -z_size/2 + 0.9],
            source_time=td.GaussianPulse(fcen, fwidth),
            angle_theta=theta,
            angle_phi=0,
            direction='forward',
            waist_radius= x_wgs * 10,
            pol_angle=np.pi/2) #S polarization.
        #x-z plane monitor.
        xz_mnt = td.FreqMonitor(center=[0, 0, 0], size=[x_size, 0, z_size], freqs=[fcen])
        # focal plane monitor.
        focal_mnt = td.FreqMonitor(center=[0, 0, z_plane + self.GP.wh + prop_dis], size=[x_wgs, y_size, 0], freqs=[fcen])
        #in_mnt = td.FreqMonitor(center=[0, 0, -z_size/2 + 0.6], size=[x_wgs, y_size, 0], freqs=[fcen])
        # Initialize simulation
        self.sim = td.Simulation(size=sim_size,
                            resolution=self.res,
                            structures=waveguides,
                            sources=[gaussian_beam],
                            monitors=[xz_mnt, focal_mnt],
                            run_time=run_time,
                            pml_layers=pml_layers)    
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
        self.sim.viz_mat_2D(normal='y', ax=ax)
        plt.show()
        return None   
    
    def upload(self, task_name):
        # tidy3D import
        from tidy3d import web
        self.task_name = task_name     
        self.project = web.new_project(self.sim.export(), task_name=task_name)
        web.monitor_project(self.project['taskId'])
        return None
    
    def download(self, data_path):
        # tidy3D import
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
    
    def vis_monitor(self,path = None):
        if path:
            if self.sim is None:
                raise Exception("init sim first, then you can download data.")
            self.sim.load_results(path + 'monitor_data.hdf5')
        monitors = self.sim.monitors
        mdata = self.sim.data(monitors[0])
        Ey_xz_raw = mdata['E'][1,:,0,:,0].T
        out_phy_size = (2 * self.GP.Knn + 1 + self.N) * self.GP.period
        step1 = self.sim.grid.mesh_step[0]
        # r = int(round(x_out_size / step1/2))
        # c = Ey_xz_raw.shape[1]//2
        # print(r, c)
        # Ey_xz_raw = Ey_xz_raw[:, c - r: c + r]
        phy_size_x = Ey_xz_raw.shape[1] * step1
        phy_size_y = Ey_xz_raw.shape[0] * step1
        index_near = int(round((1 + self.GP.wh)/step1))
        index_far = int(round((1 + self.GP.wh + self.prop_dis)/step1))
        index_in = int(round((0.6)/step1))
        Ey_near = Ey_xz_raw[index_near, :]
        Ey_far = Ey_xz_raw[index_far, :]
        Ey_in = Ey_xz_raw[index_in, :]
        num_steps2 = (2 * self.GP.Knn + 1 + self.N) * self.GP.res
        xp = np.linspace(-phy_size_x/2, phy_size_x/2, num = Ey_xz_raw.shape[1])
        x = np.linspace(-out_phy_size/2, out_phy_size/2, num_steps2)
        data_near = resize_1d(Ey_near, x, xp)
        data_far = resize_1d(Ey_far, x, xp)
        data_in = resize_1d(Ey_in, x, xp)
        plt.figure(figsize= (12, 6))
        plt.imshow(np.abs(Ey_xz_raw), origin='lower', extent = (-phy_size_x/2, phy_size_x/2, -phy_size_y/2, phy_size_y/2))
        plt.xlabel("Position [um]")
        plt.ylabel("Position [um]")
        plt.colorbar()
        plt.show()
        
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        #self.sim.viz_field_2D(monitors[0], ax=ax[0], cbar=True, comp='y', val='abs')
        ax[0].plot(data_near['x'], np.angle(data_near['Ey']), label = 'near field phase')
        ax[0].set_xlabel("Position [um]")
        ax[0].legend()
        ax[1].plot(data_far['x'], np.abs(data_far['Ey'])**2, label = 'far field Intensity')
        ax[1].set_xlabel("Position [um]")
        ax[1].legend()
        ax[2].plot(data_in['x'], np.abs(data_in['Ey'])**2, label = 'input intensity')
        ax[2].set_xlabel("Position [um]")
        ax[2].legend()
        plt.show()
        return Ey_xz_raw, data_near, data_far

def resize_1d(field, x, xp):
    out_field = np.interp(x, xp, field)
    out = {}
    out['Ey'] = out_field
    out['x'] = x
    return out


    

