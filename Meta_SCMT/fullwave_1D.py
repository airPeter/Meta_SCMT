# standard python imports
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import interpolate
# tidy3D import
import tidy3d as td
from tidy3d import web
import warnings
class Fullwave_1D():
    def __init__(self, GP) -> None:
        self.GP =GP
        self.sim = None
        
    def init_sim(self, prop_dis, N, hs, res = None, theta = 0):
        warnings.warn("Fullwave is expensive and slow. Only do fullwave on small devices. And low resolution can be inaccurate.")
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
        x_size = (N + 5) * self.GP.period
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
        X = (np.arange(N) - N//2) * self.GP.period
        positions = []
        for i in range(N):
                width = hs[i]
                x = X[i]
                positions.append(float(x))
                temp_wg = td.Box(center=[x, 0, z_plane + self.GP.wh/2], size=[width, y_size*2, self.GP.wh], material=material1)
                waveguides.append(temp_wg)
        if self.GP.n_sub != 1:
            material2 = td.Medium(epsilon=self.GP.n_sub**2)
            sub = td.Box(center=[0, 0, -z_size/2], size=[x_size*2, y_size*2, 2], material=material2)
            waveguides.append(sub)
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
            center=[0, 0, -z_size/2 + 0.5],
            source_time=td.GaussianPulse(fcen, fwidth),
            angle_theta=theta,
            angle_phi=0,
            direction='forward',
            waist_radius=x_size * 10,
            pol_angle=np.pi/2) #S polarization.
        #x-z plane monitor.
        freq_mnt1 = td.FreqMonitor(center=[0, 0, 0], size=[x_size, 0, z_size], freqs=[fcen])
        # focal plane monitor.
        freq_mnt2 = td.FreqMonitor(center=[0, 0, z_plane + self.GP.wh + prop_dis], size=[x_size, y_size, 0], freqs=[fcen])

        # Initialize simulation
        self.sim = td.Simulation(size=sim_size,
                            resolution=self.res,
                            structures=waveguides,
                            sources=[gaussian_beam],
                            monitors=[freq_mnt1, freq_mnt2],
                            run_time=run_time,
                            pml_layers=pml_layers)    
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
        self.sim.viz_mat_2D(normal='y', ax=ax)
        plt.show()
        return None   
    
    def upload(self, task_name):
        self.task_name = task_name     
        self.project = web.new_project(self.sim.export(), task_name=task_name)
        web.monitor_project(self.project['taskId'])
        return None
    
    def download(self, data_path):
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
            if self.sim == None:
                raise Exception("init sim first, then you can download data.")
            self.sim.load_results(path + 'monitor_data.hdf5')
        monitors = self.sim.monitors
        mdata = self.sim.data(monitors[0])
        Ey_xz_raw = mdata['E'][1,:,0,:,0].T
        step1 = self.sim.grid.mesh_step[0]
        phy_size_x = Ey_xz_raw.shape[1] * step1
        phy_size_y = Ey_xz_raw.shape[0] * step1
        index_near = int(round((1 + self.GP.wh)/step1))
        index_far = int(round((1 + self.GP.wh + self.prop_dis)/step1))
        Ey_near = Ey_xz_raw[index_near, :]
        Ey_far = Ey_xz_raw[index_far, :]
        num_steps2 = (2 * self.GP.Knn + 1 + self.N) * self.GP.res
        data_near = resize_1d(Ey_near, step1, num_steps2)
        data_far = resize_1d(Ey_far, step1, num_steps2)
        
        plt.figure()
        plt.imshow(np.abs(Ey_xz_raw), origin='lower', extent = (-phy_size_x/2, phy_size_x/2, -phy_size_y/2, phy_size_y/2))
        plt.xlabel("Position [um]")
        plt.ylabel("Position [um]")
        plt.colorbar()
        plt.show()
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        #self.sim.viz_field_2D(monitors[0], ax=ax[0], cbar=True, comp='y', val='abs')
        ax[0].plot(data_near['x'], np.angle(data_near['Ey']), label = 'near field phase')
        ax[0].set_xlabel("Position [um]")
        ax[0].legend()
        ax[1].plot(data_far['x'], np.abs(data_far['Ey'])**2, label = 'far field Intensity')
        ax[1].set_xlabel("Position [um]")
        ax[1].legend()
        plt.show()
        return Ey_xz_raw, data_near, data_far

    # def vis_monitor(self,path = None, return_data = False):
    #     if path:
    #         if self.sim == None:
    #             raise Exception("init sim first, then you can download data.")
    #         self.sim.load_results(path + 'monitor_data.hdf5')
    #     monitors = self.sim.monitors
    #     mdata = self.sim.data(monitors[0])
    #     Ey_xz_plane = mdata['E'][1,:,0,:,0]
    #     index_focal = int(round((1 + self.GP.wh + self.prop_dis) * self.res))
    #     I_focal = np.abs(Ey_xz_plane[:,index_focal])**2
    #     stride = int(round(self.res / self.out_res))
    #     Ey_xz_plane = Ey_xz_plane[::stride, ::stride].T
    #     px = 1/self.res * np.arange(I_focal.size)
    #     I_focal = I_focal[::stride]
    #     px = px[::stride]
    #     fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    #     #self.sim.viz_field_2D(monitors[0], ax=ax[0], cbar=True, comp='y', val='abs')
    #     plot1 = ax[0].imshow(np.abs(Ey_xz_plane))
    #     plt.colorbar(plot1, ax = ax[0])
    #     ax[0].set_title("field amplitude.")
    #     ax[1].plot(px, I_focal)
    #     ax[1].set_xlabel("Position [um]")
    #     ax[1].set_ylabel("Intensity")
    #     ax[1].set_title("Focal plane intensity.")
    #     plt.show()
    #     if return_data:
    #         return px, Ey_xz_plane, I_focal
    #     return None
    
# def resize_2d(field, step1, step2):
#     '''
#       tooooooooo slow!
#     input:
#         field 2D data needed to be resampled.
#         step1, current step size.
#         step2, output step size.
#     output:
#         a dict, that ['Ey'] , ['X'], ['Y'], data, and coordinate.
#     '''
#     phy_size_x = field.shape[1] * step1
#     phy_size_y = field.shape[0] * step1
    
#     x = np.linspace(0, phy_size_x, num = field.shape[1])
#     y = np.linspace(0, phy_size_y, num = field.shape[0])
#     X, Y = np.meshgrid(x, y)
#     f_real = interpolate.interp2d(X, Y, np.real(field), kind='linear')
#     f_imag = interpolate.interp2d(X, Y, np.imag(field), kind='linear')
    
#     x = np.arange(0, phy_size_x, step2)
#     y = np.arange(0, phy_size_y, step2)
#     X, Y = np.meshgrid(x, y)
#     out_real = f_real(X, Y)
#     out_img = f_imag(X, Y)
#     out_field = out_real + 1j * out_img
#     out = {}
#     out['Ey'] = out_field
#     out['X'] = X
#     out['Y'] = Y
#     return out

def resize_1d(field, step1, num_steps2):
    phy_size_x = field.shape[0] * step1
    xp = np.linspace(-phy_size_x/2, phy_size_x/2, num = field.shape[0])
    x = np.linspace(-phy_size_x/2, phy_size_x/2, num = num_steps2)
    out_field = np.interp(x, xp, field)
    out = {}
    out['Ey'] = out_field
    out['x'] = x
    return out


    

