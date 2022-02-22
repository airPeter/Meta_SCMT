# standard python imports
import numpy as np
import matplotlib.pyplot as plt
import h5py

# tidy3D import
import tidy3d as td
from tidy3d import web
import warnings
class Fullwave_1D():
    def __init__(self, GP) -> None:
        self.GP =GP
    def init_sim(self, prop_dis, N, hs, res = None):
        warnings.warn("Fullwave is expensive and slow. Only do fullwave on small devices. And low resolution can be inaccurate.")
        if res == None:
            self.res = int(round(1 / self.GP.dh))
        else:
            self.res = res
        print("Fullwave resolution:", str(self.res))
        self.out_res = self.GP.dx
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

        psource = td.PlaneWave(
            injection_axis='+z',
            position=-z_size/2 + 0.5,
            source_time = td.GaussianPulse(
                frequency=fcen,
                fwidth=fwidth),
            polarization='y')
        print(psource)

        #x-z plane monitor.
        freq_mnt1 = td.FreqMonitor(center=[0, 0, 0], size=[x_size * 2, 0, z_size *2], freqs=[fcen])
        # focal plane monitor.
        freq_mnt2 = td.FreqMonitor(center=[0, 0, z_plane + self.GP.wh + prop_dis], size=[x_size * 2, y_size * 2, 0], freqs=[fcen])

        # Initialize simulation
        self.sim = td.Simulation(size=sim_size,
                            resolution=self.res,
                            structures=waveguides,
                            sources=[psource],
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
        sim = td.Simulation.import_json(data_path + self.task_name + "/simulation.json")
        sim.load_results(data_path + self.task_name + '/monitor_data.hdf5')
        self.sim = sim
        return None
    
    def vis_monitor(self,path = None):
        if path:
            sim = td.Simulation.import_json(path + "simulation.json")
            sim.load_results(path + 'monitor_data.hdf5')
            self.sim = sim
        monitors = self.sim.monitors
        mdata = self.sim.data(monitors[1])
        Ey_focal_plane = mdata['E'][1,:,0,0,0]
        I = np.abs(Ey_focal_plane)**2
        px = self.sim.grid.mesh_step[0] * np.arange(Ey_focal_plane.size)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        self.sim.viz_field_2D(monitors[0], ax=ax[0], cbar=True, comp='y', val='abs')
        ax[1].plot(px, I)
        ax[1].set_xlabel("Position [um]")
        ax[1].set_ylabel("Intensity")
        ax[1].set_title("Focal plane intensity.")
        plt.show()
        return None
    

    

