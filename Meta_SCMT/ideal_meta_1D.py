'''
    currently only support ideal lens.
'''
# standard python imports
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
import cmath
import time
from .utils import lens_1D
from .SCMT_model_1D import Ideal_model
import torch

class Ideal_meta_1D():
    def __init__(self, GP) -> None:
        self.GP =GP
        self.sim = None
            
    def init_sim(self, prop_dis, N, ideal_lens = True, res = None, theta = 0, vis_path = None):
        '''
        input:
            theta: [rad]
        '''
        # tidy3D import
        import meep as mp
        self.vis_path = vis_path
        if res == None:
            self.res = int(round(1 / self.GP.dh))
        else:
            self.res = res
        print("Fullwave resolution:", str(self.res))
        self.N = N
        self.prop_dis = prop_dis
        # Simulation domain size (in micron)
        dpml = 1
        x_size = (self.N) * self.GP.period + 2 * dpml
        x_lens, phase_lens = lens_1D(int(round(x_size * self.res)), 1/self.res, self.prop_dis, 2 * np.pi / self.GP.lam)
        #create ideal lens function.
        y_size = 2 * self.GP.lam + self.prop_dis + 2 * dpml
        print(f"total_sim size x: {x_size:.2f}, y:{y_size:.2f}")
        y_plane = - y_size/2 + dpml + self.GP.lam
        cell_size = mp.Vector3(x_size,y_size)
        # Central frequency and bandwidth of pulsed excitation, in Hz
        fcen = 1 / self.GP.lam
        pml_layers = [mp.PML(dpml)]
        nonpml_vol = mp.Volume(mp.Vector3(), size=mp.Vector3(x_size-2*dpml,y_size-2*dpml))

        # k (in source medium) with correct length (plane of incidence: XY)
        k_rotate = mp.Vector3(0,fcen * 2 * math.pi,0).rotate(mp.Vector3(z=1), -theta)

        def pw_amp(k,X0):
            def _pw_amp(X):
                phase = np.interp(X.x, x_lens, phase_lens)
                return cmath.exp(1j * (k.dot(X+X0) + phase))
            return _pw_amp
        src_pt = mp.Vector3(0, y_plane - self.GP.lam/2)
        src = [mp.Source(mp.GaussianSource(fcen, fwidth=fcen/10),
                        component=mp.Ez,
                        center=src_pt,
                        size=mp.Vector3(x_size,0),
                        amp_func=pw_amp(k_rotate,src_pt))]

        self.sim = mp.Simulation(cell_size=cell_size,
                            geometry=[],
                            sources=src,
                            k_point = k_rotate, #set the Block-periodic boundary condition.
                            resolution=res,
                            force_complex_fields=True,
                            boundary_layers=pml_layers)

        #cell_vol = mp.Volume(mp.Vector3(), size=cell_size)
        self.dft_obj = self.sim.add_dft_fields([mp.Ez], fcen, 0, 1, where=nonpml_vol)  
        self.sim.init_sim()
        self.stop_condition_func = mp.stop_when_fields_decayed(dt=prop_dis, c=mp.Ez, pt=mp.Vector3(0, y_plane + prop_dis), decay_by=1e-5)
        
        self.eps_data = self.sim.get_array(vol = nonpml_vol, component=mp.Dielectric)
        self.eps_data = self.eps_data.transpose()
        plt.figure()
        plt.imshow(self.eps_data,  origin='lower', cmap = 'binary')
        if self.vis_path is None:
            plt.show()
        else:
            plt.savefig(self.vis_path + "structure.png")
        return None   

    def run(self):
        start_time = time.time()
        self.sim.run(until_after_sources= self.stop_condition_func)
        print(f"running time : {time.time() - start_time:.2f}")
        return None
    def vis(self):
        import meep as mp
        ez_data = self.sim.get_dft_array(self.dft_obj, mp.Ez, 0)
        ez_data = ez_data.transpose()
        Iz_data = np.abs(ez_data)**2
        out_phy_size = (self.N) * self.GP.period
        step1 = 1/self.res
        phy_size_x = Iz_data.shape[1] * step1
        phy_size_y = Iz_data.shape[0] * step1
        index_near = int(round((self.GP.lam)/step1))
        index_far = int(round((self.GP.lam + self.prop_dis)/step1))
        index_in = int(round((self.GP.lam/3 * 2)/step1))
        Ey_near = ez_data[index_near, :]
        Ey_far = ez_data[index_far, :]
        Ey_in = ez_data[index_in, :]
        num_steps2 = (self.N) * self.GP.res
        xp = np.linspace(-phy_size_x/2, phy_size_x/2, num = ez_data.shape[1])
        x = np.linspace(-out_phy_size/2, out_phy_size/2, num_steps2)
        data_near = resize_1d(Ey_near, x, xp)
        data_far = resize_1d(Ey_far, x, xp)
        data_in = resize_1d(Ey_in, x, xp)
        plt.figure(figsize= (12, 6))
        plt.imshow(self.eps_data,  origin='lower', cmap = 'binary', extent = (-phy_size_x/2, phy_size_x/2, -phy_size_y/2, phy_size_y/2))
        plt.imshow(Iz_data, origin='lower', cmap = 'magma', extent = (-phy_size_x/2, phy_size_x/2, -phy_size_y/2, phy_size_y/2), alpha= 0.9)
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
        ax[0].plot(data_near['x'], np.angle(data_near['Ey']), label = 'near field phase')
        ax[0].set_xlabel("Position [um]")
        ax[0].legend()
        ax[1].plot(data_far['x'], np.abs(data_far['Ey'])**2, label = 'far field Intensity')
        ax[1].set_xlabel("Position [um]")
        ax[1].legend()
        ax[2].plot(data_in['x'], np.abs(data_in['Ey'])**2, label = 'input intensity')
        ax[2].set_xlabel("Position [um]")
        ax[2].legend()
        if self.vis_path is None:
            plt.show()
        else:
            plt.savefig(self.vis_path + "near_and_far_field.png")
        return ez_data, data_in, data_near, data_far

class Ideal_meta():
    def __init__(self, GP) -> None:
        self.GP = GP
        self.model = None
        self.total_size = None
        
    def model_init(self,N, prop_dis, init_phase = None, lens = False):
        self.total_size = (N) * self.GP.res
        if init_phase is None and lens == True:
            _, init_phase = lens_1D(self.total_size, self.GP.dx, prop_dis, self.GP.k)
        self.init_phase = init_phase
        self.model = Ideal_model(prop_dis, self.GP, self.total_size)
        init_phase = torch.tensor(init_phase, dtype = torch.float)
        state_dict = self.model.state_dict()
        state_dict['phase'] = init_phase
        self.model.load_state_dict(state_dict)
        print('Model initialized.')

    def forward(self, E0 = None, theta = 0, vis = True):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        print("using device: ", self.device)
        if E0 is None:
            X = np.arange(self.total_size) * self.GP.dx
            E0 = np.exp(1j * self.GP.k * np.sin(theta) * X)/np.sqrt(self.total_size)
        E0 = E0.reshape(self.total_size,)
        I_in = (np.abs(E0)**2).sum()
        E0 = torch.tensor(E0, dtype = torch.complex64).to(self.device)
        model = self.model.to(self.device)
        with torch.no_grad():
            If = model(E0)
        If = If.cpu().numpy()
        I_out = If.sum()
        phy_x = (np.arange(self.total_size) - (self.total_size - 1)/2) * self.GP.dx
        print(f"I_in: {I_in:3f}, I_out: {I_out:3f}, I_out/I_in: {I_out/I_in:3f}.")
        if vis:
            plt.figure()
            plt.plot(phy_x, If)
            plt.show()
        return phy_x, If
    

def resize_1d(field, x, xp):
    out_field = np.interp(x, xp, field)
    out = {}
    out['Ey'] = out_field
    out['x'] = x
    return out