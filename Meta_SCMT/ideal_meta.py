'''
    given input field, output the far field of an ideal metasurface.
    eg: we can add an ideal lens phase mask and then do free space propagation.
'''
from .SCMT_model_2D import Ideal_model
import numpy as np
import torch
from .utils import lens_2D
import matplotlib.pyplot as plt
class Ideal_meta():
    def __init__(self, GP) -> None:
        self.GP = GP
        self.model = None
        self.total_size = None
        
    def model_init(self,N, prop_dis, init_phase = None, lens = False):
        self.total_size = (N) * self.GP.out_res
        self.dx = self.GP.period/self.GP.out_res
        if init_phase is None and lens == True:
            _, init_phase = lens_2D(self.total_size, self.dx, prop_dis, self.GP.k)
        self.init_phase = init_phase
        self.model = Ideal_model(prop_dis, self.GP, self.total_size, self.dx)
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
            x = np.arange(self.total_size) * self.dx
            y = x.copy()
            X, _ = np.meshgrid(x, y)
            E0 = np.exp(1j * self.GP.k * np.sin(theta) * X)
        E0 = E0.reshape(self.total_size, self.total_size)
        I_in = (np.abs(E0)**2).sum()
        E0 = torch.tensor(E0, dtype = torch.complex64).to(self.device)
        model = self.model.to(self.device)
        with torch.no_grad():
            If = model(E0)
        If = If.cpu().numpy()
        I_out = If.sum()
        print(f"I_in: {I_in:3f}, I_out: {I_out:3f}, I_out/I_in: {I_out/I_in:3f}.")
        if vis:
            phy_size_y = If.shape[0] * self.dx
            phy_size_x = phy_size_y
            show_intensity(If, phy_size_x, phy_size_y)
        return If
    
        

def show_intensity(I, phy_size_x, phy_size_y):
    plt.figure()
    plt.imshow(I, cmap = 'magma', origin='lower', extent = (-phy_size_x/2, phy_size_x/2, -phy_size_y/2, phy_size_y/2))
    plt.xlabel("Position [um]")
    plt.ylabel("Position [um]")
    plt.colorbar()
    plt.title("Intensity")
    plt.show()