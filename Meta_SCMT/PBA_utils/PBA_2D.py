from turtle import up
import numpy as np
import matplotlib.pyplot as plt
from .PBA_model_2D import PBA_model
import torch
from torch import optim
import os
from torch.utils.tensorboard import SummaryWriter
from ..utils import gen_decay_rate, quarter2whole, gaussian_func, toint
from tqdm import tqdm
from ..loss_lib_2D import max_center, max_corner
import warnings
import math

class PBA_2D():
    def __init__(self, GP):
        self.GP = GP
        self.model = None
        self.N = None
        self.prop_dis = None
        
    def init_model(self, N, prop_dis, init_hs = None, far_field = False):
        '''
            the layers will be used when re building the fitted model. If you change any of this default values when you do the fitting.
            you should also change at here.
        '''
        self.N = N
        self.prop_dis = prop_dis
        self.total_size = (self.N) * self.GP.out_res
        self.far_field = far_field

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if far_field:
            self.model = PBA_model(self.prop_dis, self.GP, self.N, self.total_size, near_field = False)
        else:
            self.model = PBA_model(self.prop_dis, self.GP, self.N, self.total_size, near_field = True)
        self.init_paras(self.model, self.GP.path, init_hs)
        self.model = self.model.to(self.device)
        return None
    
    def forward(self, theta = 0):
        #incident field plane wave
        x = np.arange(self.total_size) * self.GP.period / self.GP.out_res
        y = x.copy()
        X, _ = np.meshgrid(x, y)
        E0 = np.exp(1j * self.GP.k * np.sin(theta) * X)/self.total_size
        E0 = torch.tensor(E0, dtype = torch.complex64)
        E0 = E0.to(self.device)
        E_out = self.model(E0)
        E_out = E_out.cpu().detach().numpy()
        return E_out
    def optimize(self, notes, steps, lr = 0.1, theta = 0.0, quarter = False, minmax = False, substeps = 10):
        if type(theta) is tuple:
            #eg: theta = (-np.pi/6, np.pi/6)
            self.optimize_range_theta(notes, steps, lr, theta, minmax = minmax, substeps = substeps)
            print("the target is to make a perfect lens within the given incident angle range.")
        else:
            self.optimize_fix_theta(notes, steps, lr, theta, quarter)
            print("the target is to maximize the intensity of the center.")
        return None
    
    def optimize_range_theta(self, notes, steps, lr, theta, minmax, substeps):
        print("optimize lens, incident planewave angle range: " + str(theta))
        if not self.far_field:
            raise Exception("Should initalize model with far_field=True")
        out_path = 'output_PBA/'
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        out_path = out_path + notes + '/'
        writer = SummaryWriter(out_path + 'summary1')
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        decay_steps = steps // 10
        decay_rate = gen_decay_rate(steps, decay_steps)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        self.model.train()

        radius = self.N * self.GP.period/2
        NA =  radius/ np.sqrt(radius**2 + self.prop_dis**2)
        target_sigma = self.GP.lam / (2 * NA) / (self.GP.period / self.GP.out_res)
        print(f"the numerical aperture: {NA:.2f}, target spot size (number of points): {target_sigma:.2f}")

        for step in tqdm(range(steps + 1)):
            if minmax:
                hs_grads = []
                sub_losses = []
                for _ in range(substeps):
                    rand_theta = np.random.uniform(theta[0], theta[1])
                    rand_phi = np.random.uniform(0, 2 * np.pi)
                    cr = self.prop_dis * np.tan(rand_theta)/(self.GP.period / self.GP.out_res)
                    cx = toint(self.total_size//2 + cr * np.cos(rand_phi))
                    cy = toint(self.total_size//2 + cr * np.sin(rand_phi))
                    x = np.arange(self.total_size) * (self.GP.period / self.GP.out_res)
                    y = x.copy()
                    X, Y = np.meshgrid(x, y)
                    kx = self.GP.k * np.sin(rand_theta) * np.cos(rand_phi)
                    ky = self.GP.k * np.sin(rand_theta) * np.sin(rand_phi)
                    E0 = np.exp(1j * (kx * X + ky * Y))/self.total_size
                    E0 = torch.tensor(E0, dtype = torch.complex64)
                    E0 = E0.to(self.device)
                    # Compute prediction error
                    If = self.model(E0)
                    loss = max_center(If, (cy, cx), target_sigma)
                    sub_losses.append(loss.cpu().detach().item())
                    self.model.zero_grad()
                    loss.backward()
                    with torch.no_grad():
                        hs_grads.append(self.model.h_paras.grad)
                idx = np.argmax(np.array(sub_losses))
                grad = hs_grads[idx]
                self.model.h_paras.grad.copy_(grad)
                optimizer.step()
                        
            else:
                rand_theta = np.random.uniform(theta[0], theta[1])
                rand_phi = np.random.uniform(0, 2 * np.pi)
                cr = self.prop_dis * np.tan(rand_theta)/(self.GP.period / self.GP.out_res)
                cx = toint(self.total_size//2 + cr * np.cos(rand_phi))
                cy = toint(self.total_size//2 + cr * np.sin(rand_phi))
                x = np.arange(self.total_size) * (self.GP.period / self.GP.out_res)
                y = x.copy()
                X, Y = np.meshgrid(x, y)
                kx = self.GP.k * np.sin(rand_theta) * np.cos(rand_phi)
                ky = self.GP.k * np.sin(rand_theta) * np.sin(rand_phi)
                E0 = np.exp(1j * (kx * X + ky * Y))/self.total_size
                E0 = torch.tensor(E0, dtype = torch.complex64)
                E0 = E0.to(self.device)
                # Compute prediction error
                If = self.model(E0)
                loss = max_center(If, (cy, cx), target_sigma)
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step % decay_steps == 0 and step != 0:
                my_lr_scheduler.step()
                
            if step % decay_steps == 0:    # every 1000 mini-batches...
                xg = np.arange(self.total_size)
                yg = xg.copy()
                Xg, Yg = np.meshgrid(xg, yg)
                target_If = gaussian_func(Yg, cy, target_sigma) * gaussian_func(Xg, cx, target_sigma)
                # ...log the running loss
                writer.add_scalar('training loss',
                                scalar_value = loss.item(), global_step = step)
                writer.add_figure('hs',
                                plot_hs(self.model.hs.cpu().detach().numpy(), self.N),
                                global_step= step)
                writer.add_figure('If',
                                plot_If(If.cpu().detach().numpy(), f"theta: {math.degrees(rand_theta):.2f}, phi: {math.degrees(rand_phi):.2f}"),
                                global_step= step)    
                writer.add_figure('target If',
                                plot_If(target_If, f"theta: {math.degrees(rand_theta):.2f}, phi: {math.degrees(rand_phi):.2f}"),
                                global_step= step)    
        if minmax:
            print("final lr:", lr)
        else:
            print("final lr:", my_lr_scheduler.get_last_lr())
        out_hs = self.model.hs.cpu().detach().numpy()
        out_hs = out_hs.reshape(self.N, self.N)
        np.savetxt(out_path + 'waveguide_widths.csv', out_hs, delimiter=",")
        print('parameters saved in.', out_path)
        return None
            
    def optimize_fix_theta(self, notes, steps, lr, theta, quarter):
        '''
        input:
            quarter: if true, maximize the corner instead of center. If train lens, this is equal to train a quarter of lens.
        
        '''
        if not self.far_field:
            raise Exception("Should initalize model with far_field=True")
        out_path = 'output_PBA/'
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        out_path = out_path + notes + '/'
        writer = SummaryWriter(out_path + 'summary1')
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        decay_steps = steps // 10
        decay_rate = gen_decay_rate(steps, decay_steps)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        self.model.train()

        x = np.arange(self.total_size) * (self.GP.period / self.GP.out_res)
        y = x.copy()
        X, _ = np.meshgrid(x, y)
        E0 = np.exp(1j * self.GP.k * np.sin(theta) * X)/self.total_size
        E0 = torch.tensor(E0, dtype = torch.complex64)
        E0 = E0.to(self.device)
        radius = self.N * self.GP.period/2
        NA =  radius/ np.sqrt(radius**2 + self.prop_dis**2)
        target_sigma = self.GP.lam / (2 * NA) / (self.GP.period / self.GP.out_res)
        print(f"the numerical aperture: {NA:.2f}, target spot size (number of points): {target_sigma:.2f}")
        center = int(round(self.total_size//2))
        # circle = self.circle_mask(center, target_sigma)
        # circle = torch.tensor(circle, dtype = torch.float)
        # circle = circle.to(self.device)
        for step in tqdm(range(steps + 1)):
            # Compute prediction error
            If = self.model(E0)
            if quarter:
                loss = max_corner(If, self.GP.Knn, self.GP.out_res, target_sigma)
            else:
                loss = max_center(If, (center, center), target_sigma)
            #loss = - (If * circle).sum()
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #grad = model.metalayer1.n_eff_paras.detach()
            #grad_peek(grad)
            if step % decay_steps == 0 and step != 0:
                my_lr_scheduler.step()
                
            if step % decay_steps == 0:    # every 1000 mini-batches...

                # ...log the running loss
                writer.add_scalar('training loss',
                                scalar_value = loss.item(), global_step = step)
                writer.add_figure('hs',
                                plot_hs(self.model.hs.cpu().detach().numpy(), self.N),
                                global_step= step)
                writer.add_figure('If',
                                plot_If(If.cpu().detach().numpy()),
                                global_step= step)       
                # loss = loss.item()
                # loss_list.append(loss)
                # print(f"loss: {loss:>7f}  [{step:>5d}/{train_steps:>5d}]")
        print("final lr:", my_lr_scheduler.get_last_lr())
        out_hs = self.model.hs.cpu().detach().numpy()
        out_hs = out_hs.reshape(self.N, self.N)
        if quarter:
            np.savetxt(out_path + 'waveguide_widths_quarter.csv', out_hs, delimiter=",")
            out_hs = quarter2whole(out_hs)
        np.savetxt(out_path + 'waveguide_widths.csv', out_hs, delimiter=",")
        print('parameters saved in.', out_path)
        return None

    
    def init_paras(self, model, cache_path, init_hs = None):
        model.reset(cache_path)
        if init_hs is None:
            print('initialized by default h_paras.')
            return None
        else:
            #h_paras = np.genfromtxt(path, delimiter=',')
            if init_hs.max() > self.GP.h_max:
                warnings.warn("bad initial widths for waveguides.")
                print("initial widths larger than h_max, replaced by h_max")
            if init_hs.min() < self.GP.h_min:
                warnings.warn("bad initial widths for waveguides.")
                print("initial widths smaller than h_min, replaced by h_min")
            hs_paras = (init_hs - self.GP.h_min)/ (self.GP.h_max - self.GP.h_min)
            hs_paras = np.minimum(np.maximum(hs_paras, 0.01), 0.99)
            #becuase in our model we use Sigmoid  function, here, the hs paras is generated by inverse function.
            init_hs_para = np.log(hs_paras / (1 - hs_paras)) 
            init_hs_para = init_hs_para.reshape(self.N, self.N)
            init_hs_para = torch.tensor(init_hs_para, dtype = torch.float)
            state_dict = model.state_dict()
            state_dict['h_paras'] = init_hs_para
            model.load_state_dict(state_dict)
            #with torch.no_grad():
            #    model.metalayer1.h_paras.data = h_paras_initial
            print('initialized by loaded h_paras.')
            return None 
    def vis_field(self, E):
        if self.far_field:
            fig = plt.figure()
            plt.imshow(E, cmap = 'magma')
            plt.colorbar()
        else:
            fig, axs = plt.subplots(1, 2, figsize = (12, 6))
            plot1 = axs[0].imshow(np.angle(E), cmap = 'magma')
            plt.colorbar(plot1, ax = axs[0])
            plot2 = axs[1].imshow(np.abs(E), cmap = 'magma')
            plt.colorbar(plot2, ax = axs[1])
            axs[0].set_title("Phase")
            axs[1].set_title("Amp")
        plt.show()
        return None

def plot_hs(out_hs, N):
    out_hs = out_hs.reshape(N, N)
    fig = plt.figure()
    plt.imshow(out_hs, cmap = 'magma')
    plt.colorbar()
    return fig

def plot_If(If, title = None):
    size = If.shape[0]
    c = size//2
    r = np.minimum(c, 60)
    If_c = If[c-r:c+r, c-r:c+r]
    fig, axs = plt.subplots(1, 2, figsize = (12, 6))
    plot1 = axs[0].imshow(If, cmap = 'magma')
    if title:
        axs[0].set_title(title)
    plt.colorbar(plot1, ax = axs[0])
    plot2 = axs[1].imshow(If_c, cmap = 'magma')
    axs[1].set_title("Zoom in (central)")
    plt.colorbar(plot2, ax = axs[1])
    
    return fig
