from turtle import up
import numpy as np
import matplotlib.pyplot as plt
from .PBA_model_1D import PBA_model
import torch
from torch import optim
import os
from torch.utils.tensorboard import SummaryWriter
from .utils import gen_decay_rate
from tqdm import tqdm
import warnings

class PBA_1D():
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
        self.total_size = (self.N + 2 * self.GP.Knn + 1) * self.GP.res
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
        X = np.arange(self.total_size) * self.GP.dx
        E0 = np.exp(1j * self.GP.k * np.sin(theta) * X)
        E0 = torch.tensor(E0, dtype = torch.complex64)
        E0 = E0.to(self.device)
        E_out = self.model(E0)
        E_out = E_out.cpu().detach().numpy()
        return E_out
    
    def optimize(self, notes, steps, lr = 0.01, theta = 0.0, minmax = True, substeps = 5):
        if type(theta) is tuple:
            self.optimize_range_theta(notes, steps, lr, theta, minmax = True, substeps = 5)
            print("the target is to make a perfect lens within the given incident angle range.")
        else:
            self.optimize_fix_theta(notes, steps, lr, theta)
            print("the target is to maximize the intensity of the center.")
        return None
    
    def optimize_range_theta(self, notes, steps, lr = 0.01, theta = (-np.pi/6, np.pi/6), minmax = True, substeps = 5):
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
        target_sigma = self.GP.lam / (2 * NA) / self.GP.dx
        print("the numerical aperture: ", NA, "target spot size (number of points):", target_sigma)
        
        
        for step in tqdm(range(steps + 1)):
            #rand_theta = np.random.normal(loc = (theta[0] + theta[1])/2, scale = (theta[1] - theta[0])/2)
            if minmax:
                # def custom_optimizer(grad, paras, lr):
                #     return paras - lr * grad
                hs_grads = []
                sub_losses = []
                for _ in range(substeps):
                    rand_theta = np.random.uniform(theta[0], theta[1])
                    center = int(self.total_size//2 + self.prop_dis * np.tan(rand_theta)/self.GP.dx)
                    X = np.arange(self.total_size) * self.GP.dx
                    E0 = np.exp(1j * self.GP.k * np.sin(rand_theta) * X)
                    E0 = torch.tensor(E0, dtype = torch.complex64)
                    E0 = E0.to(self.device)
                    # Compute prediction error
                    If = self.model(E0)
                    loss = loss_max_center(If, center, target_sigma)
                    sub_losses.append(loss.cpu().detach().item())
                    self.model.zero_grad()
                    loss.backward()
                    with torch.no_grad():
                        hs_grads.append(self.model.metalayer1.h_paras.grad)
                idx = np.argmax(np.array(sub_losses))
                grad = hs_grads[idx]
                self.model.metalayer1.h_paras.grad.copy_(grad)
                optimizer.step()
                #with torch.no_grad():
                    #grad = hs_grads[idx]
                    #updated_paras = custom_optimizer(grad, self.model.metalayer1.h_paras, lr)
                    #self.model.metalayer1.h_paras.copy_(updated_paras)
                        
            else:
                rand_theta = np.random.uniform(theta[0], theta[1])
                center = int(self.total_size//2 + self.prop_dis * np.tan(rand_theta)/self.GP.dx)
                X = np.arange(self.total_size) * self.GP.dx
                E0 = np.exp(1j * self.GP.k * np.sin(rand_theta) * X)
                E0 = torch.tensor(E0, dtype = torch.complex64)
                E0 = E0.to(self.device)
                # Compute prediction error
                If = self.model(E0)
                loss = loss_max_center(If, center, target_sigma)
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #grad = model.metalayer1.n_eff_paras.detach()
            #grad_peek(grad)
            if step % decay_steps == 0 and step != 0:
                my_lr_scheduler.step()
                
            if step % decay_steps == 0:    # every 1000 mini-batches...
                target_If = gaussian_func(np.arange(self.total_size), center, target_sigma)
                # ...log the running loss
                writer.add_scalar('training loss',
                                scalar_value = loss.item(), global_step = step)
                writer.add_figure('hs',
                                plot_hs(self.model.hs.cpu().detach().numpy()),
                                global_step= step)
                writer.add_figure('If',
                                plot_If(If, target_If),
                                global_step= step)       
                # loss = loss.item()
                # loss_list.append(loss)
                # print(f"loss: {loss:>7f}  [{step:>5d}/{train_steps:>5d}]")
        if minmax:
            print("final lr:", lr)
        else:
            print("final lr:", my_lr_scheduler.get_last_lr())
        out_pos = (np.arange(self.N) - (self.N - 1)/2) * self.GP.period
        out_hs = self.model.hs.cpu().detach().numpy()
        out_data = np.c_[out_pos.reshape(-1,1), out_hs.reshape(-1, 1)]
        np.savetxt(out_path + 'waveguide_widths.csv', out_data, delimiter=",")
        print('parameters saved in.', out_path)
        return None
    
    def optimize_fix_theta(self, notes, steps, lr = 0.01, theta = 0.0):
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

        X = np.arange(self.total_size) * self.GP.dx
        E0 = np.exp(1j * self.GP.k * np.sin(theta) * X)
        E0 = torch.tensor(E0, dtype = torch.complex64)
        E0 = E0.to(self.device)
        radius = self.N * self.GP.period/2
        NA =  radius/ np.sqrt(radius**2 + self.prop_dis**2)
        target_sigma = self.GP.lam / (2 * NA) / self.GP.dx
        print("the numerical aperture: ", NA, "target spot size (number of points):", target_sigma)
        center = int(self.total_size//2)
        for step in tqdm(range(steps + 1)):
            # Compute prediction error
            If = self.model(E0)
            loss = loss_max_center(If, center, target_sigma)
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
                                plot_hs(self.model.hs.cpu().detach().numpy()),
                                global_step= step)
                writer.add_figure('If',
                                plot_If(If),
                                global_step= step)       

                # loss = loss.item()
                # loss_list.append(loss)
                # print(f"loss: {loss:>7f}  [{step:>5d}/{train_steps:>5d}]")
        print("final lr:", my_lr_scheduler.get_last_lr())
        out_pos = (np.arange(self.N) - (self.N - 1)/2) * self.GP.period
        out_hs = self.model.hs.cpu().detach().numpy()
        out_data = np.c_[out_pos.reshape(-1,1), out_hs.reshape(-1, 1)]
        np.savetxt(out_path + 'waveguide_widths.csv', out_data, delimiter=",")
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
            init_hs_para = torch.tensor(init_hs_para, dtype = torch.float)
            state_dict = model.state_dict()
            state_dict['h_paras'] = init_hs_para
            model.load_state_dict(state_dict)
            #with torch.no_grad():
            #    model.metalayer1.h_paras.data = h_paras_initial
            print('initialized by loaded h_paras.')
            return None 
    def vis_field(self, E):
        px = (np.arange(self.total_size) - self.total_size//2) * self.GP.dx
        if self.far_field:
            plt.plot(px, E, label = 'intensity')
            plt.legend()
            plt.xlabel("postion [um]")
        else:
            fig, axs = plt.subplots(1, 2, figsize = (12, 6))
            
            plot1 = axs[0].plot(px, np.angle(E), label = "phase")
            axs[0].legend()
            axs[0].set_xlabel("postion [um]")
            plot2 = axs[1].plot(px, np.abs(E), label = "amp")
            axs[1].legend()
            axs[1].set_xlabel("postion [um]")
        plt.show()
        return None

def plot_hs(out_hs):
    fig = plt.figure()
    plt.plot(out_hs)
    return fig

def plot_If(If, target_If = None):
    out_If = If.cpu().detach().numpy()
    fig, axs = plt.subplots(1, 1, figsize = (8, 6))
    plt.ioff()
    axs.plot(out_If, label = 'output')
    #print('sum of intensity:', out_If.sum())
    if not (target_If is None):
        target_If = target_If * out_If.max()
        axs.plot(target_If, label = 'target normalized by max(out_If)')
    axs.legend()
    #plt.ylabel('intensity normalized by max')
    return fig

def loss_max_center(If, center, max_length):
    intensity = torch.sum(torch.abs(If[center - int(max_length//2): center + int(max_length//2)]))
    return - intensity

def gaussian_func(x, mu, sigma):
    return np.exp(- (x - mu)**2 / (2 * sigma**2))