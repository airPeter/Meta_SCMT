'''
    multi wavelength SCMT model, only support normal incidence.
'''

import numpy as np
import matplotlib.pyplot as plt
from .SCMT_model_2D_lam import Metalayer, SCMT_Model
import torch
from torch import optim
import os
from torch.utils.tensorboard import SummaryWriter
from .utils import gen_decay_rate, quarter2whole
from tqdm import tqdm
from .loss_lib_2D import max_center, max_corner
import cv2
import warnings

class SCMT_2D():
    def __init__(self, GP):
        self.GP = GP
        self.model = None
        self.APPROX = None
        self.COUPLING = None
        self.Euler_steps = None
        self.N = None
        self.Ni = None
        self.k_row = None
        self.prop_dis = None
        
    def init_model(self, N, prop_dis, APPROX, Ni = None, k_row = None, Euler_steps = None, devs = None, COUPLING = True, init_hs = None, far_field = False):
        '''
            the layers will be used when re building the fitted model. If you change any of this default values when you do the fitting.
            you should also change at here.
            
        '''
        self.N = N
        self.prop_dis = prop_dis
        self.total_size = (self.N) * self.GP.out_res
        self.far_field = far_field
        if Ni == None:
            self.Ni = 11 * N
        else:
            self.Ni = Ni
            if self.Ni%N != 0:
                raise Exception("Ni should be divied by N.")
            if self.Ni < 5 * N:
                raise Exception("Ni should be at least 5 * N.")
        if k_row == None:
            self.k_row = N
        else:
            self.k_row = k_row
        self.APPROX = APPROX
        self.COUPLING = COUPLING
        if devs is None:
            self.devs = ["cuda"] if torch.cuda.is_available() else ["cpu"]
        else:
            self.devs = devs
        print("Optimizing by devs:", str(self.devs))
        if Euler_steps == None:
            self.Euler_steps = 1000
        else:
            self.Euler_steps = Euler_steps
        if far_field:
            self.model = SCMT_Model(self.prop_dis, self.Euler_steps, self.devs, self.GP, self.COUPLING, self.APPROX, self.Ni, self.k_row, self.N)
        else:
            self.model = Metalayer(self.Euler_steps, self.devs, self.GP, self.COUPLING, self.APPROX, self.Ni, self.k_row, self.N)
        self.init_paras(self.model, init_hs)
        return None
    
    def forward(self):
        '''
        output:
            if far_field == True, output is intensity otherwise is field.
        '''
        #incident field plane wave
        E0 = np.ones((self.total_size, self.total_size))/self.total_size
        E0 = torch.tensor(E0, dtype = torch.complex64)
        E0 = E0.to(self.devs[0])
        E_out = self.model(E0)
        E_out = [E.cpu().detach().numpy() for E in E_out]
        return E_out
    
    def optimize(self, notes, steps, lr = 0.1, minmax = False, quarter = False, loss_weights = None):
        '''
        input:
            quarter: if true, maximize the corner instead of center. If train lens, this is equal to train a quarter of lens.
            loss_weights: we use loss_weights to compensate the intensity difference between different lam, when optimizing by minmax method.
        '''
        if not self.far_field:
            raise Exception("Should initalize model with far_field=True")
        if self.COUPLING:
            out_path = 'output_cmt/'
        else:
            out_path = 'output_no_coupling/'
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        out_path = out_path + notes + '/'
        writer = SummaryWriter(out_path + 'summary1')
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        decay_steps = steps // 10
        decay_rate = gen_decay_rate(steps, decay_steps)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        self.model.train()
        E0 = E0 = np.ones((self.total_size, self.total_size))/self.total_size
        E0 = torch.tensor(E0, dtype = torch.complex64)
        E0 = E0.to(self.devs[0])
        radius = self.N * self.GP.period/2
        NA =  radius/ np.sqrt(radius**2 + self.prop_dis**2)
        #target_sigma = (min(self.GP.lams) + max(self.GP.lams)) * 0.5 / (2 * NA) / (self.GP.period / self.GP.out_res)
        target_sigma = min(self.GP.lams)  / (2 * NA) / (self.GP.period / self.GP.out_res)
        print(f"the numerical aperture: {NA:.2f}, target spot size (number of points): {target_sigma:.2f}")
        center = int(round(self.total_size//2))
        # circle = self.circle_mask(center, target_sigma)
        # circle = torch.tensor(circle, dtype = torch.float)
        # circle = circle.to(self.devs[0])
        for step in tqdm(range(steps + 1)):
            # Compute prediction error
            Ifs = self.model(E0)
            if quarter:
                if minmax:
                    losses = [max_corner(If, target_sigma) for If in Ifs]
                    if not (loss_weights is None):
                        for idx, w in enumerate(loss_weights):
                            losses[idx] = losses[idx] * w
                    loss = - np.inf
                    for tmp_loss in losses:
                        if tmp_loss > loss:
                            loss = tmp_loss
                else:
                    idx = np.random.randint(0, len(self.GP.lams))
                    loss = max_corner(If, target_sigma)
            else:
                if minmax:
                    losses = [max_center(If, (center, center), target_sigma) for If in Ifs]
                    if not (loss_weights is None):
                        for idx, w in enumerate(loss_weights):
                            losses[idx] = losses[idx] * w
                    loss = - np.inf
                    for tmp_loss in losses:
                        if tmp_loss > loss:
                            loss = tmp_loss
                else:
                    idx = np.random.randint(0, len(self.GP.lams))
                    loss = max_center(Ifs[idx], (center, center), target_sigma)

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
                                plot_hs(self.model.metalayer1.hs.cpu().detach().numpy(), self.N),
                                global_step= step)
                for i, If in enumerate(Ifs):
                    out_If = If.cpu().detach().numpy()
                    writer.add_figure(f"If, lam: {self.GP.lams[i]} um",
                                    plot_If(out_If),
                                    global_step= step)      
                # loss = loss.item()
                # loss_list.append(loss)
                # print(f"loss: {loss:>7f}  [{step:>5d}/{train_steps:>5d}]")
        print("final lr:", my_lr_scheduler.get_last_lr())
        out_hs = self.model.metalayer1.hs.cpu().detach().numpy()
        out_hs = out_hs.reshape(self.N, self.N)
        if quarter:
            np.savetxt(out_path + 'waveguide_widths_quarter.csv', out_hs, delimiter=",")
            out_hs = quarter2whole(out_hs)
        np.savetxt(out_path + 'waveguide_widths.csv', np.round(out_hs,3), delimiter=",")
        print('parameters saved in.', out_path)
        return None

    def circle_mask(self,center, sigma):
        radius = int(round(sigma//2 + 1))
        circle = np.zeros((self.total_size, self.total_size))
        circle = cv2.circle(circle, (center, center), radius, 1, -1)
        return circle
    
    def init_paras(self, model, init_hs = None):
        model.reset()
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
            init_hs_para = init_hs_para.reshape(self.N**2, )
            init_hs_para = torch.tensor(init_hs_para, dtype = torch.float)
            state_dict = model.state_dict()
            if self.far_field:
                state_dict['metalayer1.h_paras'] = init_hs_para
            else:
                state_dict['h_paras'] = init_hs_para
            model.load_state_dict(state_dict)
            #with torch.no_grad():
            #    model.metalayer1.h_paras.data = h_paras_initial
            print('initialized by loaded h_paras.')
            return None 
    
    def vis_field(self, E):
        if self.far_field:
            for i, lam in enumerate(self.GP.lams):
                Ii = E[i]
                fig = plt.figure()
                plt.imshow(Ii, cmap = 'magma')
                plt.colorbar()
                plt.title("wavelength: " + str(lam))
        else:
            for i, lam in enumerate(self.GP.lams):
                Ei = E[i]
                fig, axs = plt.subplots(1, 2, figsize = (12, 6))
                plot1 = axs[0].imshow(np.angle(Ei), cmap = 'magma')
                plt.colorbar(plot1, ax = axs[0])
                plot2 = axs[1].imshow(np.abs(Ei), cmap = 'magma')
                plt.colorbar(plot2, ax = axs[1])
                axs[0].set_title("wavelength: " + str(lam) + "Phase")
                axs[1].set_title("wavelength: " + str(lam) + "Amp")
        plt.show()
        return None
def plot_hs(out_hs, N):
    out_hs = out_hs.reshape(N, N)
    fig = plt.figure()
    plt.imshow(out_hs, cmap = 'magma')
    plt.colorbar()
    return fig

def plot_If(If):
    size = If.shape[0]
    c = size//2
    r = np.minimum(c, 60)
    If_c = If[c-r:c+r, c-r:c+r]
    fig, axs = plt.subplots(1, 2, figsize = (12, 6))
    plot1 = axs[0].imshow(If, cmap = 'magma')
    plt.colorbar(plot1, ax = axs[0])
    plot2 = axs[1].imshow(If_c, cmap = 'magma')
    plt.colorbar(plot2, ax = axs[1])
    return fig
