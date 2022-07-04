from turtle import up
import numpy as np
import matplotlib.pyplot as plt
from .PBA_model_2D_lam import PBA_model
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
        self.init_paras(self.model, init_hs)
        self.model = self.model.to(self.device)
        return None
    
    def forward(self):
        #incident field plane wave
        E0 = np.ones((self.total_size, self.total_size))/self.total_size
        E0 = torch.tensor(E0, dtype = torch.complex64)
        E0 = E0.to(self.device)
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
        E0 = E0 = np.ones((self.total_size, self.total_size))/self.total_size
        E0 = torch.tensor(E0, dtype = torch.complex64)
        E0 = E0.to(self.device)
        radius = self.N * self.GP.period/2
        NA =  radius/ np.sqrt(radius**2 + self.prop_dis**2)
        #target_sigma = (min(self.GP.lams) + max(self.GP.lams)) * 0.5 / (2 * NA) / (self.GP.period / self.GP.out_res)
        target_sigma = min(self.GP.lams)  / (2 * NA) / (self.GP.period / self.GP.out_res)
        print(f"the numerical aperture: {NA:.2f}, target spot size (number of points): {target_sigma:.2f}")
        center = int(round(self.total_size//2))
        # circle = self.circle_mask(center, target_sigma)
        # circle = torch.tensor(circle, dtype = torch.float)
        # circle = circle.to(self.device)
        for step in tqdm(range(steps + 1)):
            # Compute prediction error
            Ifs = self.model(E0)
            if quarter:
                if minmax:
                    losses = [max_corner(If, self.GP.Knn, self.GP.out_res, target_sigma) for If in Ifs]
                    if not (loss_weights is None):
                        for idx, w in enumerate(loss_weights):
                            losses[idx] = losses[idx] * w
                    loss = - np.inf
                    for tmp_loss in losses:
                        if tmp_loss > loss:
                            loss = tmp_loss
                else:
                    idx = np.random.randint(0, len(self.GP.lams))
                    loss = max_corner(If, self.GP.Knn, self.GP.out_res, target_sigma)
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
                                plot_hs(self.model.hs.cpu().detach().numpy(), self.N),
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
        out_hs = self.model.hs.cpu().detach().numpy()
        out_hs = out_hs.reshape(self.N, self.N)
        if quarter:
            np.savetxt(out_path + 'waveguide_widths_quarter.csv', out_hs, delimiter=",")
            out_hs = quarter2whole(out_hs)
        np.savetxt(out_path + 'waveguide_widths.csv', out_hs, delimiter=",")
        print('parameters saved in.', out_path)
        return None
    
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
