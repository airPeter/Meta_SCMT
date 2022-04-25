'''
    multi wavelength SCMT model, only support normal incidence.
'''
import numpy as np
import matplotlib.pyplot as plt
from .SCMT_model_1D_lam import Metalayer, SCMT_Model
import torch
from torch import optim
import os
from torch.utils.tensorboard import SummaryWriter
from .utils import gen_decay_rate
from tqdm import tqdm
import warnings
class SCMT_1D():
    def __init__(self, GP):
        self.GP = GP
        self.model = None
        self.COUPLING = None
        self.Euler_steps = None
        self.N = None
        self.prop_dis = None
        
    def init_model(self, N, prop_dis, COUPLING = True, init_hs = None, far_field = False):
        '''
            the layers will be used when re building the fitted model. If you change any of this default values when you do the fitting.
            you should also change at here.
        '''
        self.N = N
        self.prop_dis = prop_dis
        self.total_size = (self.N + 2 * self.GP.Knn + 1) * self.GP.res
        self.far_field = far_field
        self.COUPLING = COUPLING
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if far_field:
            self.model = SCMT_Model(self.prop_dis, self.GP, COUPLING, N)
        else:
            self.model = Metalayer(self.GP, COUPLING, N)
        self.init_paras(self.model, init_hs)
        self.model = self.model.to(self.device)
        return None
    
    def forward(self):
        '''
        output:
            if far_field == True, output is intensity otherwise is field.
        '''
        #incident field plane wave
        # X = np.arange(self.total_size) * self.GP.dx
        # E0 = np.exp(1j * self.GP.k * np.sin(theta) * X)
        E0 = np.ones((self.total_size,))
        E0 = torch.tensor(E0, dtype = torch.complex64)
        E0 = E0.to(self.device)
        E_out = self.model(E0)
        E_out = [E.cpu().detach().numpy() for E in E_out]
        return E_out
    
    def optimize(self, notes, steps, lr = 0.01, minmax = False):
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

        # X = np.arange(self.total_size) * self.GP.dx
        # E0 = np.exp(1j * self.GP.k * np.sin(theta) * X)
        E0 = np.ones((self.total_size,))
        E0 = torch.tensor(E0, dtype = torch.complex64)
        E0 = E0.to(self.device)
        radius = self.N * self.GP.period/2
        NA =  radius/ np.sqrt(radius**2 + self.prop_dis**2)
        target_sigma = max(self.GP.lams) / (2 * NA) / self.GP.dx
        print("the numerical aperture: ", NA, "target spot size for max[lams], (number of points):", target_sigma)
        center = int(self.total_size//2)
        for step in tqdm(range(steps + 1)):
            # Compute prediction error
            Ifs = self.model(E0)
            if minmax:
                losses = [loss_max_center(If, center, target_sigma) for If in Ifs]
                loss = - np.inf
                for tmp_loss in losses:
                    if tmp_loss > loss:
                        loss = tmp_loss
            else:
                idx = np.random.randint(0, len(self.GP.lams))
                loss = loss_max_center(Ifs[idx], center, target_sigma)
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
                                plot_hs(self.model.metalayer1.hs.cpu().detach().numpy()),
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
        out_pos = (np.arange(self.N) - (self.N - 1)/2) * self.GP.period
        out_hs = self.model.metalayer1.hs.cpu().detach().numpy()
        out_data = np.c_[out_pos.reshape(-1,1), out_hs.reshape(-1, 1)]
        np.savetxt(out_path + 'waveguide_widths.csv', out_data, delimiter=",")
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
        px = (np.arange(self.total_size) - self.total_size//2) * self.GP.dx
        if self.far_field:
            for i, lam in enumerate(self.GP.lams):
                Ii = E[i]
                plt.figure()
                plt.plot(px, Ii, label = 'intensity')
                plt.legend()
                plt.xlabel("postion [um]")
                plt.title("wavelength: " + str(lam))
                plt.show()
        else:
            for i, lam in enumerate(self.GP.lams):
                Ei = E[i]
                fig, axs = plt.subplots(1, 2, figsize = (12, 6))
                _ = axs[0].plot(px, np.angle(Ei), label = "phase")
                axs[0].legend()
                axs[0].set_xlabel("postion [um]")
                _ = axs[1].plot(px, np.abs(Ei), label = "amp")
                axs[1].legend()
                axs[1].set_xlabel("postion [um]")
                plt.title("wavelength: " + str(lam))
                plt.show()
        return None

def plot_hs(out_hs):
    fig = plt.figure()
    plt.plot(out_hs)
    return fig

def plot_If(out_If, target_If = None):
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