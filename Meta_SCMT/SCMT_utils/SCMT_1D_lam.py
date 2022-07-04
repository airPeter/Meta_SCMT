'''
    multi wavelength SCMT model, only support normal incidence.
'''
import numpy as np
import matplotlib.pyplot as plt
from .SCMT_model_1D_lam import Metalayer, SCMT_Model
from .SCMT_1D import plot_hs, plot_FFT_I, plot_If, gen_x_angle, loss_max_center, loss_max_range
import torch
from torch import optim
import os
from torch.utils.tensorboard import SummaryWriter
from ..utils import gen_decay_rate
from tqdm import tqdm
import warnings
from typing import List


class SCMT_1D():
    def __init__(self, GP):
        self.GP = GP
        self.model = None
        self.COUPLING = None
        self.Euler_steps = None
        self.N = None
        self.prop_dis = None

    def init_model(self, N, prop_dis, COUPLING=True, init_hs=None, far_field=False):
        '''
            the layers will be used when re building the fitted model. If you change any of this default values when you do the fitting.
            you should also change at here.
        '''
        self.N = N
        self.prop_dis = prop_dis
        self.total_size = (self.N) * self.GP.res
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
        # incident field plane wave
        # X = np.arange(self.total_size) * self.GP.dx
        # E0 = np.exp(1j * self.GP.k * np.sin(theta) * X)
        E0 = np.ones((self.total_size,))/np.sqrt(self.total_size)
        E0 = torch.tensor(E0, dtype=torch.complex64)
        E0 = E0.to(self.device)
        E_out = self.model(E0)
        E_out = [E.cpu().detach().numpy() for E in E_out]
        return E_out

    def optimize(self, notes, steps, lr=0.01, minmax=True, loss_weights: List = None, target='lens', **kargs):
        '''
        loss_weights: we use loss_weights to compensate the intensity difference between different lam, when optimizing by minmax method.
        '''
        if target == 'lens':
            deflecting_angle = None
            pass

        elif target == 'deflector':
            assert(kargs.get("deflecting_angle")
                   is not None), "need to specify deflecting angle."
            assert(kargs.get("delta_degree")
                   is not None), "need to specify the delta deflecting angle."
            deflecting_angle = kargs['deflecting_angle']
            delta_degree = kargs['delta_degree']

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
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=decay_rate)
        self.model.train()

        # X = np.arange(self.total_size) * self.GP.dx
        # E0 = np.exp(1j * self.GP.k * np.sin(theta) * X)
        E0 = np.ones((self.total_size,))/np.sqrt(self.total_size)
        E0 = torch.tensor(E0, dtype=torch.complex64)
        E0 = E0.to(self.device)
        if deflecting_angle is not None:
            axis_angles = [gen_x_angle(
                self.total_size, self.GP.dx, deflecting_angle, lam, delta_degree=delta_degree) for lam in self.GP.lams]
        else:
            radius = self.N * self.GP.period/2
            NA = radius / np.sqrt(radius**2 + self.prop_dis**2)
            target_sigma = self.GP.lam / (2 * NA) / self.GP.dx
            print("the numerical aperture: ", NA,
                  "target spot size (number of points):", target_sigma)
            center = int(self.total_size//2)

        for step in tqdm(range(steps + 1)):
            # Compute prediction error
            Efs = self.model(E0)
            Ifs = [torch.abs(E)**2 for E in Efs]
            Ef_FFTs = [torch.fft.fft(Ef) for Ef in Efs]
            FFT_Is = [torch.abs(Ef_FFT) for Ef_FFT in Ef_FFTs]
            if minmax:
                if deflecting_angle is not None:
                    losses = [loss_max_range(FFT_I, axis_angles[idx][1], axis_angles[idx][2])
                              for idx, FFT_I in enumerate(FFT_Is)]
                else:
                    losses = [loss_max_center(If, center, target_sigma)
                              for If in Ifs]
                if not (loss_weights is None):
                    for idx, w in enumerate(loss_weights):
                        losses[idx] = losses[idx] * w
                loss = - np.inf
                for tmp_loss in losses:
                    if tmp_loss > loss:
                        loss = tmp_loss
            else:
                idx = np.random.randint(0, len(self.GP.lams))
                if deflecting_angle is not None:
                    loss = loss_max_range(
                        FFT_Is[idx], axis_angles[idx][1], axis_angles[idx][2])
                else:
                    loss = loss_max_center(Ifs[idx], center, target_sigma)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #grad = model.metalayer1.n_eff_paras.detach()
            # grad_peek(grad)
            if step % decay_steps == 0 and step != 0:
                my_lr_scheduler.step()

            if step % decay_steps == 0:    # every 1000 mini-batches...

                # ...log the running loss
                writer.add_scalar('training loss',
                                  scalar_value=loss.item(), global_step=step)
                writer.add_figure('hs',
                                  plot_hs(
                                      self.model.metalayer1.hs.cpu().detach().numpy()),
                                  global_step=step)
                for idx, If in enumerate(Ifs):
                    out_If = If.cpu().detach().numpy()
                    writer.add_figure(f"If, lam: {self.GP.lams[idx]} um",
                                      plot_If(out_If),
                                      global_step=step)
                    if deflecting_angle is not None:
                        writer.add_figure(f"abs(FFT of Ef), lam: {self.GP.lams[idx]} um",
                                          plot_FFT_I(
                                              axis_angles[idx][0], axis_angles[idx][1], axis_angles[idx][2], FFT_Is[idx]),
                                          global_step=step)
                # loss = loss.item()
                # loss_list.append(loss)
                # print(f"loss: {loss:>7f}  [{step:>5d}/{train_steps:>5d}]")
        print("final lr:", my_lr_scheduler.get_last_lr())
        out_pos = (np.arange(self.N) - (self.N - 1)/2) * self.GP.period
        out_hs = self.model.metalayer1.hs.cpu().detach().numpy()
        #out_hs = out_hs//self.GP.dh * self.GP.dh
        out_data = np.c_[
            out_pos.reshape(-1, 1), np.round(out_hs.reshape(-1, 1), 3)]
        np.savetxt(out_path + 'waveguide_widths.csv', out_data, delimiter=",")
        print('parameters saved in.', out_path)
        return None

    def init_paras(self, model, init_hs=None):
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
            hs_paras = (init_hs - self.GP.h_min) / \
                (self.GP.h_max - self.GP.h_min)
            hs_paras = np.minimum(np.maximum(hs_paras, 0.01), 0.99)
            # becuase in our model we use Sigmoid  function, here, the hs paras is generated by inverse function.
            init_hs_para = np.log(hs_paras / (1 - hs_paras))
            init_hs_para = torch.tensor(init_hs_para, dtype=torch.float)
            state_dict = model.state_dict()
            if self.far_field:
                state_dict['metalayer1.h_paras'] = init_hs_para
            else:
                state_dict['h_paras'] = init_hs_para
            model.load_state_dict(state_dict)
            # with torch.no_grad():
            #    model.metalayer1.h_paras.data = h_paras_initial
            print('initialized by loaded h_paras.')
            return None

    def vis_field(self, E):
        px = (np.arange(self.total_size) - self.total_size//2) * self.GP.dx
        if self.far_field:
            for i, lam in enumerate(self.GP.lams):
                Ii = E[i]
                plt.figure()
                plt.plot(px, Ii, label='intensity')
                plt.legend()
                plt.xlabel("postion [um]")
                plt.title("wavelength: " + str(lam))
                plt.show()
        else:
            for i, lam in enumerate(self.GP.lams):
                Ei = E[i]
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                _ = axs[0].plot(px, np.angle(Ei), label="phase")
                axs[0].legend()
                axs[0].set_xlabel("postion [um]")
                _ = axs[1].plot(px, np.abs(Ei), label="amp")
                axs[1].legend()
                axs[1].set_xlabel("postion [um]")
                plt.title("wavelength: " + str(lam))
                plt.show()
        return None
