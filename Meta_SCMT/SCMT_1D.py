from turtle import up
import numpy as np
import matplotlib.pyplot as plt
from .SCMT_model_1D import Metalayer, SCMT_Model
import torch
from torch import optim
import os
from torch.utils.tensorboard import SummaryWriter
from .utils import gen_decay_rate, gaussian_func
from tqdm import tqdm
import warnings
import math


class SCMT_1D():
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

    def init_model(self, N, prop_dis, COUPLING=True, init_hs=None, far_field=False):
        '''
            the layers will be used when re building the fitted model. If you change any of this default values when you do the fitting.
            you should also change at here.
        '''
        self.N = N
        self.prop_dis = prop_dis
        self.total_size = (self.N) * self.GP.res
        self.far_field = far_field
        # if not Ni:
        #     Ni = 5 * N
        # if not k_row:
        #     k_row = N
        # self.APPROX = APPROX
        self.COUPLING = COUPLING
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if far_field:
            self.model = SCMT_Model(self.prop_dis, self.GP, COUPLING, N)
        else:
            self.model = Metalayer(self.GP, COUPLING, N)
        self.init_paras(self.model, self.GP.path, init_hs)
        self.model = self.model.to(self.device)
        return None

    def forward(self, theta=0):
        # incident field plane wave
        X = np.arange(self.total_size) * self.GP.dx
        E0 = np.exp(1j * self.GP.k * np.sin(theta) * X) / \
            np.sqrt(self.total_size)
        E0 = torch.tensor(E0, dtype=torch.complex64)
        E0 = E0.to(self.device)
        E_out = self.model(E0)
        E_out = E_out.cpu().detach().numpy()
        return E_out

    def optimize(self, notes, steps, lr=0.01, theta=0.0, minmax=True, substeps=5, target='lens', **kargs):
        if target == 'lens':
            if type(theta) is tuple:
                # eg: theta =  = (-np.pi/6, np.pi/6)
                self.optimize_range_theta(
                    notes, steps, lr, theta, minmax, substeps)
                print(
                    "the target is to make a perfect lens within the given incident angle range.")
            else:
                self.optimize_fix_theta(notes, steps, lr, theta)
                print("the target is to maximize the intensity of the center.")
        elif target == 'deflector':
            assert(kargs.get("deflecting_angle")
                   is not None), "need to specify deflecting angle."
            assert(kargs.get("delta_degree")
                   is not None), "need to specify the delta deflecting angle."
            self.optimize_fix_theta(
                notes, steps, lr, theta, kargs['deflecting_angle'], kargs['delta_degree'])
        return None

    def optimize_range_theta(self, notes, steps, lr, theta, minmax, substeps):
        print("optimize lens, incident planewave angle range: " + str(theta))
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

        radius = self.N * self.GP.period/2
        NA = radius / np.sqrt(radius**2 + self.prop_dis**2)
        target_sigma = self.GP.lam / (2 * NA) / self.GP.dx
        print("the numerical aperture: ", NA,
              "target spot size (number of points):", target_sigma)

        for step in tqdm(range(steps + 1)):
            #rand_theta = np.random.normal(loc = (theta[0] + theta[1])/2, scale = (theta[1] - theta[0])/2)
            if minmax:
                # def custom_optimizer(grad, paras, lr):
                #     return paras - lr * grad
                hs_grads = []
                sub_losses = []
                for _ in range(substeps):
                    rand_theta = np.random.uniform(theta[0], theta[1])
                    center = int(self.total_size//2 +
                                 self.prop_dis * np.tan(rand_theta)/self.GP.dx)
                    X = np.arange(self.total_size) * self.GP.dx
                    E0 = np.exp(1j * self.GP.k * np.sin(rand_theta)
                                * X)/np.sqrt(self.total_size)
                    E0 = torch.tensor(E0, dtype=torch.complex64)
                    E0 = E0.to(self.device)
                    # Compute prediction error
                    Ef = self.model(E0)
                    If = torch.abs(Ef)**2
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
                # with torch.no_grad():
                #grad = hs_grads[idx]
                #updated_paras = custom_optimizer(grad, self.model.metalayer1.h_paras, lr)
                # self.model.metalayer1.h_paras.copy_(updated_paras)

            else:
                rand_theta = np.random.uniform(theta[0], theta[1])
                center = int(self.total_size//2 + self.prop_dis *
                             np.tan(rand_theta)/self.GP.dx)
                X = np.arange(self.total_size) * self.GP.dx
                E0 = np.exp(1j * self.GP.k * np.sin(rand_theta) * X)
                E0 = torch.tensor(E0, dtype=torch.complex64)
                E0 = E0.to(self.device)
                # Compute prediction error
                Ef = self.model(E0)
                If = torch.abs(Ef)**2
                loss = loss_max_center(If, center, target_sigma)
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #grad = model.metalayer1.n_eff_paras.detach()
            # grad_peek(grad)
            if step % decay_steps == 0 and step != 0:
                my_lr_scheduler.step()

            if step % decay_steps == 0:    # every 1000 mini-batches...
                target_If = gaussian_func(
                    np.arange(self.total_size), center, target_sigma)
                # ...log the running loss
                writer.add_scalar('training loss',
                                  scalar_value=loss.item(), global_step=step)
                writer.add_figure('hs',
                                  plot_hs(
                                      self.model.metalayer1.hs.cpu().detach().numpy()),
                                  global_step=step)
                writer.add_figure('If',
                                  plot_If(If, target_If),
                                  global_step=step)
                # loss = loss.item()
                # loss_list.append(loss)
                # print(f"loss: {loss:>7f}  [{step:>5d}/{train_steps:>5d}]")
        if minmax:
            print("final lr:", lr)
        else:
            print("final lr:", my_lr_scheduler.get_last_lr())
        out_pos = (np.arange(self.N) - (self.N - 1)/2) * self.GP.period
        out_hs = self.model.metalayer1.hs.cpu().detach().numpy()
        # out_hs = out_hs//self.GP.dh * self.GP.dh
        out_data = np.c_[
            out_pos.reshape(-1, 1), np.round(out_hs.reshape(-1, 1), 3)]
        np.savetxt(out_path + 'waveguide_widths.csv', out_data, delimiter=",")
        print('parameters saved in.', out_path)
        return None

    def optimize_fix_theta(self, notes, steps, lr, theta, deflecting_angle=None, delta_degree = 1):
        '''
            deflecting_angle unit [rad]
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
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=decay_rate)
        self.model.train()

        X = np.arange(self.total_size) * self.GP.dx
        E0 = np.exp(1j * self.GP.k * np.sin(theta) * X)
        E0 = torch.tensor(E0, dtype=torch.complex64)
        E0 = E0.to(self.device)
        if deflecting_angle is not None:
            axis_angle, idx_min, idx_max = gen_x_angle(
                self.total_size, self.GP.dx, deflecting_angle, self.GP.lam, delta_degree=delta_degree)
        else:
            radius = self.N * self.GP.period/2
            NA = radius / np.sqrt(radius**2 + self.prop_dis**2)
            target_sigma = self.GP.lam / (2 * NA) / self.GP.dx
            print("the numerical aperture: ", NA,
                  "target spot size (number of points):", target_sigma)
            center = int(self.total_size//2)
        for step in tqdm(range(steps + 1)):
            # Compute prediction error
            Ef = self.model(E0)
            If = torch.abs(Ef)**2
            if deflecting_angle is not None:
                Ef_FFT = torch.fft.fft(Ef)
                FFT_I = torch.abs(Ef_FFT)
                loss = loss_max_range(FFT_I, idx_min, idx_max)
            else:
                loss = loss_max_center(If, center, target_sigma)
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
                writer.add_figure('If',
                                  plot_If(If),
                                  global_step=step)
                if deflecting_angle is not None:
                    writer.add_figure('abs(FFT of Ef)',
                                      plot_FFT_I(
                                          axis_angle, idx_min, idx_max, FFT_I),
                                      global_step=step)
                # loss = loss.item()
                # loss_list.append(loss)
                # print(f"loss: {loss:>7f}  [{step:>5d}/{train_steps:>5d}]")
        print("final lr:", my_lr_scheduler.get_last_lr())
        out_pos = (np.arange(self.N) - (self.N - 1)/2) * self.GP.period
        out_hs = self.model.metalayer1.hs.cpu().detach().numpy()
        # out_hs = out_hs//self.GP.dh * self.GP.dh
        out_data = np.c_[out_pos.reshape(-1, 1), out_hs.reshape(-1, 1)]
        np.savetxt(out_path + 'waveguide_widths.csv', out_data, delimiter=",")
        print('parameters saved in.', out_path)
        return None

    def init_paras(self, model, cache_path, init_hs=None):
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
            plt.plot(px, E, label='intensity')
            plt.legend()
            plt.xlabel("postion [um]")
        else:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            plot1 = axs[0].plot(px, np.angle(E), label="phase")
            axs[0].legend()
            axs[0].set_xlabel("postion [um]")
            plot2 = axs[1].plot(px, np.abs(E), label="amp")
            axs[1].legend()
            axs[1].set_xlabel("postion [um]")
        plt.show()
        return None


def plot_hs(out_hs):
    fig = plt.figure()
    plt.plot(out_hs)
    return fig


def plot_If(If, target_If=None):
    if type(If) == torch.Tensor:
        out_If = If.cpu().detach().numpy()
    else:
        out_If = If
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    plt.ioff()
    axs.plot(out_If, label='output')
    #print('sum of intensity:', out_If.sum())
    if not (target_If is None):
        target_If = target_If * out_If.max()
        axs.plot(target_If, label='target normalized by max(out_If)')
    axs.legend()
    #plt.ylabel('intensity normalized by max')
    return fig


def gen_x_angle(N, dx, degree, lam, delta_degree=1):
    degree = np.degrees(degree)
    delta_degree = np.degrees(delta_degree)
    axis_x = np.degrees(np.arcsin(np.arange(N) / (dx * N) / (1 / lam)))
    first_nan_idx = np.argmin(axis_x)
    print(f"First nan start ad index: {first_nan_idx}")
    axis_x_valid = axis_x[:first_nan_idx]
    idx_min = np.argmin(np.abs(axis_x_valid - (degree - delta_degree)))
    idx_max = np.argmin(np.abs(axis_x_valid - (degree + delta_degree)))
    print(f"the idx_min:{idx_min} and idx_max: {idx_max}.")
    return axis_x, idx_min, idx_max


def plot_FFT_I(axis_x, idx_min, idx_max, FFT_I):
    FFT_I = FFT_I.cpu().detach().numpy()
    fig = plt.figure()
    plt.plot(axis_x, FFT_I)
    plt.axvline(x=axis_x[idx_min], color='k', alpha=0.5,
                label='axvline - full height')
    plt.axvline(x=axis_x[idx_max], color='k', alpha=0.5,
                label='axvline - full height')
    plt.xlabel('degree [deg]')
    plt.ylabel('Fourier transform intensity')
    return fig


def loss_max_center(If, center, max_length):
    intensity = torch.sum(
        torch.abs(If[center - int(max_length//2): center + int(max_length//2)]))
    return - intensity


def loss_max_range(I, idx_min, idx_max):
    intensity = torch.sum(
        torch.abs(I[idx_min: idx_max + 1]))
    return - intensity
