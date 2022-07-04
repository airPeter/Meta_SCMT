'''
    design metasurface by periodic boundary approximation.
    for multiple wavelength. need to build lib for each wavelength using the PBA_design first.
'''
# standard python imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ..utils import lens_2D, lens_1D, Model, train, deflector_1D, deflector_2D
import torch
import warnings
from typing import List

class PBA():
    def __init__(self, GP, dim):
        self.GP = GP
        self.dim = dim  # dim = 1 or 2.
        self.width_phase_maps = []
        self.model = None

    # TODO: modify design lens for multiple lam.
    # def design_lens(self, N, focal_length, load = False, vis = True, quarter = False):
    #     if load:
    #         self.width_phase_map = np.load(self.GP.path + "rcwa_width_phase_map.npy")
    #     else:
    #         if self.width_phase_map is None:
    #             self.gen_lib()
    #     if self.dim == 1:
    #         x_lens, lens = lens_1D(N, self.GP.period, focal_length, self.GP.k)
    #     elif self.dim == 2:
    #         if quarter:
    #             N2 = 2 * N
    #             x_lens, lens = lens_2D(N2, self.GP.period, focal_length, self.GP.k)
    #             x_lens = x_lens[N:]
    #             lens = lens[N:,N:]
    #         else:
    #             x_lens, lens = lens_2D(N, self.GP.period, focal_length, self.GP.k)
    #     lens_phase = lens%(2 * np.pi) - np.pi
    #     widths_map = gen_width_from_phase(self.width_phase_map, lens_phase)
    #     widths_map = np.around(widths_map, 3)

    #     if vis:
    #         if self.dim == 2:
    #             fig, axs = plt.subplots(1, 2, figsize = (12, 6))
    #             plot1 = axs[0].imshow(lens, cmap = 'magma', extent = (x_lens.min(), x_lens.max(),x_lens.min(), x_lens.max()))
    #             plt.colorbar(plot1, ax = axs[0])
    #             plot2 = axs[1].imshow(widths_map, cmap = 'magma', extent = (x_lens.min(), x_lens.max(),x_lens.min(), x_lens.max()))
    #             plt.colorbar(plot2, ax = axs[1])
    #             axs[0].set_title("Lens phase")
    #             axs[0].set_xlabel("Position [um]")
    #             axs[0].set_ylabel("Position [um]")
    #             axs[1].set_title("Lens widths")
    #             axs[1].set_xlabel("Position [um]")
    #             axs[1].set_ylabel("Position [um]")
    #             plt.show()
    #         elif self.dim == 1:
    #             fig, axs = plt.subplots(2, 1, figsize = (12, 12))
    #             axs[0].plot(x_lens, lens)
    #             axs[1].plot(x_lens, widths_map)
    #             axs[0].set_title("Lens phase")
    #             axs[0].set_xlabel("Position [um]")
    #             axs[1].set_title("Lens widths")
    #             axs[1].set_xlabel("Position [um]")
    #             plt.show()
    #     return widths_map

    def design_deflector(self, N, degree, vis=True):
        designed_phases = []
        designed_lenses = []
        for idx, lam in enumerate(self.GP.lams):
            self.width_phase_maps.append(
                np.load(self.GP.paths[idx] + "rcwa_width_phase_map.npy"))
            if self.dim == 1:
                x_lens, lens = deflector_1D(
                    N, self.GP.period, degree, 2 * np.pi / lam)
            elif self.dim == 2:
                x_lens, lens = deflector_2D(
                    N, self.GP.period, degree, 2 * np.pi / lam)
            designed_lenses.append(lens)
            lens_phase = lens % (2 * np.pi) - np.pi
            designed_phases.append(lens_phase)

        widths_map = gen_width_from_phase(
            self.width_phase_maps, designed_phases)
        widths_map = np.around(widths_map, 3)

        if vis:
            num_plots = len(self.GP.lams)
            if self.dim == 2:
                fig, axs = plt.subplots(
                    1, num_plots, figsize=(6 * num_plots, 6))
                for i in range(num_plots):
                    plot = axs[i].imshow(designed_lenses[i], cmap='magma', extent=(
                        x_lens.min(), x_lens.max(), x_lens.min(), x_lens.max()))
                    plt.colorbar(plot, ax=axs[i])
                    axs[i].set_title(f"Lens phase, lam: {self.GP.lams[i]}")
                    axs[i].set_xlabel("Position [um]")
                    axs[i].set_ylabel("Position [um]")
                plt.show()
                plt.figure()
                plt.imshow(widths_map, cmap='magma', extent=(
                    x_lens.min(), x_lens.max(), x_lens.min(), x_lens.max()))
                plt.colorbar()
                plt.title("Lens widths")
                plt.xlabel("Position [um]")
                plt.ylabel("Position [um]")
                plt.show()
            elif self.dim == 1:
                fig, axs = plt.subplots(
                    num_plots, 1, figsize=(8, 6 * num_plots))
                for i in range(num_plots):
                    axs[i].plot(x_lens, designed_lenses[i])
                    axs[i].set_title(f"Lens phase, lam: {self.GP.lams[i]}")
                    axs[i].set_xlabel("Position [um]")
                plt.show()
                
                plt.figure()
                plt.plot(x_lens, widths_map)
                plt.xlabel("Position [um]")
                plt.show()
        return designed_phases, widths_map


def gen_width_from_phase(width_phase_maps: List, target_phases: List):
    widths = width_phase_maps[0][0]
    phases = [width_phase_map[1] for width_phase_map in width_phase_maps]
    assert (len(phases) == len(target_phases)), "set of phases in width phase map should be equal to set of target phases"
    num_phases = len(phases)
    phases = np.array(phases)
    target_phases = np.array(target_phases)
    target_shape = target_phases.shape[1:]
    target_phases = target_phases.T
    phases = phases.T
    phases = phases.reshape(1, -1, num_phases)
    target_phases = target_phases.reshape(-1, 1, num_phases)
    diff = np.sum((target_phases - phases)**2, axis = -1)
    indexes = np.argmin(diff, axis=-1)
    widths_map = np.take(widths, indexes)
    widths_map = widths_map.reshape(target_shape)
    return widths_map

