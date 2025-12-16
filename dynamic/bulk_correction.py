#!/usr/bin/env python3
from dynamic.spots import SpotsList
import numpy as np
from dynamic.calc import fit_exp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class BulkFitter:

    def __init__(self, spots: SpotsList, resolution_bin_size: int = 100):

        self.spots = spots
        self.positive_spots, self.negative_spots = self.split_by_sign()
        self.resolution_bin_size = resolution_bin_size
        self.fit_params = None
        self.fit_function = None
        self.x_fit = None
        self.y_fit = None
        self.x_to_fit = None
        self.y_to_fit = None

    def split_by_sign(self):

        positive_spots = []
        negative_spots = []
        for spot in self.spots.spots:
            if spot.intensity > 0:
                positive_spots.append(spot)
            else:
                negative_spots.append(spot)

        positive_spots = sorted(positive_spots, key=lambda obj: obj.resolution)
        negative_spots = sorted(negative_spots, key=lambda obj: obj.resolution)
        return positive_spots, negative_spots

    def fit(self):

        resolution_limits = []

        y = []
        x = []

        for spot in self.positive_spots:
            x.append(spot.resolution)
            diff = spot.Fo_scaled**2 - spot.Fc**2
            y.append(diff)
        y = np.array(y)
        x = np.array(x)

        n = len(x)
        bin_size = self.resolution_bin_size

        x_to_fit = []
        y_to_fit = []

        for idx in range(0, n - bin_size, bin_size):
            resolution_limits.append([x[idx], x[idx + bin_size]])

            r_mid = 0.5 * (x[idx] + x[idx + bin_size])
            x_to_fit.append(r_mid)

            y_mid = []
            for j in range(idx, idx + bin_size):
                y_mid.append(y[j])
            y_mid = np.array(y_mid)
            y_to_fit.append(y_mid.mean())

        x_to_fit = np.array(x_to_fit)
        y_to_fit = np.array(y_to_fit)
        resolution_limits = np.array(resolution_limits)

        fit_params, fit_function = fit_exp(x_to_fit, y_to_fit)
        self.fit_params = fit_params
        self.fit_function = fit_function
        self.resolution_limits = resolution_limits

        x_fit = np.linspace(x[0], x[-1], 2000)
        y_fit = fit_function(x_fit)

        self.x_to_fit = x_to_fit
        self.y_to_fit = y_to_fit
        self.x_fit = x_fit
        self.y_fit = y_fit

        out_file = f"{self.spots.output_prefix}_fit_params.npz"
        np.savez(out_file, params=fit_params, limits=resolution_limits)

    def plot_fitted(self):
        fig = plt.figure(figsize=(3.375, 3.0))
        ax = fig.add_axes([0.17, 0.15, 0.8, 0.8])

        for lind, lim in enumerate(self.resolution_limits):
            if lind % 2 == 0:
                x0, x1 = lim
                ax.axvspan(x0, x1, color='#BEBEBE', lw=0, alpha=0.5)

        ax.plot(self.x_fit, self.y_fit, c='C3', lw=1)
        ax.plot(self.x_to_fit, self.y_to_fit, marker='o',
                mew=0, ms=3, c='C0', lw=0)

        l_min = self.resolution_limits[0][0]
        l_max = self.resolution_limits[-1][1]
        ax.set_xlim(l_min, l_max)

        y_min = self.y_fit.min()

        if y_min < -20:
            ax.set_ylim(-20.0, 1.5)
        else:
            ax.set_ylim(y_min - 0.5, 1.5)

        ax.set_xlabel(r'Resolution ($\rm \AA$)')
        ax.set_ylabel(r'$\langle |\rm F_{\rm e}|^2 \rangle$')

        fig_name = f"{self.spots.output_prefix}_fit.png"
        plt.savefig(fig_name, dpi=400)

    def replace_with_fc(self):

        for spot in self.spots.spots:
            spot.Fo_corrected = spot.Fc

    def correct_fo(self):

        for spot in self.spots.spots:

            fe_fit = double_exp(spot.resolution, *self.fit_params)
            scale_correct = (spot.Fc**2 + fe_fit) / spot.Fc**2

            if spot.intensity > 0:
                if scale_correct > 0:
                    spot.scale_correct = np.sqrt(scale_correct)
                    spot.Fo_corrected = spot.Fo_scaled / spot.scale_correct
                else:
                    spot.scale_correct = -1.0
                    spot.Fo_corrected = spot.Fo_scaled
            else:
                if scale_correct < 0:
                    spot.scale_correct = np.sqrt(-scale_correct)
                    Fo_scaled = np.sqrt(-spot.intensity) / self.spots.scale
                    spot.Fo_corrected = Fo_scaled / spot.scale_correct
                else:
                    spot.scale_correct = -1.0
                    spot.Fo_corrected = None

    def plot_corrected(self):

        plt.figure(figsize=(3.375*1.5, 3.0))
        gs = gridspec.GridSpec(1, 2, top=0.95, bottom=0.15,
                               left=0.07, right=0.95,
                               wspace=0.2, hspace=0.2)

        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])

        fc_positive = []
        fo_scaled_positive = []
        fo_corrected_positive = []

        fc_negative = []
        fo_scaled_negative = []
        fo_corrected_negative = []

        self.positive_spots, self.negative_spots = self.split_by_sign()

        for spot in self.positive_spots:
            fc_positive.append(spot.Fc)
            fo_corrected_positive.append(spot.Fo_corrected)
            fo_scaled_positive.append(spot.Fo_scaled)

        for spot in self.negative_spots:
            fc_negative.append(spot.Fc)
            fo_corrected_negative.append(spot.Fo_corrected)
            fo_scaled_negative.append(spot.Fo_scaled)

        fc_positive = np.array(fc_positive)
        fo_corrected_positive = np.array(fo_corrected_positive)
        fo_scaled_positive = np.array(fo_scaled_positive)

        fc_negative = np.array(fc_negative)
        fo_corrected_negative = np.array(fo_corrected_negative)
        fo_scaled_negative = np.array(fo_scaled_negative)

        ax2.plot(fc_positive, fo_corrected_positive, marker='o',
                 mew=0, lw=0, c='C0', ms=1)
        ax2.plot(fc_negative, fo_corrected_negative, marker='o',
                 mew=0, lw=0, c='C3', ms=1)

        ax1.plot(fc_positive, fo_scaled_positive, marker='o',
                 mew=0, lw=0, c='C0', ms=1)
        ax1.plot(fc_negative, fo_scaled_negative, marker='o',
                 mew=0, lw=0, c='C3', ms=1)

        fig_name = f"{self.spots.output_prefix}_corrected.png"

        ax1.set_xlim(0, 10)
        ax2.set_xlim(0, 10)

        ax1.set_ylim(0, 10)
        ax2.set_ylim(0, 10)
        plt.savefig(fig_name, dpi=400)


def double_exp(x, a, b, c, d):
    return a*np.exp(b*x) + c*np.exp(d*x)
