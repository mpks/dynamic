# excitation_error.py
from dials.command_line.frame_orientations import extract_experiment_data
from dxtbx.model.experiment_list import ExperimentListFactory
from scitbx import matrix
from dials.array_family import flex
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dynamic.calc import find_best_scale
import numpy as np


class ExcitationErrorCalculator:
    """
    Computes excitation error for DIALS reflections or custom Spot objects.
    Uses the same definition as DIALS/PETS:
        r = UB * hkl
        s1 = r + s0
        excitation_error = |s0| - |s1|
    """

    def __init__(self, experiments_json: str, exp_id: int = 0,
                 refl_file=None):

        self.reflections = flex.reflection_table.from_file(refl_file)
        experiments = ExperimentListFactory.from_json_file(experiments_json)
        self.experiment = experiments[exp_id]

        self.frame_orientations = extract_experiment_data(self.experiment,
                                                          scale=1)

        Ut = matrix.sqr(self.experiment.crystal.get_U()).transpose()
        self.frame_orientations['rotated_missets'] = [
            M * Ut for M in self.frame_orientations["orientations"]
        ]

        self.crystal = self.experiment.crystal
        self.beam = self.experiment.beam

        self.s0 = matrix.col(self.beam.get_s0())
        self.inv_wl = self.s0.length()            # Inverse wavelength

    def ee(self):

        rotated_missets = self.frame_orientations["rotated_missets"]
        scan = self.experiment.scan
        s0 = matrix.col(self.experiment.beam.get_s0())
        inv_wl = s0.length()
        A = matrix.sqr(self.experiment.crystal.get_A())

        # _, _, frames = self.reflections["xyzcal.px"].parts()

        self.virtual_frames = []

        arr_start, arr_end = scan.get_array_range()

        # Loop over the virtual frames
        starts = list(range(arr_start, arr_end, 1))
        # print("VIRTUAL FRAMES", arr_start, arr_end)

        # How many real frames to merge into virtual
        # In the original DIALS code it can be both odd and even number.
        # To avoid extra complications, here we will assume it is always one.
        self.n_merged = 1

        self.ee_for_miller_z = {}

        for start in starts:
            stop = start + self.n_merged
            if stop > arr_end:
                break

            # Look up the orientation data using an index, which is the centre
            # of the virtual frame, offset so that the scan starts from 0
            centre = (start + stop) / 2.0
            index = int(centre) - arr_start

            # Works only when merging an odd number of real frames
            M = rotated_missets[index]

            # Calculate the excitation error at this orientation
            UB = flex.mat3_double(len(self.reflections), M * A)
            r = UB * self.reflections["miller_index"].as_vec3_double()
            s1 = r + s0
            excitation_err = inv_wl - s1.norms()

            millers = self.reflections["miller_index"]
            zs = self.reflections["xyzobs.px.value"].parts()[2]

            for hkl, zval, ee, in zip(millers, zs, excitation_err):
                H, K, L = hkl
                z_int = int(zval)

                if z_int == start:
                    self.ee_for_miller_z[(H, K, L, z_int)] = ee

    def attach_to_spots_list(self, spots_list):
        """
        Compute excitation errors for all Spot objects in a SpotsList.
        Adds spot.excitation_error attribute.
        """

        for spot in spots_list.spots:

            # If no image index -> undefined
            if spot.z is None:
                spot.excitation_error = None
                continue

            H, K, L = spot.H, spot.K, spot.L
            z = spot.z

            key = (H, K, L, z)

            if key in self.ee_for_miller_z:
                ee = self.ee_for_miller_z[key]
                spot.excitation_error = ee
            else:
                print("Couldn't find the miller + z spot", H, K, L, z)
                spot.excitation_error = None

    def plot_ee_per_image(self, spots, out_path, set_idx):

        all_z = np.array([int(spot.z) for spot in spots.spots])
        all_ee = []
        all_ratios = []
        snr_all = []

        for ind in range(all_z.min(), all_z.max()):
            # for ind in range(2, 3):
            print(ind)

            xs = []
            ys = []
            ees = []
            ratios = []
            sigmas = []
            ints = []
            f_cal = []
            labels = []

            for spot in spots.spots:
                if spot.z == ind and spot.intensity > 0:
                    xs.append(spot.x)
                    ys.append(spot.y)
                    ees.append(spot.excitation_error)
                    ints.append(spot.intensity)
                    sigmas.append(spot.sigma)

                    labels.append((spot.H, spot.K, spot.L))

                    all_ee.append(spot.excitation_error)

                    # int_fc = abs(spot.Fc * spots.global_scale)**2
                    f_cal.append(spot.Fc)

                    # int_obs = spot.intensity
                    # ratio = int_obs / int_fc

                    # if abs(ratio) > 100:
                    #     ratio = 100

                    # log_ratio = 10*np.log(ratio)

                    # ratios.append(log_ratio)
                    # all_ratios.append(ratio)
                    snr_all.append(spot.intensity / spot.sigma)
            print("IND", ind, labels)
            ints = np.array(ints)
            f_cal = np.array(f_cal)
            fo_temp = np.sqrt(ints)

            scale, r1_fac = find_best_scale(fo_temp, f_cal)

            i_cal = (f_cal * scale)**2
            ratios = ints / i_cal

            ratios[ratios > 1000] = 1000
            all_ratios.extend(ratios)

            ratios = 10*np.log(ratios)

            xs = np.array(xs)
            ys = np.array(ys)
            # ratios = np.array(ratios)
            ees = np.array(ees)

            fig = plt.figure(figsize=(6*1.2, 6.0*1.2))
            gs = gridspec.GridSpec(2, 2, top=0.95, bottom=0.07,
                                   left=0.15, right=0.94,
                                   wspace=0.0, hspace=0.0)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[0, 1])
            ax3 = plt.subplot(gs[1, 0])
            ax4 = plt.subplot(gs[1, 1])

            ax2.tick_params(labelleft=False)
            ax4.tick_params(labelleft=False)
            ax1.tick_params(labelbottom=False)
            ax2.tick_params(labelbottom=False)

            # base_size = 1.0

            # Plot ratios
            xr_pos = xs[ratios >= 0]
            yr_pos = ys[ratios >= 0]
            rr_pos = abs(ratios[ratios >= 0])

            xr_neg = xs[ratios < 0]
            yr_neg = ys[ratios < 0]
            rr_neg = abs(ratios[ratios < 0])

            ax1.scatter(xr_pos, yr_pos, s=rr_pos,
                        edgecolors='none', c='C0')
            ax1.scatter(xr_neg, yr_neg, s=rr_neg,
                        edgecolors='none', c='C1')

            ax1.scatter([0], [650], [10], c='C0', edgecolors='none')
            ax1.scatter([0], [600], [10], c='C1', edgecolors='none')
            ax1.text(30, 650, r'$f > 0$ (boosted)', va='center', ha='left')
            ax1.text(30, 600, r'$f < 0$ (suppressed)', va='center', ha='left')

            ax1.scatter([500], [650], [1], c='#BEBEBE', edgecolors='none')
            ax1.scatter([500], [600], [10], c='#BEBEBE', edgecolors='none')
            ax1.scatter([500], [550], [50], c='#BEBEBE', edgecolors='none')
            ax1.text(530, 650, r'$|f| = 1$', va='center', ha='left')
            ax1.text(530, 600, r'$|f| = 10$', va='center', ha='left')
            ax1.text(530, 550, r'$|f| = 50$', va='center', ha='left')

            # Plot excitation error
            ees_scale = np.abs(np.array(ees)).mean()
            ees /= ees_scale
            ees *= 20.

            xe_pos = xs[ees >= 0]
            ye_pos = ys[ees >= 0]
            ee_pos = abs(ees[ees >= 0])

            xe_neg = xs[ees < 0]
            ye_neg = ys[ees < 0]
            ee_neg = abs(ees[ees < 0])

            ax2.scatter(xe_pos, ye_pos, s=100./ee_pos,
                        edgecolors='none', c='C2')
            ax2.scatter(xe_neg, ye_neg, s=100./ee_neg,
                        edgecolors='none', c='C3')

            ints = np.array(ints)
            sigmas = np.array(sigmas)

            q1, q2 = np.quantile(ints, [0.5, 1.0])

            log_ints = np.log(ints + 2.0)
            log_ints[log_ints > 15] = 15

            xs_weak = xs[ints < q1]
            ys_weak = ys[ints < q1]
            ints_weak = log_ints[ints < q1]

#            for x, y, label in zip(xs, ys, labels):
#                ax3.text(x, y, label, va='center', ha='left',
#                         fontsize=5)

#            xs_strong = xs[ints >= q1]
#            ys_strong = ys[ints >= q1]
#            ints_strong = ints[ints < q1]

            ax3.scatter(xs, ys, s=2 + 5*log_ints, edgecolors='none', c='C0')
            ax3.scatter(xs_weak, ys_weak, s=2 + 5*ints_weak,
                        edgecolors='none', c='C1')

            r1_label = r'$\rm R_1 = $'
            ax1.text(0.01, 0.97, f"{r1_label} = {r1_fac:.4f}",
                     transform=fig.transFigure)

            sig_ratio = ints / sigmas
            ax4.scatter(xs, ys, s=200*abs(sig_ratio), edgecolors='none',
                        c='C0')

            lab_str = r'$f = {\rm log}(\rm I_{\rm o} / I_{\rm c})$'
            ax1.text(0.5, 1.01, lab_str, va='bottom', ha='center',
                     transform=ax1.transAxes)
            ax3.text(0.8, 0.91, r'${\rm log}{(\rm I_{\rm o})}$',
                     va='bottom', ha='center', transform=ax3.transAxes)
            ax4.text(0.5, 0.91, r'${\rm I}/{\sigma(\rm I)}$',
                     va='bottom', ha='center', transform=ax4.transAxes)
            ax2.text(0.5, 1.01, r'Inverse Exc. error $1 / |s_{\rm g}|$',
                     va='bottom', ha='center', transform=ax2.transAxes)

            ax3.scatter([0], [650], [20], c='C0', edgecolors='none')
            ax3.scatter([0], [600], [20], c='C1', edgecolors='none')
            ax3.text(30, 650, r'$\rm I > I_{50 \%}$ (strong)',
                     va='center', ha='left')
            ax3.text(30, 600, r'$\rm I < I_{50 \%}$ (weak)',
                     va='center', ha='left')

            ax2.scatter([0], [650], [20], c='C3', edgecolors='none')
            ax2.scatter([0], [600], [20], c='C2', edgecolors='none')
            ax2.text(30, 650, r'$s_{\rm g} < 0$', va='center', ha='left')
            ax2.text(30, 600, r'$s_{\rm g} > 0$', va='center', ha='left')

            ax1.set_ylabel('Pixel Index Y', labelpad=10)
            ax3.set_ylabel('Pixel Index Y', labelpad=10)
            ax3.set_xlabel('Pixel Index X', labelpad=10)
            ax4.set_xlabel('Pixel Index X', labelpad=10)

            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_xlim(-50, 700)
                ax.set_ylim(-50, 700)

            fig_name = out_path + f'ee_{set_idx}_img_{ind:04d}'
            plt.savefig(fig_name, dpi=400)
            plt.close(fig)

        fig = plt.figure(figsize=(3.375, 3.0))
        ax = fig.add_axes([0.13, 0.14, 0.84, 0.82])

        ax.set_ylim(-10, 30)
        all_ee = np.array(all_ee)
        snr_all = np.array(snr_all)
        all_ratios = np.array(all_ratios)

        all_ee = all_ee[all_ratios < 1000]
        snr_all = snr_all[all_ratios < 1000]
        all_ratios = all_ratios[all_ratios < 1000]

        e_idx = np.argsort(all_ee)
        e_sorted = all_ee[e_idx]
        rr_sorted = all_ratios[e_idx]

        nbins = 10
        bin_edges = np.quantile(e_sorted, np.linspace(0, 1, nbins+1))

        bin_index = np.digitize(e_sorted, bin_edges) - 1

        mean_r = np.zeros(nbins)
        for b in range(nbins):
            sel = bin_index == b
            mean_r[b] = np.mean(rr_sorted[sel])
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        ax.plot(all_ee, all_ratios,
                marker='o', mew=0, ms=1, c='C0', lw=0)
        ax.set_xlim(-0.02, 0.02)
        ax.set_ylim(-1, 50)
        ax.plot(bin_centers, 0.2*mean_r, ms=1,
                marker='o', lw=0.5, mew=0, c='C3')
        ax.set_xlabel(r'Excitation Error $s_{g}$')
        ax.set_ylabel(r'${\rm I_{\rm o}}/{\rm I_{\rm c}}$')

        plt.savefig(f'all_{set_idx}.png', dpi=400)
