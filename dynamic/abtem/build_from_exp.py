import abtem
import ase
import sys   # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from abtem.bloch import BlochWaves, StructureFactor
from dynamic.abtem.plot import plot_spots
from dynamic.extract_dials_experiment import extract_expt
from dynamic.extract_dials_experiment import axis_angle_rotation_matrix


def build_from_exp(cif_file, expt_file, plot_cell=True):
    """
    Process a batch of substep angles in one worker process.

    Returns
    -------
    partial_image : ndarray (npy, npx)
    rocking_partial : dict  hkl -> list of float
    last_positions, last_millers, last_mean_ints
    """
    abtem.config.set({"mkl.threads": 2})
    abtem.config.set({"fftw.threads": 2})

    atoms_base = ase.io.read(cif_file)

    (ort, S, F, U, B, angles, rotation_axis) = extract_expt(expt_file)

    thickness_A = 30.0
    tilt_rad = 0

    thick_corr = thickness_A / np.cos(tilt_rad)
    thickness_A = thick_corr * 10.0

    atoms = atoms_base.copy()
    # new_crystal_cell = np.linalg.inv(B)
    # atoms.set_cell(new_crystal_cell, scale_atoms=True)

    # angles_span = np.linspace(0, 180, 180)
    original_cell = atoms.get_cell()
    oc = np.array(original_cell)
    print("original cell")
    print(oc)
    print("New cell")
    print(np.linalg.inv(B).T)
    print("Ortogonalization matrix")
    print(ort)

    for i, angle in enumerate(angles):

        if i < 100:

            print(i)

            R = axis_angle_rotation_matrix(rotation_axis, angle)
            new_cell = (S @ R @ F @ U @ original_cell.T).T
            atoms.set_cell(new_cell, scale_atoms=True)
            atoms.wrap()

            if plot_cell:
                fig = plt.figure(figsize=(3.375, 3.0))
                ax1 = fig.add_axes([0.13, 0.14, 0.84, 0.82])
                abtem.show_atoms(atoms, ax=ax1, plane="xz", show_cell=True)
                plt.xlim(-20, 20)
                plt.ylim(-20, 20)
                plt.savefig(f'system_bloch_{i:04d}.png', dpi=400)
                plt.close(fig)

            positions, millers, mean_ints = bloch_diffraction(
                atoms=atoms,
                thickness_angs=50.0,
                k_max=3.0,
                sg_max=0.05,
                energy_eV=200e3,
                num_phonon_configs=1,
                phonon_sigmas=0,
                phonon_seed=42)
            out_file = f'spots_{i:04d}.png'
            plot_spots(positions, millers, mean_ints, out_file,
                       top_spots=200)

    return positions, millers, mean_ints


def bloch_diffraction(atoms, thickness_angs, k_max, sg_max, energy_eV,
                      num_phonon_configs, phonon_sigmas, phonon_seed):
    """
    Bloch wave simulation for one orientation, averaged over
    frozen phonon configurations.

    Returns
    -------
    positions : ndarray (N, 3)  kx, ky, kz in Å⁻¹
    miller_indices : list
    mean_intensities : ndarray (N,)
    """
    ff = abtem.FrozenPhonons
    frozen_phonons = ff(atoms, num_configs=num_phonon_configs,
                        sigmas=phonon_sigmas, seed=phonon_seed)

    accumulated = None
    positions = None
    millers = None

    for cfg_idx, atoms_cfg in enumerate(frozen_phonons):

        sf = StructureFactor(atoms_cfg, k_max=k_max,
                             parametrization="lobato")

        bw = BlochWaves(structure_factor=sf, energy=200e3,
                        sg_max=sg_max)

        if cfg_idx == 0:
            print(f"Beams: {len(bw)}")

        patterns = bw.calculate_diffraction_patterns([thickness_angs])
        patterns = patterns.compute()

        spots = patterns[0].remove_low_intensity(1e-20)
        print("Nonzero", len(spots))
        ints = np.array(spots.intensities)

        if accumulated is None:
            accumulated = ints.copy()
            positions = spots.positions
            millers = spots.miller_indices
        elif len(ints) == len(accumulated):
            accumulated += ints
        else:
            print(f"WARNING: phonon config {cfg_idx}"
                  f"has {len(ints)} spots vs "
                  f"{len(accumulated)} — skipping.")

        mean_intensities = accumulated / num_phonon_configs
        return positions, millers, mean_intensities
