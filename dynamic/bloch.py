#!/usr/bin/env python3
import abtem
from abtem.bloch import BlochWaves, StructureFactor
import ase
import matplotlib.pyplot as plt
import numpy as np
import numba
numba.set_num_threads(2)


def make_system_bloch(cif_file, angles_file,
                      thickness_nm=1,
                      image_index=0,
                      show=False,
                      k_max=5,
                      output_path=None,
                      sg_max=0.1,
                      correction_model='cuboid',
                      oscillation_deg=1,
                      subsamples=50,
                      start_angle_deg=-30,
                      # --- Frozen phonon parameters ---
                      num_phonon_configs=10,
                      phonon_sigmas=0.1,
                      phonon_seed=42,
                      ):
    """
    phonon_sigmas : float or dict
        Standard deviation of atomic displacements in Angstrom.
        Use a float for all atoms identically, e.g. 0.1 A is a
        reasonable default if you don't have Debye-Waller factors.
        Use a dict per species e.g. {'C': 0.08, 'H': 0.14, 'N': 0.07}
        for more realistic values from published Debye-Waller data.
    num_phonon_configs : int
        Number of frozen phonon snapshots to average over.
        10 is usually sufficient for Bloch wave; increase to 20-30
        for thicker samples or if intensities are not converged.
    """
    abtem.config.set({"mkl.threads": 10})
    abtem.config.set({"fftw.threads": 10})

    data = np.load(angles_file)
    angles = data['angles']
    subsamples = data['subsamples']
    n_angles = len(angles)
    nimg = (n_angles - 1) / subsamples + 1
    rot_angles = np.linspace(start_angle_deg,
                             start_angle_deg + (nimg-1)*oscillation_deg,
                             n_angles)
    rotation_angle = rot_angles[image_index] * np.pi / 180.
    if correction_model == 'cuboid':
        thickness_nm_corr = thickness_nm / np.cos(rotation_angle)
    else:
        thickness_nm_corr = thickness_nm

    if len(angles) <= image_index:
        return False

    initial_orientations = data['initial_orientations']
    alpha, beta, gamma = angles[image_index]
    a0, b0, c0 = initial_orientations

    tpb = ase.io.read(cif_file)
    new_cell = np.array([a0, b0, c0])
    tpb.set_cell(new_cell, scale_atoms=False)

    if show:
        abtem.show_atoms(tpb, plane="xy", scale=0.5, legend=True)

    tpb.rotate(alpha, 'x', rotate_cell=True)
    tpb.rotate(beta,  'y', rotate_cell=True)
    tpb.rotate(gamma, 'z', rotate_cell=True)
    tpb.wrap()

    fig = plt.figure(figsize=(3.375, 3.0))
    ax1 = fig.add_axes([0.13, 0.14, 0.84, 0.82])
    abtem.show_atoms(tpb, ax=ax1, plane="xz")
    plt.savefig('system_bloch.png', dpi=400)
    plt.close(fig)
    print("Atoms in unit cell:", len(tpb))

    thickness = thickness_nm_corr * 10
    thicknesses = [thickness]
    print("Calculating diffraction patterns at thickness:" +
          f"{thickness:.1f} Angstroms")

    out_str = f"{image_index:06d}_z_{thickness_nm:05.1f}_nm_kmax_{k_max:03d}"

    # ------------------------------------------------------------------
    # Frozen phonon loop
    # FrozenPhonons doesn't plug directly into StructureFactor/BlochWaves
    # the way it does for multislice Potential. Instead we iterate over
    # displaced configurations manually and average the intensities.
    # ------------------------------------------------------------------
    frozen_phonons = abtem.FrozenPhonons(
        tpb,
        num_configs=num_phonon_configs,
        sigmas=phonon_sigmas,
        seed=phonon_seed,
    )

    # We'll accumulate intensities keyed by Miller index across configs.
    # On the first config we record the spot positions and Miller indices;
    # subsequent configs must match (same hkl set from same BlochWaves).
    accumulated_intensities = None
    reference_positions = None
    reference_millers = None

    for config_idx, atoms_config in enumerate(frozen_phonons):
        print(f"  Phonon config {config_idx + 1} / {num_phonon_configs}")

        structure_factor = StructureFactor(
            atoms_config,
            k_max=k_max,
            parametrization="lobato",
        )

        bloch_waves = BlochWaves(
            structure_factor=structure_factor,
            energy=200e3,
            sg_max=sg_max,
        )

        if config_idx == 0:
            print(f"  Number of beams: {len(bloch_waves)}")

        dif_patterns = bloch_waves.calculate_diffraction_patterns(thicknesses)
        dif_patterns = dif_patterns.compute()
        dif_pattern = dif_patterns[0]

        spots = dif_pattern.remove_low_intensity(1e-20)
        intensities = np.array(spots.intensities)

        if accumulated_intensities is None:
            accumulated_intensities = intensities.copy()
            reference_positions = spots.positions
            reference_millers = spots.miller_indices
        else:
            # Shapes should match since the same BlochWaves hkl set is used.
            # Guard against edge case where a config drops/adds a beam.
            if len(intensities) == len(accumulated_intensities):
                accumulated_intensities += intensities
            else:
                print(f"  WARNING: config {config_idx} has different number "
                      f"of spots ({len(intensities)} vs "
                      f"{len(accumulated_intensities)}), skipping.")

    # Average over all configs
    mean_intensities = accumulated_intensities / num_phonon_configs

    print("Phonon-averaged diffraction patterns computed.")

    plot_spots(reference_positions, reference_millers, mean_intensities,
               out_str, out_path=output_path)

    # Save averaged spots — reuse the dataframe structure from the last config
    # but replace intensities with the averaged values
    df = spots.to_dataframe()
    # Overwrite the intensity row with averaged values
    df.iloc[0] = mean_intensities

    out_file = f"spots_bloch_{out_str}.npz"
    if output_path is not None:
        out_file = output_path + "/" + out_file
    save_as_npz_phonon(reference_millers, mean_intensities, out_file)

    return True


def save_as_npz_phonon(miller_indices, intensities, filename):
    """
    Save averaged frozen phonon spot data directly
    from miller/intensity arrays.
    """
    h_all, k_all, l_all, millers_all = [], [], [], []
    for miller, intensity in zip(miller_indices, intensities):
        h, k, ll = int(miller[0]), int(miller[1]), int(miller[2])
        h_all.append(h)
        k_all.append(k)
        l_all.append(ll)
        millers_all.append([h, k, ll])
    np.savez(filename,
             h=h_all, k=k_all, l=l_all,
             miller=millers_all,
             intensity=list(intensities))


# ---- unchanged helpers below ----

def plot_spots(all_pos, all_miller, intensities, out_str, top_spots=60,
               out_path=None):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.15, 0.12, 0.80, 0.84])
    ax.set_xlabel(r'$k_{\rm x}\ (\rm \AA)$')
    ax.set_ylabel(r'$k_{\rm y}\ (\rm \AA)$')
    if len(intensities) > top_spots:
        icutoff = np.argsort(intensities)[-top_spots]
        intensity_cutoff = intensities[icutoff]
    else:
        intensity_cutoff = 0
    kxs, kys, ints_all = [], [], []
    kxs_other, kys_other = [], []
    for pos, miller, ints in zip(all_pos, all_miller, intensities):
        kx, ky, kz = pos
        if ints > intensity_cutoff:
            kxs.append(kx)
            kys.append(ky)
            ints_all.append(ints)
            label = f"{miller[0]} {miller[1]} {miller[2]}"
            plt.text(kx, ky + 0.10, label, color='blue', va='top',
                     ha='center', fontsize=4,
                     bbox=dict(facecolor='white', alpha=0.5,
                               edgecolor='none',
                               boxstyle='square, pad=0.3'))
        else:
            kxs_other.append(kx)
            kys_other.append(ky)
    ax.scatter(kxs_other, kys_other, marker='o', s=0.5,
               c='#BEBEBE', linewidths=0)
    ax.scatter(kxs, kys, marker='o',
               s=1 + 2 * abs(np.log(ints_all)),
               c='C3', linewidths=0, cmap='jet')
    out_file = f'image_{out_str}.png'
    if out_path is not None:
        out_file = out_path + '/' + out_file
    plt.savefig(out_file, dpi=400)
    plt.close(fig)


def save_as_npz(df, filename):
    dd = df.iloc[0]
    millers = dd.index.tolist()
    millers_all, intensities_all = [], []
    h_all, k_all, l_all = [], [], []
    for miller in millers:
        hs, ks, ls = miller.split()
        h, k, ll = int(hs), int(ks), int(ls)
        intensity = float(dd.loc[miller])
        millers_all.append([h, k, ll])
        h_all.append(h)
        k_all.append(k)
        l_all.append(ll)
        intensities_all.append(intensity)
    np.savez(filename, h=h_all, k=k_all, l=l_all,
             miller=millers_all, intensity=intensities_all)


def rotation_matrix_from_angles(alpha, beta, gamma, degrees=True):
    if degrees:
        alpha, beta, gamma = np.deg2rad([alpha, beta, gamma])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha),  np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma),  np.cos(gamma), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotate_unit_cell(cell, alpha, beta, gamma, degrees=True):
    R = rotation_matrix_from_angles(alpha, beta, gamma, degrees=degrees)
    return (R @ cell.T).T
