#!/usr/bin/env python3
import abtem
from abtem.bloch import BlochWaves, StructureFactor
import ase
import matplotlib.pyplot as plt
import numpy as np
import numba

numba.set_num_threads(2)


def make_system_bloch(cif_file, angles_file, thickness_nm=1, image_index=0,
                      show=False, k_max=5, output_path=None, sg_max=0.1):

    abtem.config.set({"mkl.threads": 10})
    abtem.config.set({"fftw.threads": 10})

    data = np.load(angles_file)

    angles = data['angles']

    if len(angles) <= image_index:
        return False

    initial_orientations = data['initial_orientations']

    alpha, beta, gamma = angles[image_index]
    a0, b0, c0 = initial_orientations[image_index]

    tpb = ase.io.read(cif_file)

    # Set the experimental unit cell
    new_cell = np.array([a0, b0, c0])
    tpb.set_cell(new_cell, scale_atoms=False)

    if show:
        abtem.show_atoms(tpb, plane="xy", scale=0.5, legend=True)

    # Rotate the atoms
    tpb.rotate(alpha, 'x', rotate_cell=True)
    tpb.rotate(beta, 'y', rotate_cell=True)
    tpb.rotate(gamma, 'z', rotate_cell=True)
    tpb.wrap()

    fig = plt.figure(figsize=(3.375, 3.0))
    ax1 = fig.add_axes([0.13, 0.14, 0.84, 0.82])
    abtem.show_atoms(tpb, ax=ax1, plane="xz")
    plt.savefig('system_bloch.png', dpi=400)

    print("Atoms in unit cell:", len(tpb))

    # Create structure factor first
    # k_max = 12  # controls accuracy
    structure_factor = StructureFactor(
        tpb,
        k_max=k_max,
        parametrization="lobato",
    )

    # Create Bloch waves from structure factor
    sg_max = sg_max  # maximum scattering angle
    bloch_waves = BlochWaves(
        structure_factor=structure_factor,
        energy=200e3,
        sg_max=sg_max,
    )

    print(f"Number of beams: {len(bloch_waves)}")
    thickness = thickness_nm * 10
    thicknesses = [thickness]  # can be a list of thicknesses

    print("Calculating diffraction patterns at thickness:")
    print(f"  {thickness:.1f} Angstroms")
    dif_patterns = bloch_waves.calculate_diffraction_patterns(thicknesses)

    print("Computing diffraction patterns")
    dif_patterns = dif_patterns.compute()

    out_str = f"{image_index:04d}_z_{thickness_nm:05.1f}_nm_"
    out_str += f"kmax_{k_max:03d}"

    # Get the first (only) thickness
    dif_pattern = dif_patterns[0]

    print("Indexing diffraction spots")
    spots = dif_pattern.remove_low_intensity(1e-20)

    all_pos = spots.positions
    all_miller = spots.miller_indices
    intensities = np.array(spots.intensities)

    plot_spots(all_pos, all_miller, intensities, out_str,
               out_path=output_path)

    # Save spots
    df = spots.to_dataframe()
    out_file = f"spots_bloch_{out_str}.npz"
    if output_path is not None:
        out_file = output_path + "/" + out_file
    save_as_npz(df, out_file)

    return True


def plot_spots(all_pos, all_miller, intensities, out_str, top_spots=60,
               out_path=None):

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.15, 0.12, 0.80, 0.84])
    ax.set_xlabel(r'$k_{\rm x}\ (\rm \AA)$')
    ax.set_ylabel(r'$k_{\rm y}\ (\rm \AA)$')

    # Find top 60 spots
    if len(intensities) > top_spots:
        icutoff = np.argsort(intensities)[-top_spots]
        intensity_cutoff = intensities[icutoff]
    else:
        intensity_cutoff = 0

    kxs = []
    kys = []
    ints_all = []
    kxs_other = []
    kys_other = []

    for pos, miller, ints in zip(all_pos, all_miller, intensities):
        if ints > intensity_cutoff:
            kx, ky, kz = pos  # Bloch wave positions are 2D
            kxs.append(kx)
            kys.append(ky)
            ints_all.append(ints)
            label = f"{miller[0]} {miller[1]} {miller[2]}"
            plt.text(kx, ky+0.10, label, color='blue', va='top', ha='center',
                     fontsize=4, bbox=dict(facecolor='white',
                                           alpha=0.5,
                                           edgecolor='none',
                                           boxstyle='square, pad=0.3'
                                           )
                     )
        else:
            kx, ky, kz = pos  # Bloch wave positions are 2D
            kxs_other.append(kx)
            kys_other.append(ky)

    ax.scatter(kxs_other, kys_other, marker='o', s=0.5, c='#BEBEBE',
               linewidths=0)
    ax.scatter(kxs, kys, marker='o', s=1 + 2*abs(np.log(ints_all)),
               c='C3', linewidths=0, cmap='jet')
    out_file = f'image_{out_str}.png'

    if out_path is not None:
        out_file = out_path + '/' + out_file
    plt.savefig(out_file, dpi=400)
    plt.close(fig)


def save_as_npz(df, filename):

    dd = df.iloc[0]
    millers = dd.index.tolist()

    millers_all = []
    intensities_all = []

    h_all = []
    k_all = []
    l_all = []

    for miller in millers:
        hs, ks, ls = miller.split()
        h = int(hs)
        k = int(ks)
        ll = int(ls)

        intensity = float(dd.loc[miller])

        millers_all.append([h, k, ll])
        h_all.append(h)
        k_all.append(k)
        l_all.append(ll)
        intensities_all.append(intensity)

    np.savez(filename, h=h_all, k=k_all, l=l_all, miller=millers_all,
             intensity=intensities_all)


def rotation_matrix_from_angles(alpha, beta, gamma, degrees=True):
    """
    Build a rotation matrix from Euler angles (x=alpha, y=beta, z=gamma).
    """
    if degrees:
        alpha, beta, gamma = np.deg2rad([alpha, beta, gamma])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha),  np.cos(alpha)]
    ])
    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    Rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma),  np.cos(gamma), 0],
        [0, 0, 1]
    ])

    # total rotation = Rz * Ry * Rx
    return Rz @ Ry @ Rx


def rotate_unit_cell(cell, alpha, beta, gamma, degrees=True):
    """
    Rotate a unit cell [a0, b0, c0] by Euler angles
    (x=alpha, y=beta, z=gamma).
    """
    R = rotation_matrix_from_angles(alpha, beta, gamma, degrees=degrees)
    return (R @ cell.T).T  # rotate each vector
