#!/usr/bin/env python3
import abtem
import ase
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
import numba

numba.set_num_threads(2)


def make_system(cif_file, angles_file, thickness_nm=1, size_xy_nm=None,
                shape='cylinder', uncut_block_size=(50, 50, 50),
                image_index=1, show=False,
                sampling=0.05, slice_thickness=1.0, gpts=512,
                extent=100):

    abtem.config.set({"mkl.threads": 10})
    abtem.config.set({"fftw.threads": 10})

    data = np.load(angles_file)

    angles = data['angles']

    if len(angles) <= image_index:
        return False

    initial_orientations = data['initial_orientations']
    original_file = data['original_file']

    print(f"Original file: {original_file}")

    alpha, beta, gamma = angles[image_index]
    a0, b0, c0 = initial_orientations[image_index]

    tpb = ase.io.read(cif_file)

    new_cell = np.array([a0, b0, c0])
    tpb.set_cell(new_cell, scale_atoms=False)
    unit_cell = np.array(tpb.cell)

    rotated_unit_cell = rotate_unit_cell(unit_cell, alpha=alpha,
                                         beta=beta, gamma=gamma,
                                         degrees=True)
    if show:
        abtem.show_atoms(tpb, plane="xy", scale=0.5, legend=True)

    sx, sy, sz = uncut_block_size
    repeated_tpb = tpb * (sx, sy, sz)
    com = repeated_tpb.get_center_of_mass()
    repeated_tpb.translate(-com)

    a_cell = unit_cell[0][0]
    b_cell = unit_cell[1][1]
    c_cell = unit_cell[2][2]

    print("Uncut block size: ")
    ss = f"({sx}, {sy}, {sz}) "
    ss += f"-> {0.1*a_cell*sx:.0f} "
    ss += f"{0.1*b_cell*sy:.0f} {0.1*c_cell*sz:.0f} nm"
    print(ss)

    thickness_angs = thickness_nm * 10
    size_xy_angs = size_xy_nm * 10

    repeated_tpb.rotate(alpha, 'x', rotate_cell=False)
    repeated_tpb.rotate(beta, 'y', rotate_cell=False)
    repeated_tpb.rotate(gamma, 'z', rotate_cell=False)

    if shape == 'sphere':

        repeated_tpb = cut_to_sphere(repeated_tpb,
                                     radius_angs=thickness_angs,
                                     center=(0, 0, 0))

    elif shape == 'cuboid':

        repeated_tpb = cut_to_cuboid(repeated_tpb,
                                     xy_size=size_xy_angs,
                                     z_size=thickness_angs,
                                     center=(0, 0, 0))

    elif shape == 'cylinder':

        repeated_tpb = cut_to_cylinder(repeated_tpb,
                                       radius=size_xy_angs / 2.0,
                                       height=thickness_angs,
                                       center=(0, 0, 0))
    repeated_tpb.center(vacuum=2)

    fig = plt.figure(figsize=(3.375, 3.0))
    ax1 = fig.add_axes([0.13, 0.14, 0.84, 0.82])

    abtem.show_atoms(repeated_tpb, ax=ax1, plane="xz")
    base = f"{shape}_z_{thickness_nm:05.1f}_nm_xy_{size_xy_nm:05.1f}_nm"
    out_str = f"{image_index:03d}_{base}"
    plt.savefig(f'system_{base}_multiclice.png', dpi=400)

    nx = int(size_xy_angs / a_cell)
    ny = int(size_xy_angs / b_cell)
    nz = int(thickness_angs / c_cell)
    print(f"Unit cells in shape (Nx Ny Nz) = {nx} {ny} {nz}")

    print("Atoms in sphere:", len(repeated_tpb))

    potential = abtem.Potential(repeated_tpb, sampling=sampling,
                                parametrization="lobato",
                                slice_thickness=slice_thickness,
                                projection="finite")

    plane_wave = abtem.PlaneWave(gpts=gpts, extent=extent, energy=200e3)
    waves = plane_wave.build()
    waves.compute()

    print("Doing multislice")
    exit_wave = plane_wave.multislice(potential, max_batch=8)

    print("Computing the exit wave")
    exit_wave.compute()

    print("Computing diffraction patterns")
    diff_patterns = exit_wave.diffraction_patterns()

    fig, dax = plt.subplots(figsize=(6, 6))

    print("Computing detector projection")
    diff_patterns.crop(30).block_direct().show(cbar=True,
                                               vmax=1.e-5,
                                               vmin=-1.e-10, ax=dax)

    rc = rotated_unit_cell
    print("Computing spots")
    spots = diff_patterns.crop(120).index_diffraction_spots(cell=rc)

    all_pos = spots.all_positions
    all_miller = spots.miller_indices
    intensities = np.array(spots.intensities)

    kxs = []
    kys = []

    icutoff = np.argsort(intensities)[-100]
    intensity_cutoff = intensities[icutoff]
    for pos, miller, ints in zip(all_pos, all_miller, intensities):
        if ints > intensity_cutoff:

            kx, ky, kz = pos
            kxs.append(kx)
            kys.append(ky)
            label = f"{miller[0]} {miller[1]} {miller[2]}"
            plt.text(kx, ky+0.05, label, color='blue', va='top', ha='center',
                     fontsize=1, bbox=dict(facecolor='white',
                                           alpha=0.5,
                                           edgecolor='none',
                                           boxstyle='square, pad=0.3'
                                           )
                     )
    plt.scatter(kxs, kys, marker='o', s=0.5, color='red')
    plt.savefig(f'image_{out_str}.png', dpi=800)

    df = spots.remove_low_intensity(1.e-20).to_dataframe()
    save_as_npz(df, f"spots_{out_str}.npz")
    return True


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


def cut_to_cylinder(
        atoms: Atoms,
        radius: float = 20.0,
        height: float = 100.0,
        center: np.ndarray | None = None,
        keep_pbc: bool = False) -> Atoms:
    """
    Return a new ASE Atoms object containing only atoms inside a cylinder.
    The cylinder axis is along z.

    Parameters
    ----------
    atoms : ase.Atoms
        Input atomic configuration (can be from abtem).
    radius : float
        Radius of the cylinder in Angstrom.
    height : float
        Height of the cylinder along z in Angstrom.
    center : array_like or None
        Centre of the cylinder in Angstrom (3 floats).
        If None, use center of mass.
    keep_pbc : bool
        If False (default) the result will have pbc=False and an orthogonal
        cell sized to contain the cylinder (+1 Å margin).
        If True, original pbc/cell kept.

    Returns
    -------
    ase.Atoms
        New Atoms object with the selected atoms.
    """
    if center is None:
        center = atoms.get_center_of_mass()
    else:
        center = np.asarray(center, dtype=float)

    positions = atoms.get_positions()
    rel_pos = positions - center

    # Distance from z-axis (radial distance in xy plane)
    r_xy = np.sqrt(rel_pos[:, 0]**2 + rel_pos[:, 1]**2)

    # Check if atoms are inside the cylinder
    mask = (
        (r_xy <= radius) &                        # inside radius
        (np.abs(rel_pos[:, 2]) <= height / 2)     # inside height
    )

    # Select atoms
    cylinder_atoms = atoms[mask].copy()

    # Adjust cell / pbc if desired
    if not keep_pbc:
        cylinder_atoms.set_pbc(False)

        # Set a cubic cell that comfortably contains the cylinder
        margin = 1.0  # Å margin
        diameter = 2 * radius
        Lxy = diameter + 2 * margin
        Lz = height + 2 * margin

        cylinder_atoms.set_cell([Lxy, Lxy, Lz])

        # Recenter positions into the new cell
        # Place cylinder centre at cell centre
        new_center = np.array([Lxy/2, Lxy/2, Lz/2])

        position = cylinder_atoms.get_positions() - center + new_center
        cylinder_atoms.set_positions(position)

    return cylinder_atoms


def cut_to_cuboid(
        atoms: Atoms,
        xy_size: float = 20.0,
        z_size: float = 100.0,
        center: np.ndarray | None = None,
        keep_pbc: bool = False) -> Atoms:
    """
    Return a new ASE Atoms object containing only atoms inside a cuboid.

    Parameters
    ----------
    atoms : ase.Atoms
        Input atomic configuration (can be from abtem).
    xy_size : float
        Width of the cuboid in x and y directions in Angstrom.
    z_size : float
        Height of the cuboid in z direction in Angstrom.
    center : array_like or None
        Centre of the cuboid in Angstrom (3 floats).
        If None, use center of mass.
    keep_pbc : bool
        If False (default) the result will have pbc=False and an orthogonal
        cell sized to contain the cuboid (+1 Å margin).
        If True, original pbc/cell kept.

    Returns
    -------
    ase.Atoms
        New Atoms object with the selected atoms.
    """
    if center is None:
        center = atoms.get_center_of_mass()
    else:
        center = np.asarray(center, dtype=float)

    positions = atoms.get_positions()

    # Calculate distance from center in each direction
    rel_pos = positions - center

    # Check if atoms are inside the cuboid
    mask = (
        (np.abs(rel_pos[:, 0]) <= xy_size / 2) &  # x bounds
        (np.abs(rel_pos[:, 1]) <= xy_size / 2) &  # y bounds
        (np.abs(rel_pos[:, 2]) <= z_size / 2)     # z bounds
    )

    # Select atoms
    cuboid_atoms = atoms[mask].copy()

    # Adjust cell / pbc if desired
    if not keep_pbc:
        cuboid_atoms.set_pbc(False)

        # Set a cell that comfortably contains the cuboid
        margin = 1.0  # Å margin around cuboid
        Lx = xy_size + 2 * margin
        Ly = xy_size + 2 * margin
        Lz = z_size + 2 * margin

        cuboid_atoms.set_cell([Lx, Ly, Lz])

        # Recenter positions into the new cell
        # Place cuboid centre at cell centre
        new_center = np.array([Lx/2, Ly/2, Lz/2])

        position = cuboid_atoms.get_positions() - center + new_center
        cuboid_atoms.set_positions(position)

    return cuboid_atoms


def cut_to_sphere(
        atoms: Atoms,
        radius: float = 10.0,
        center: np.ndarray | None = None,
        keep_pbc: bool = False) -> Atoms:
    """
    Return a new ASE Atoms object containing only atoms inside a sphere.

    Parameters
    ----------
    atoms : ase.Atoms
        Input atomic configuration (can be from abtem).
    radius : float
        Sphere radius in Angstrom.
    center : array_like or None
        Centre of the sphere in Angstrom (3 floats).
        If None, use center of mass.
    keep_pbc : bool
        If False (default) the result will have pbc=False and an orthogonal
        cell sized to contain the sphere (+1 Å margin).
        If True, original pbc/cell kept.

    Returns
    -------
    ase.Atoms
        New Atoms object with the selected atoms.
    """
    if center is None:
        center = atoms.get_center_of_mass()
    else:
        center = np.asarray(center, dtype=float)

    diameter = radius * 2.0
    positions = atoms.get_positions()
    dists = np.linalg.norm(positions - center, axis=1)
    mask = dists <= radius  # <= keeps atoms exactly on the surface

    # select atoms
    sphere_atoms = atoms[mask].copy()

    # adjust cell / pbc if desired
    if not keep_pbc:
        sphere_atoms.set_pbc(False)

        # Set a small cubic cell that comfortably contains the sphere
        margin = 1.0  # Å margin around sphere
        L = diameter + 2 * margin
        sphere_atoms.set_cell(np.eye(3) * L)

        # Recenter positions into the new cell
        # Place sphere centre at cell centre
        new_center = np.array([L/2, L/2, L/2])

        position = sphere_atoms.get_positions() - center + new_center

        sphere_atoms.set_positions(position)

    return sphere_atoms


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

    Parameters
    ----------
    cell : array_like
        3x3 matrix, columns (or rows) are [a0, b0, c0] vectors
    alpha, beta, gamma : float
        rotation angles around x, y, z axes
    degrees : bool
        if True, interpret angles as degrees (default)

    Returns
    -------
    rotated_cell : ndarray
        3x3 array with rotated [a, b, c] vectors
    """
    R = rotation_matrix_from_angles(alpha, beta, gamma, degrees=degrees)
    return (R @ cell.T).T  # rotate each vector
