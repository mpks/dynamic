#!/usr/bin/env python3
"""
Bloch wave diffraction simulation with synthetic rotation scan.

This script:
  - Draws a random starting crystal orientation (seeded)
  - Draws a random rotation axis (seeded)
  - Draws a random crystal thickness in [20, 200] nm range (seeded)
  - Rotates the crystal from `start_angle` to `end_angle` in steps of
    `delta` around that axis, and simulates the diffraction pattern for
    `image_index`.

Usage example
-------------
python bloch_synthetic.py my_structure.cif \
    --image_index 1345 \
    --start_angle -15 \
    --end_angle    15 \
    --delta         0.01 \
    --seed         42
"""
import argparse

import abtem
from abtem.bloch import BlochWaves, StructureFactor
import ase
import ase.io
import matplotlib.pyplot as plt
import numpy as np
import numba
from scipy.spatial.transform import Rotation

numba.set_num_threads(2)


def main():

    args = _parse_args()
    make_system_bloch_synthetic(
        cif_file=args.cif_file,
        image_index=args.image_index,
        start_angle=args.start_angle,
        end_angle=args.end_angle,
        delta=args.delta,
        seed=args.seed,
        show=args.show,
        k_max=args.k_max,
        sg_max=args.sg_max,
        output_path=args.output_path,
        num_phonon_configs=args.num_phonon_configs,
        phonon_sigmas=args.phonon_sigmas,
        phonon_seed=args.phonon_seed,
    )

# ---------------------------------------------------------------------------
# Synthetic geometry helpers
# ---------------------------------------------------------------------------


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    """Return a uniformly distributed random unit vector on the sphere."""
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """
    Return a uniformly distributed random 3-D rotation matrix
    using the Haar-measure algorithm (Shoemake 1992).
    """
    u1, u2, u3 = rng.uniform(0, 1, 3)
    q = np.array([
            np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
            np.sqrt(u1) * np.sin(2 * np.pi * u3),
            np.sqrt(u1) * np.cos(2 * np.pi * u3),
           ])
    # Quaternion → rotation matrix
    return Rotation.from_quat(q).as_matrix()


def rotation_about_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotation matrix for `angle_deg` degrees around unit `axis`."""
    axis = axis / np.linalg.norm(axis)
    return Rotation.from_rotvec(np.deg2rad(angle_deg) * axis).as_matrix()


def build_synthetic_geometry(seed: int):
    """
    Return a consistent set of synthetic experiment parameters derived
    from `seed`:

        initial_R  – 3×3 rotation matrix: random starting orientation
        axis – unit vector: rotation axis for the scan
        thickness_nm – crystal thickness in nm, drawn from U[20, 200]

    All three are fully determined by `seed`, so identical seeds reproduce
    identical virtual experiments.
    """
    rng = np.random.default_rng(seed)
    initial_R = random_rotation_matrix(rng)
    axis = random_unit_vector(rng)
    thickness_nm = float(rng.uniform(20.0, 200.0))
    return initial_R, axis, thickness_nm


def make_system_bloch_synthetic(
        cif_file: str,
        image_index: int,
        start_angle: float = -15.0,
        end_angle: float = 15.0,
        delta: float = 0.01,
        seed: int = 42,
        show: bool = False,
        k_max: float = 5,
        output_path: str | None = None,
        sg_max: float = 0.1,
        num_phonon_configs: int = 10,
        phonon_sigmas: float | dict = 0.1,
        phonon_seed: int = 42,
) -> bool:
    """
    Simulate the diffraction pattern for `image_index` in a synthetic
    rotation scan defined by [start_angle, end_angle, delta].

    Parameters
    ----------
    cif_file : str
        Path to the CIF structure file.
    image_index : int
        Zero-based index of the image to simulate (e.g. 1345 out of 3000).
    start_angle, end_angle : float
        Angular range of the scan in degrees.
    delta : float
        Angular step size in degrees.
    seed : int
        Master seed that fixes the starting orientation, rotation axis,
        and crystal thickness.  Use the same seed for every image in the
        same virtual experiment.
    num_phonon_configs : int
        Number of frozen-phonon snapshots to average.
    phonon_sigmas : float or dict
        Atomic displacement amplitudes in Angstrom.
    phonon_seed : int
        Seed for the frozen-phonon random displacements.
    """

    abtem.config.set({"mkl.threads": 10})
    abtem.config.set({"fftw.threads": 10})

    # ------------------------------------------------------------------
    # 1. Build the synthetic scan geometry
    # ------------------------------------------------------------------
    initial_R, axis, thickness_nm = build_synthetic_geometry(seed)

    # Angle for this particular image
    n_images = int(round((end_angle - start_angle) / delta)) + 1
    if image_index < 0 or image_index >= n_images:
        raise ValueError(
            f"image_index {image_index} is out of range "
            f"[0, {n_images - 1}] for the given scan parameters."
        )

    scan_angle = start_angle + image_index * delta   # degrees
    print(f"[Synthetic] seed={seed}")
    print(f"[Synthetic] Rotation axis: {axis}")
    print(f"[Synthetic] Thickness: {thickness_nm:.1f} nm")
    print(f"[Synthetic] Scan angle for image {image_index}: "
          f"{scan_angle:.4f} deg  (range {start_angle}…{end_angle}, "
          f"Δ={delta} deg, {n_images} images total)")

    # Total rotation = initial random orientation + scan tilt around axis
    scan_R = rotation_about_axis(axis, scan_angle)
    total_R = scan_R @ initial_R     # apply scan on top of initial

    # Optional cosine correction for thickness (beam path length)
    tilt_rad = np.deg2rad(scan_angle)
    thickness_nm_corr = thickness_nm / np.cos(tilt_rad)

    # ------------------------------------------------------------------
    # 2. Load and orient the crystal
    # ------------------------------------------------------------------
    atoms = ase.io.read(cif_file)

    if show:
        abtem.show_atoms(atoms, plane="xy", scale=0.5, legend=True)

    # Apply the combined rotation to the unit cell vectors
    cell = atoms.get_cell()
    new_cell = (total_R @ cell.array.T).T
    atoms.set_cell(new_cell, scale_atoms=False)
    atoms.wrap()

    # Diagnostic figure
    fig = plt.figure(figsize=(3.375, 3.0))
    ax1 = fig.add_axes([0.13, 0.14, 0.84, 0.82])
    abtem.show_atoms(atoms, ax=ax1, plane="xz")
    plt.savefig("system_bloch.png", dpi=400)
    plt.close(fig)
    print(f"Atoms in unit cell: {len(atoms)}")

    # ------------------------------------------------------------------
    # 3. Bloch wave simulation with frozen phonons
    # ------------------------------------------------------------------
    thickness_A = thickness_nm_corr * 10.0   # nm → Ångström
    thicknesses = [thickness_A]
    print(f"Thickness used for simulation: {thickness_A:.1f} Å "
          f"({thickness_nm_corr:.2f} nm, corrected for tilt)")

    out_str = (f"{image_index:06d}_kmax_{k_max:03d}")

    frozen_phonons = abtem.FrozenPhonons(atoms,
                                         num_configs=num_phonon_configs,
                                         sigmas=phonon_sigmas,
                                         seed=phonon_seed,
                                         )

    accumulated_intensities = None
    reference_positions = None
    reference_millers = None
    spots = None   # keep last for dataframe structure

    for config_idx, atoms_config in enumerate(frozen_phonons):
        print(f"  Phonon config {config_idx + 1} / {num_phonon_configs}")

        structure_factor = StructureFactor(atoms_config,
                                           k_max=k_max,
                                           parametrization="lobato",
                                           )

        bloch_waves = BlochWaves(structure_factor=structure_factor,
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
            if len(intensities) == len(accumulated_intensities):
                accumulated_intensities += intensities
            else:
                print(f"  WARNING: config {config_idx} has "
                      f"{len(intensities)} spots vs "
                      f"{len(accumulated_intensities)} — skipping.")

    mean_intensities = accumulated_intensities / num_phonon_configs
    print("Phonon-averaged diffraction patterns computed.")

    # ------------------------------------------------------------------
    # 4. Save outputs
    # ------------------------------------------------------------------
    plot_spots(reference_positions, reference_millers, mean_intensities,
               out_str, out_path=output_path)

    out_file = f"spots_bloch_{out_str}.npz"
    if output_path is not None:
        out_file = output_path + "/" + out_file
    save_as_npz_phonon(reference_millers, mean_intensities, out_file,
                       positions=reference_positions)

    # Also save the geometry for this image so downstream scripts can
    # reconstruct which orientation was used
    geo_file = out_file.replace("spots_bloch_", "geometry_")
    np.savez(geo_file,
             seed=seed,
             image_index=image_index,
             scan_angle_deg=scan_angle,
             start_angle_deg=start_angle,
             end_angle_deg=end_angle,
             delta_deg=delta,
             axis=axis,
             initial_R=initial_R,
             total_R=total_R,
             thickness_nm=thickness_nm,
             thickness_nm_corrected=thickness_nm_corr)
    print(f"Geometry saved to {geo_file}")

    return True


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
        kx, ky, _ = pos
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

    ax.scatter(kxs_other, kys_other, marker='o', s=0.5, c='#BEBEBE',
               linewidths=0)

    ax.scatter(kxs, kys, marker='o', s=1 + 2 * abs(np.log(ints_all)),
               c='C3', linewidths=0, cmap='jet')

    out_file = f'image_{out_str}.png'
    if out_path is not None:
        out_file = out_path + '/' + out_file
    plt.savefig(out_file, dpi=400)
    plt.close(fig)


def save_as_npz_phonon(miller_indices, intensities, filename,
                       positions=None):

    h_all, k_all, l_all, millers_all = [], [], [], []
    for miller, intensity in zip(miller_indices, intensities):
        h, k, ll = int(miller[0]), int(miller[1]), int(miller[2])
        h_all.append(h)
        k_all.append(k)
        l_all.append(ll)
        millers_all.append([h, k, ll])
    save_kwargs = dict(
        h=h_all, k=k_all, l=l_all,
        miller=millers_all,
        intensity=list(intensities),
    )
    if positions is not None:
        save_kwargs['positions'] = np.array(positions)
    np.savez(filename, **save_kwargs)


def _parse_args():

    p = argparse.ArgumentParser(
            description="Bloch wave simulation — synthetic rotation scan",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

    p.add_argument("cif_file", help="Path to the CIF structure file")

    p.add_argument("--image_index", type=int, default=0,
                   help="Zero-based image number to simulate")

    p.add_argument("--start_angle", type=float, default=-15.0,
                   help="Start of rotation scan (degrees)")

    p.add_argument("--end_angle", type=float, default=15.0,
                   help="End of rotation scan (degrees)")

    p.add_argument("--delta", type=float, default=0.01,
                   help="Angular step size (degrees)")

    p.add_argument("--seed", type=int, default=0,
                   help="Master seed for orientation / axis / thickness")

    p.add_argument("--k_max", type=float, default=5)
    p.add_argument("--sg_max", type=float, default=0.1)
    p.add_argument("--output_path", type=str, default=None)
    p.add_argument("--num_phonon_configs", type=int, default=10)
    p.add_argument("--phonon_sigmas", type=float, default=0.0)
    p.add_argument("--phonon_seed", type=int, default=42)
    p.add_argument("--show", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    main()
