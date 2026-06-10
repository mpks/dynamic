#!/usr/bin/env python3
"""
Bloch wave diffraction simulation with synthetic rotation scan.

This script:
  - Draws a random starting crystal orientation (orientation_seed)
  - Draws a random crystal thickness in [20, 200] nm (thickness_seed)
  - Rotates the crystal around the fixed y-axis
    (perpendicular to beam)
  - For each output image, integrates n_substeps diffraction
    patterns evenly spaced within one delta-wide angular window
  - The substep integration is parallelised across n_workers
    processes using ProcessPoolExecutor
  - Saves the summed result as a miniCBF file (DIALS-compatible)
  - Optionally saves rocking curves for specified Miller indices

Usage example
-------------
python bloch_synthetic.py my_structure.cif \\
    --image_index  0 \\
    --start_angle -15.0 \\
    --end_angle    15.0 \\
    --delta        1.0 \\
    --n_substeps   100 \\
    --n_workers    5 \\
    --orientation_seed 42 \\
    --thickness_seed   7 \\
    --rocking_hkl  "0,0,2" "1,0,1"
"""

import argparse
import datetime
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import abtem
from abtem.bloch import BlochWaves, StructureFactor
import ase
import ase.io
import fabio
import fabio.cbfimage
import matplotlib.pyplot as plt
import numba
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

numba.set_num_threads(2)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROTATION_AXIS = np.array([0.0, 1.0, 0.0])  # y-axis, fixed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    rocking_hkl = _parse_hkl_list(args.rocking_hkl)
    simulate_image(
        cif_file=args.cif_file,
        image_index=args.image_index,
        start_angle=args.start_angle,
        end_angle=args.end_angle,
        delta=args.delta,
        n_substeps=args.n_substeps,
        n_workers=args.n_workers,
        orientation_seed=args.orientation_seed,
        thickness_seed=args.thickness_seed,
        thickness_nm=args.thickness_nm,
        k_max=args.k_max,
        sg_max=args.sg_max,
        num_phonon_configs=args.num_phonon_configs,
        phonon_sigmas=args.phonon_sigmas,
        phonon_seed=args.phonon_seed,
        voltage_kV=args.voltage_kV,
        distance_mm=args.distance_mm,
        pixel_size_um=args.pixel_size_um,
        npx=args.npx,
        npy=args.npy,
        psf_sigma=args.psf_sigma,
        scale=args.scale,
        readout_noise=args.readout_noise,
        noise_seed=args.noise_seed,
        output_path=args.output_path,
        rocking_hkl=rocking_hkl,
        rocking_hkl_file=args.rocking_hkl_file,
        show=args.show,
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def random_rotation_matrix(rng):
    """
    Uniformly distributed random 3-D rotation matrix.
    Uses the Haar-measure algorithm (Shoemake 1992)
    via quaternions.
    """
    u1, u2, u3 = rng.uniform(0, 1, 3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ])
    return Rotation.from_quat(q).as_matrix()


def rotation_about_y(angle_deg):
    """Rotation matrix for angle_deg degrees around the y-axis."""
    rotvec = np.deg2rad(angle_deg) * ROTATION_AXIS
    return Rotation.from_rotvec(rotvec).as_matrix()


def build_crystal_geometry(orientation_seed, thickness_seed):
    """
    Return the fixed experiment geometry from two seeds.

    Parameters
    ----------
    orientation_seed : int
        Seeds the random starting crystal orientation.
    thickness_seed : int
        Seeds the thickness draw from U[20, 200] nm.

    Returns
    -------
    initial_R : ndarray (3, 3)
    thickness_nm : float
    """
    rng_or = np.random.default_rng(orientation_seed)
    initial_R = random_rotation_matrix(rng_or)
    rng_th = np.random.default_rng(thickness_seed)
    thickness_nm = float(rng_th.uniform(20.0, 200.0))
    return initial_R, thickness_nm


def substep_angles(image_index, start_angle, delta, n_substeps):
    """
    Return the n_substeps angles integrated for image_index.

    Substeps are centred within n_substeps equal sub-intervals
    of the window [start + i*delta, start + (i+1)*delta).
    """
    window_start = start_angle + image_index * delta
    step = delta / n_substeps
    offsets = (np.arange(n_substeps) + 0.5) * step
    return window_start + offsets


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def electron_wavelength_A(voltage_kV):
    """Relativistic de Broglie wavelength in Angstrom."""
    mass_e_eV = 510998.9461
    h_eVs = 4.135667696e-15
    c_ms = 299792458.0
    energy_eV = voltage_kV * 1000.0
    numerator = h_eVs * c_ms
    under_root = energy_eV**2 + 2 * mass_e_eV * energy_eV
    wavelength_m = numerator / np.sqrt(under_root)
    return wavelength_m * 1e10


def thickness_correction(thickness_nm, scan_angle_deg):
    """
    Path-length correction for a tilted slab.
    At scan_angle=0 the correction is 1.
    """
    tilt_rad = np.deg2rad(scan_angle_deg)
    return thickness_nm / np.cos(tilt_rad)


# ---------------------------------------------------------------------------
# Bloch wave calculation for a single orientation
# ---------------------------------------------------------------------------

def bloch_diffraction(
    atoms,
    thickness_A,
    k_max,
    sg_max,
    voltage_kV,
    num_phonon_configs,
    phonon_sigmas,
    phonon_seed,
):
    """
    Run a Bloch wave simulation for one orientation, averaging
    over frozen phonon configurations.

    Returns
    -------
    positions : ndarray (N, 3)  kx, ky, kz in Å⁻¹
    miller_indices : list of (h, k, l)
    mean_intensities : ndarray (N,)
    """
    frozen_phonons = abtem.FrozenPhonons(
        atoms,
        num_configs=num_phonon_configs,
        sigmas=phonon_sigmas,
        seed=phonon_seed,
    )

    accumulated = None
    positions = None
    millers = None

    for cfg_idx, atoms_cfg in enumerate(frozen_phonons):
        sf = StructureFactor(
            atoms_cfg,
            k_max=k_max,
            parametrization="lobato",
        )
        bw = BlochWaves(
            structure_factor=sf,
            energy=voltage_kV * 1e3,
            sg_max=sg_max,
        )
        if cfg_idx == 0:
            print(f"    Beams: {len(bw)}")

        patterns = bw.calculate_diffraction_patterns(
            [thickness_A]
        )
        patterns = patterns.compute()
        spots = patterns[0].remove_low_intensity(1e-20)
        ints = np.array(spots.intensities)

        if accumulated is None:
            accumulated = ints.copy()
            positions = spots.positions
            millers = spots.miller_indices
        elif len(ints) == len(accumulated):
            accumulated += ints
        else:
            print(
                f"    WARNING: phonon config {cfg_idx} has "
                f"{len(ints)} spots vs "
                f"{len(accumulated)} — skipping."
            )

    mean_intensities = accumulated / num_phonon_configs
    return positions, millers, mean_intensities


# ---------------------------------------------------------------------------
# Spot → pixel image
# ---------------------------------------------------------------------------

def spots_to_image(
    positions,
    intensities,
    npx,
    npy,
    wavelength_A,
    distance_mm,
    pixel_size_mm,
    psf_sigma_px,
):
    """
    Project reciprocal-space spots onto a flat detector.

    Uses the gnomonic (central) projection:
        kz  = sqrt(k0^2 - kx^2 - ky^2),  k0 = 1/lambda
        x   = (kx / kz) * L
        y   = -(ky / kz) * L

    This keeps reciprocal lattice rows straight on the detector,
    unlike the small-angle approximation (x = kx * lambda * L)
    which curves rows at larger scattering angles.

    Intensities are used as-is (raw). Normalisation to counts
    happens once in simulate_image after all substeps are summed.

    Parameters
    ----------
    positions : ndarray (N, 3)  kx, ky, kz in Å⁻¹
    intensities : ndarray (N,)  raw Bloch wave intensities

    Returns
    -------
    image : ndarray (npy, npx), float64
    """
    image = np.zeros((npy, npx), dtype=np.float64)
    cx = npx / 2.0
    cy = npy / 2.0
    # Correct forward projection: the scattered beam direction
    # is s0 + rlp = (0, 0, k0) + (kx, ky, kz_lattice).
    # The detector displacement is therefore:
    #
    #   dx =  kx / (k0 + kz_lattice) * L
    #   dy = -ky / (k0 + kz_lattice) * L
    #
    # This is the exact inverse of what DIALS does when it
    # back-projects a pixel to reciprocal space via s1 - s0.
    # The previous code used sqrt(k0^2 - kx^2 - ky^2) as the
    # denominator, which ignores the actual kz of the lattice
    # point and places spots on the wrong pixels.
    k0 = 1.0 / wavelength_A

    for pos, cnt in zip(positions, intensities):
        kx, ky, kz_lattice = pos[0], pos[1], pos[2]
        kz_beam = k0 + kz_lattice
        if kz_beam <= 0:
            continue
        dx = (kx / kz_beam) * distance_mm
        dy = -(ky / kz_beam) * distance_mm
        px_x = cx + dx / pixel_size_mm
        px_y = cy + dy / pixel_size_mm
        ix = int(round(px_x))
        iy = int(round(px_y))
        if 0 <= ix < npx and 0 <= iy < npy:
            image[iy, ix] += cnt

    if psf_sigma_px > 0:
        image = gaussian_filter(image, sigma=psf_sigma_px)

    return image


# ---------------------------------------------------------------------------
# Worker function — must be top-level for pickling
# ---------------------------------------------------------------------------

def _run_substep_batch(
    angle_batch,
    cif_file,
    initial_R,
    thickness_nm,
    k_max,
    sg_max,
    voltage_kV,
    num_phonon_configs,
    phonon_sigmas,
    phonon_seed,
    npx,
    npy,
    distance_mm,
    pixel_size_mm,
    psf_sigma,
    rocking_hkl,
):
    """
    Process a batch of substep angles in a single worker process.

    Called by ProcessPoolExecutor — must be a module-level function
    so it can be pickled.

    Returns
    -------
    partial_image : ndarray (npy, npx), float64
        Sum of projected images for this batch.
    rocking_partial : dict  hkl -> list of float
        Per-substep intensities for each requested Miller index.
    last_positions : ndarray
    last_millers : list
    last_mean_ints : ndarray
    """
    abtem.config.set({"mkl.threads": 2})
    abtem.config.set({"fftw.threads": 2})

    wavelength_A = electron_wavelength_A(voltage_kV)
    atoms_base = ase.io.read(cif_file)

    partial_image = np.zeros((npy, npx), dtype=np.float64)
    rocking_partial = {hkl: [] for hkl in rocking_hkl}

    last_positions = None
    last_millers = None
    last_mean_ints = None

    for angle in angle_batch:
        thick_corr = thickness_correction(thickness_nm, angle)
        thickness_A = thick_corr * 10.0

        scan_R = rotation_about_y(angle)
        total_R = scan_R @ initial_R

        atoms = atoms_base.copy()
        cell = atoms.get_cell()
        new_cell = (total_R @ cell.array.T).T
        atoms.set_cell(new_cell, scale_atoms=False)
        atoms.wrap()

        positions, millers, mean_ints = bloch_diffraction(
            atoms=atoms,
            thickness_A=thickness_A,
            k_max=k_max,
            sg_max=sg_max,
            voltage_kV=voltage_kV,
            num_phonon_configs=num_phonon_configs,
            phonon_sigmas=phonon_sigmas,
            phonon_seed=phonon_seed,
        )

        sub_image = spots_to_image(
            positions=positions,
            intensities=mean_ints,
            npx=npx,
            npy=npy,
            wavelength_A=wavelength_A,
            distance_mm=distance_mm,
            pixel_size_mm=pixel_size_mm,
            psf_sigma_px=psf_sigma,
        )
        partial_image += sub_image

        for hkl in rocking_hkl:
            idx = find_spot_index(millers, hkl)
            value = float(mean_ints[idx]) if idx is not None else 0.0
            rocking_partial[hkl].append(value)

        last_positions = positions
        last_millers = millers
        last_mean_ints = mean_ints

    return (
        partial_image,
        rocking_partial,
        last_positions,
        last_millers,
        last_mean_ints,
    )


# ---------------------------------------------------------------------------
# CBF output
# ---------------------------------------------------------------------------

def build_minicbf_header(
    wavelength_A,
    distance_mm,
    pixel_size_mm,
    npx,
    npy,
    start_angle_deg,
    angle_increment_deg,
    image_index,
):
    """
    Build _array_data.header_contents matching the working
    ELDICO ED-1 / Eiger Quadro format exactly.

    Convention PILATUS_1.2 + 'Eiger Quadro' in the detector
    line routes dxtbx to FormatCBFMiniEigerQuadroED1, which
    sets up an unpolarised electron beam (Probe.electron) and
    disables parallax correction (SimplePxMmStrategy).

    The wavelength check in FormatCBFMiniEigerQuadroED1
    requires round(wl, 3) == 0.029 (~160 kV electrons).
    At other voltages dxtbx falls back to FormatCBFMiniEiger.
    """
    px_m = pixel_size_mm * 1e-3
    d_m = distance_mm * 1e-3
    cx = npx / 2.0
    cy = npy / 2.0
    ts = datetime.datetime.utcnow().strftime(
        "%Y-%m-%dT%H:%M:%S.000"
    )
    lines = [
        "# Detector: Eiger Quadro S/N E-01-0000",
        f"# {ts}",
        f"# Pixel_size {px_m:.2e} m x {px_m:.2e} m",
        "# Silicon sensor, thickness 0.000450 m",
        "# Exposure_time 1.0000000 s",
        "# Exposure_period 1.0000001 s",
        "# Tau = 0 s",
        "# Count_cutoff 16777216 counts",
        "# Threshold_setting: 21743 eV",
        "# Gain_setting: high gain (vrf = -0.150)",
        "# N_excluded_pixels = 0",
        "# Excluded_pixels: badpix_mask.tif",
        "# Flat_field: x.tif",
        "# Trim_file: x.bin",
        "# Image_path: /data/synthetic/",
        "# Retrigger_mode: 1",
        f"# Wavelength {wavelength_A:.6f} A",
        f"# Beam_xy ({cx:.2f}, {cy:.2f}) pixels",
        f"# Detector_distance {d_m:.6f} m",
        f"# Start_angle {start_angle_deg:.3f} deg.",
        f"# Angle_increment {angle_increment_deg:.3f} deg.",
        "# Detector_2theta 0.000 deg.",
        "# Alpha 0.0000 deg.",
        "# Kappa 0.0000 deg.",
        f"# Phi {start_angle_deg:.3f} deg.",
        f"# Phi_increment {angle_increment_deg:.3f} deg.",
        "# Omega 0.0000 deg.",
        "# Omega_increment 0.0000 deg.",
        "# Chi 267.000 deg.",
        "# Oscillation_axis PHI",
    ]
    return "\r\n".join(lines) + "\r\n"


def write_cbf(image_int32, header_contents, output_path):
    """
    Write a miniCBF file using PILATUS_1.2 convention so that
    dxtbx selects FormatCBFMiniEigerQuadroED1 (electron probe).
    """
    fabio_header = {
        "_audit.creation_method": (
            "Created by bloch_synthetic.py"
        ),
        "_audit_author.name": "Bloch wave simulation",
        "_array_data.header_convention": "PILATUS_1.2",
        "_array_data.header_contents": header_contents,
        "_array_data.description": "",
    }
    cbf = fabio.cbfimage.CbfImage(
        data=image_int32,
        header=fabio_header,
    )
    cbf.write(output_path)


# ---------------------------------------------------------------------------
# Rocking curve helpers
# ---------------------------------------------------------------------------

def find_spot_index(miller_indices, target_hkl):
    """
    Return the index of target_hkl in miller_indices,
    or None if absent.
    """
    th, tk, tl = target_hkl
    for idx, m in enumerate(miller_indices):
        if (int(m[0]) == th
                and int(m[1]) == tk
                and int(m[2]) == tl):
            return idx
    return None


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------

def plot_spots(
    positions, miller_indices, intensities, out_str,
    top_spots=60, out_path=None,
):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.15, 0.12, 0.80, 0.84])
    ax.set_xlabel(r'$k_{\rm x}\ (\rm \AA)$')
    ax.set_ylabel(r'$k_{\rm y}\ (\rm \AA)$')

    if len(intensities) > top_spots:
        cutoff_idx = np.argsort(intensities)[-top_spots]
        cutoff = intensities[cutoff_idx]
    else:
        cutoff = 0.0

    kxs, kys, ints_top = [], [], []
    kxs_rest, kys_rest = [], []

    for pos, miller, ints in zip(
        positions, miller_indices, intensities
    ):
        kx, ky = pos[0], pos[1]
        if ints > cutoff:
            kxs.append(kx)
            kys.append(ky)
            ints_top.append(ints)
            h = int(miller[0])
            k = int(miller[1])
            l = int(miller[2])
            label = f"{h} {k} {l}"
            ax.text(
                kx, ky + 0.10, label,
                color='blue', va='top', ha='center',
                fontsize=4,
                bbox=dict(
                    facecolor='white', alpha=0.5,
                    edgecolor='none',
                    boxstyle='square, pad=0.3',
                ),
            )
        else:
            kxs_rest.append(kx)
            kys_rest.append(ky)

    ax.scatter(
        kxs_rest, kys_rest,
        marker='o', s=0.5, c='#BEBEBE', linewidths=0,
    )
    sizes = 1 + 2 * np.abs(np.log(ints_top))
    ax.scatter(
        kxs, kys,
        marker='o', s=sizes, c='C3', linewidths=0,
    )

    fname = f"image_{out_str}.png"
    if out_path is not None:
        fname = os.path.join(out_path, fname)
    plt.savefig(fname, dpi=400)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Intensity statistics report
# ---------------------------------------------------------------------------

def print_intensity_stats(intensities, scale, readout_noise):
    """Print a summary of spot intensities and SNR estimates."""
    i_max = intensities.max()
    i_min = intensities.min()
    i_mean = intensities.mean()
    i_median = np.median(intensities)
    i_sorted = np.sort(intensities)[::-1]
    nonzero = intensities[intensities > 0]
    if len(nonzero) > 1:
        dyn_range = nonzero.max() / nonzero.min()
    else:
        dyn_range = 1.0

    n = len(intensities)
    print(f"\n--- Spot intensity statistics ({n} spots) ---")
    print(f"  max           : {i_max:.6e}")
    print(f"  min           : {i_min:.6e}")
    print(f"  mean          : {i_mean:.6e}")
    print(f"  median        : {i_median:.6e}")
    print(f"  dynamic range : {dyn_range:.2e}  (max/min nonzero)")
    top5 = [f"{v:.3e}" for v in i_sorted[:5]]
    bot5 = [f"{v:.3e}" for v in i_sorted[-5:]]
    print(f"  top 5         : {top5}")
    print(f"  bottom 5      : {bot5}")

    snr_threshold = 3.0 * readout_noise
    peak_counts = intensities / i_max * scale
    n_visible = int(np.sum(peak_counts > snr_threshold))
    print(f"\n  scale={scale:.0f}, readout_noise={readout_noise:.1f}")
    print(f"  SNR>3 threshold : {snr_threshold:.1f} counts")
    print(f"  Spots above SNR>3 : {n_visible} / {n}")
    if n_visible == 0:
        suggested = snr_threshold * 10.0 / (i_median / i_max)
        print(f"  WARNING: no spots visible at current scale!")
        print(
            f"  -> Try --scale {suggested:.0f}"
            f"  (median spot at SNR~10)"
        )
    elif n_visible < 10:
        print(
            "  WARNING: very few spots visible"
            " — consider raising --scale"
        )
    print()


# ---------------------------------------------------------------------------
# Detector plotting helpers
# ---------------------------------------------------------------------------

def _project_spots_to_pixels(
    positions, wavelength_A, distance_mm,
    pixel_size_mm, beam_centre_px, npx, npy,
):
    """Project reciprocal-space positions to detector pixels."""
    k0 = 1.0 / wavelength_A
    cx, cy = beam_centre_px
    px_coords = []
    for pos in positions:
        kx, ky, kz_lattice = pos[0], pos[1], pos[2]
        kz_beam = k0 + kz_lattice
        if kz_beam <= 0:
            px_coords.append(None)
            continue
        dx = (kx / kz_beam) * distance_mm
        dy = -(ky / kz_beam) * distance_mm
        px_x = cx + dx / pixel_size_mm
        px_y = cy + dy / pixel_size_mm
        px_coords.append((px_x, px_y))
    return px_coords


def plot_detector_image(
    image, positions, miller_indices, intensities,
    npx, npy, wavelength_A, distance_mm,
    pixel_size_mm, beam_centre_px, out_path,
    top_spots=20,
):
    """
    Save a raster PNG of the detector image matching the
    dials.image_viewer look (binary cmap, log scale) with
    red Miller index labels for the top_spots brightest spots.
    """
    px_coords = _project_spots_to_pixels(
        positions, wavelength_A, distance_mm,
        pixel_size_mm, beam_centre_px, npx, npy,
    )
    i_sorted = np.argsort(intensities)[::-1]
    top_idx = set(i_sorted[:top_spots].tolist())

    dpi = 100
    fig, ax = plt.subplots(
        figsize=(npx / dpi, npy / dpi), dpi=dpi
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    display = np.log1p(image.astype(np.float32))
    ax.imshow(
        display,
        cmap='binary',
        origin='upper',
        extent=[0, npx, npy, 0],
        interpolation='nearest',
        aspect='equal',
    )
    for i, (pos, miller, ints) in enumerate(
        zip(px_coords, miller_indices, intensities)
    ):
        if pos is None:
            continue
        px_x, px_y = pos
        if not (0 <= px_x < npx and 0 <= px_y < npy):
            continue
        if i in top_idx:
            h, k, l = int(miller[0]), int(miller[1]), int(miller[2])
            ax.plot(px_x, px_y, 'o',
                    color='red', markersize=4,
                    markerfacecolor='none',
                    markeredgewidth=1)
            ax.text(
                px_x + 3, px_y - 3, f"{h},{k},{l}",
                color='red', fontsize=4,
                va='bottom', ha='left',
            )
        else:
            ax.plot(px_x, px_y, '.', color='red',
                    markersize=1, alpha=0.5)
    ax.set_xlim(0, npx)
    ax.set_ylim(npy, 0)
    ax.axis('off')
    plt.savefig(
        out_path, dpi=dpi * 2,
        bbox_inches='tight', pad_inches=0,
        facecolor='white',
    )
    plt.close(fig)
    print(f"Detector image saved: {out_path}")


def plot_detector_labels(
    positions, miller_indices, intensities,
    npx, npy, wavelength_A, distance_mm,
    pixel_size_mm, beam_centre_px, out_path,
    top_spots=80,
):
    """
    Save a PNG showing all spot positions with Miller index
    labels on a dark background — for comparing abTEM vs
    DIALS indexing.
    """
    px_coords = _project_spots_to_pixels(
        positions, wavelength_A, distance_mm,
        pixel_size_mm, beam_centre_px, npx, npy,
    )
    i_sorted = np.argsort(intensities)[::-1]
    top_idx = set(i_sorted[:top_spots].tolist())

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, npx)
    ax.set_ylim(npy, 0)
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#1a1a1a")
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    for i, (pos, miller, ints) in enumerate(
        zip(px_coords, miller_indices, intensities)
    ):
        if pos is None:
            continue
        px_x, px_y = pos
        if not (0 <= px_x < npx and 0 <= px_y < npy):
            continue
        if i in top_idx:
            h, k, l = int(miller[0]), int(miller[1]), int(miller[2])
            size = 3 + 6 * (ints / intensities.max())
            ax.plot(px_x, px_y, 'o',
                    color='red', markersize=size)
            ax.text(
                px_x + 4, px_y - 4, f"{h},{k},{l}",
                color='yellow', fontsize=5,
                va='bottom', ha='left',
            )
        else:
            ax.plot(px_x, px_y, '.', color='#666666',
                    markersize=1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Detector labels saved: {out_path}")


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def simulate_image(
    cif_file,
    image_index,
    start_angle=-15.0,
    end_angle=15.0,
    delta=1.0,
    n_substeps=100,
    n_workers=1,
    orientation_seed=42,
    thickness_seed=0,
    thickness_nm=None,
    k_max=5.0,
    sg_max=0.1,
    num_phonon_configs=1,
    phonon_sigmas=0.0,
    phonon_seed=42,
    voltage_kV=200.0,
    distance_mm=578.3,
    pixel_size_um=75.0,
    npx=512,
    npy=512,
    psf_sigma=1.0,
    scale=10000.0,
    readout_noise=1.0,
    noise_seed=0,
    output_path=None,
    rocking_hkl=None,
    rocking_hkl_file=None,
    show=False,
):
    """
    Simulate one integrated diffraction image and save as CBF.

    Each output image covers one angular window of width `delta`
    degrees, integrated over `n_substeps` equally-spaced
    sub-angles.  The substep work is split across `n_workers`
    parallel processes; each worker handles a contiguous slice
    of the substep angle array and returns a partial pixel image
    which is summed in the main process.

    Parameters
    ----------
    cif_file : str
        Path to the CIF structure file.
    image_index : int
        Zero-based image number.
    start_angle, end_angle : float
        Full scan range in degrees.
    delta : float
        Angular width of one image in degrees.
    n_substeps : int
        Number of sub-angles integrated per image.
    n_workers : int
        Number of parallel worker processes.  Each worker gets
        roughly n_substeps // n_workers substeps.
    orientation_seed : int
        Seeds the random starting crystal orientation.
    thickness_seed : int
        Seeds the crystal thickness draw.
    rocking_hkl : list of (h, k, l) tuples or None
        Miller indices for which to save rocking curves.
    """
    # ------------------------------------------------------------------
    # 1. Geometry
    # ------------------------------------------------------------------
    initial_R, thickness_nm_seed = build_crystal_geometry(
        orientation_seed, thickness_seed
    )
    # If thickness_nm is explicitly provided, use it and ignore
    # the thickness seed. If not, use the seeded random value.
    if thickness_nm is None:
        thickness_nm = thickness_nm_seed
    else:
        print(
            f"[Synthetic] Using explicit thickness: "
            f"{thickness_nm:.1f} nm "
            f"(thickness_seed ignored)"
        )

    n_images = int(round((end_angle - start_angle) / delta))
    if image_index < 0 or image_index >= n_images:
        raise ValueError(
            f"image_index {image_index} out of range "
            f"[0, {n_images - 1}]."
        )

    angles = substep_angles(
        image_index, start_angle, delta, n_substeps
    )
    window_centre = start_angle + (image_index + 0.5) * delta
    window_start = start_angle + image_index * delta

    print(f"[Synthetic] orientation_seed={orientation_seed}, "
          f"thickness_seed={thickness_seed}")
    print(f"[Synthetic] Thickness: {thickness_nm:.1f} nm")
    print(
        f"[Synthetic] Image {image_index}: "
        f"integrating {n_substeps} substeps "
        f"over [{angles[0]:.4f}, {angles[-1]:.4f}] deg "
        f"using {n_workers} worker(s)"
    )

    wavelength_A = electron_wavelength_A(voltage_kV)
    pixel_size_mm = pixel_size_um * 1e-3

    if rocking_hkl_file is not None:
        file_hkl = _read_hkl_file(rocking_hkl_file)
        print(
            f"Loaded {len(file_hkl)} Miller indices "
            f"from {rocking_hkl_file}"
        )
        if rocking_hkl is None:
            rocking_hkl = file_hkl
        else:
            rocking_hkl = list(rocking_hkl) + file_hkl
    if rocking_hkl is None:
        rocking_hkl = []

    # ------------------------------------------------------------------
    # 2. Split substep angles into n_workers batches
    # ------------------------------------------------------------------
    batches = np.array_split(angles, n_workers)
    # Drop any empty batches (when n_workers > n_substeps)
    batches = [b for b in batches if len(b) > 0]

    # Common kwargs passed to every worker
    worker_kwargs = dict(
        cif_file=cif_file,
        initial_R=initial_R,
        thickness_nm=thickness_nm,
        k_max=k_max,
        sg_max=sg_max,
        voltage_kV=voltage_kV,
        num_phonon_configs=num_phonon_configs,
        phonon_sigmas=phonon_sigmas,
        phonon_seed=phonon_seed,
        npx=npx,
        npy=npy,
        distance_mm=distance_mm,
        pixel_size_mm=pixel_size_mm,
        psf_sigma=psf_sigma,
        rocking_hkl=rocking_hkl,
    )

    # ------------------------------------------------------------------
    # 3. Dispatch workers and collect results
    # ------------------------------------------------------------------
    integrated_image = np.zeros((npy, npx), dtype=np.float64)
    rocking_curves = {hkl: [] for hkl in rocking_hkl}
    last_positions = None
    last_millers = None
    last_mean_ints = None

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _run_substep_batch,
                angle_batch=batch,
                **worker_kwargs,
            ): i
            for i, batch in enumerate(batches)
        }

        # Collect in completion order; reassemble rocking curves
        # in batch order afterwards
        batch_results = {}
        for future in as_completed(futures):
            batch_idx = futures[future]
            result = future.result()
            batch_results[batch_idx] = result
            print(f"  Batch {batch_idx + 1}/{len(batches)} done")

    # Sum partial images; rocking curves need to be in angle order
    for batch_idx in sorted(batch_results.keys()):
        (
            partial_image,
            rocking_partial,
            b_positions,
            b_millers,
            b_mean_ints,
        ) = batch_results[batch_idx]

        integrated_image += partial_image

        for hkl in rocking_hkl:
            rocking_curves[hkl].extend(rocking_partial[hkl])

        # Keep the result from the last batch (highest angles)
        # as representative for plots and stats
        last_positions = b_positions
        last_millers = b_millers
        last_mean_ints = b_mean_ints

    # ------------------------------------------------------------------
    # 4. Scale then add noise
    # ------------------------------------------------------------------
    # Normalise once so the strongest pixel across all substeps
    # gets `scale` counts.  Preserves relative intensities.
    img_max = integrated_image.max()
    if img_max > 0:
        integrated_image = integrated_image / img_max * scale

    rng = np.random.default_rng(noise_seed)
    clipped = np.maximum(integrated_image, 0)
    noisy = rng.poisson(clipped).astype(np.float64)
    noisy += rng.normal(0, readout_noise, noisy.shape)
    noisy = np.clip(np.round(noisy), 0, None).astype(np.int32)

    # ------------------------------------------------------------------
    # 5. Print intensity statistics (last batch, last substep)
    # ------------------------------------------------------------------
    print_intensity_stats(last_mean_ints, scale, readout_noise)

    # ------------------------------------------------------------------
    # 6. Diagnostic plot
    # ------------------------------------------------------------------
    out_tag = f"{image_index:06d}_kmax_{k_max:03.0f}"
    plot_spots(
        last_positions, last_millers, last_mean_ints,
        out_tag, out_path=output_path,
    )

    if show:
        fig = plt.figure(figsize=(3.375, 3.0))
        ax1 = fig.add_axes([0.13, 0.14, 0.84, 0.82])
        atoms_show = ase.io.read(cif_file)
        scan_R_c = rotation_about_y(window_centre)
        total_R_c = scan_R_c @ initial_R
        cell = atoms_show.get_cell()
        new_cell = (total_R_c @ cell.array.T).T
        atoms_show.set_cell(new_cell, scale_atoms=False)
        atoms_show.wrap()
        abtem.show_atoms(atoms_show, ax=ax1, plane="xz")
        plt.savefig("system_bloch.png", dpi=400)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 7. Save CBF
    # ------------------------------------------------------------------
    cbf_name = f"image_{out_tag}.cbf"
    if output_path is not None:
        cbf_name = os.path.join(output_path, cbf_name)

    header = build_minicbf_header(
        wavelength_A=wavelength_A,
        distance_mm=distance_mm,
        pixel_size_mm=pixel_size_mm,
        npx=npx,
        npy=npy,
        start_angle_deg=window_start,
        angle_increment_deg=delta,
        image_index=image_index,
    )
    write_cbf(noisy, header, cbf_name)
    print(f"CBF written: {cbf_name}")

    # ------------------------------------------------------------------
    # 7b. Detector image and label plots
    # ------------------------------------------------------------------
    beam_centre_px = (npx / 2.0, npy / 2.0)
    raster_name = f"detector_image_{out_tag}.png"
    label_name = f"detector_labels_{out_tag}.png"
    if output_path is not None:
        raster_name = os.path.join(output_path, raster_name)
        label_name = os.path.join(output_path, label_name)
    plot_detector_image(
        image=noisy,
        positions=last_positions,
        miller_indices=last_millers,
        intensities=last_mean_ints,
        npx=npx,
        npy=npy,
        wavelength_A=wavelength_A,
        distance_mm=distance_mm,
        pixel_size_mm=pixel_size_mm,
        beam_centre_px=beam_centre_px,
        out_path=raster_name,
    )
    plot_detector_labels(
        positions=last_positions,
        miller_indices=last_millers,
        intensities=last_mean_ints,
        npx=npx,
        npy=npy,
        wavelength_A=wavelength_A,
        distance_mm=distance_mm,
        pixel_size_mm=pixel_size_mm,
        beam_centre_px=beam_centre_px,
        out_path=label_name,
    )

    # ------------------------------------------------------------------
    # 8. Save geometry NPZ
    # ------------------------------------------------------------------
    scan_R_centre = rotation_about_y(window_centre)
    total_R_centre = scan_R_centre @ initial_R
    thick_corr_centre = thickness_correction(
        thickness_nm, window_centre
    )

    geo_name = f"geometry_{out_tag}.npz"
    if output_path is not None:
        geo_name = os.path.join(output_path, geo_name)

    np.savez(
        geo_name,
        orientation_seed=orientation_seed,
        thickness_seed=thickness_seed,
        image_index=image_index,
        window_start_deg=window_start,
        window_centre_deg=window_centre,
        delta_deg=delta,
        n_substeps=n_substeps,
        n_workers=n_workers,
        axis=ROTATION_AXIS,
        initial_R=initial_R,
        total_R_centre=total_R_centre,
        thickness_nm=thickness_nm,
        thickness_nm_corrected=thick_corr_centre,
        substep_angles=angles,
    )
    print(f"Geometry saved: {geo_name}")

    # ------------------------------------------------------------------
    # 9. Save rocking curves NPZ
    # ------------------------------------------------------------------
    if rocking_curves:
        rc_name = f"rocking_{out_tag}.npz"
        if output_path is not None:
            rc_name = os.path.join(output_path, rc_name)
        hkl_list = list(rocking_curves.keys())
        hkl_arr = np.array(hkl_list, dtype=np.int32)
        rc_save = {
            "angles": angles,
            "hkl_indices": hkl_arr,
            "image_index": image_index,
        }
        for hkl, curve in rocking_curves.items():
            key = f"hkl_{hkl[0]}_{hkl[1]}_{hkl[2]}"
            rc_save[key] = np.array(curve)
        np.savez(rc_name, **rc_save)
        print(
            f"Rocking curves saved: {rc_name} "
            f"({len(hkl_list)} reflections)"
        )

    return True


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _read_hkl_file(path):
    """
    Read Miller indices from a text file, one per line.
    Blank lines and lines starting with # are ignored.
    Accepted formats:
        h k l   (space-separated)
        h,k,l   (comma-separated)
    Returns a list of (h, k, l) tuples.
    """
    result = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.replace(",", " ")
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Cannot parse HKL line: {line!r}"
                )
            result.append(tuple(int(x) for x in parts))
    return result


def _parse_hkl_list(hkl_strings):
    """Parse 'h,k,l' strings into (h, k, l) tuples."""
    if not hkl_strings:
        return []
    result = []
    for s in hkl_strings:
        parts = s.split(",")
        result.append(tuple(int(x) for x in parts))
    return result


def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Bloch wave simulation — synthetic rotation scan"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("cif_file",
                   help="Path to the CIF structure file")
    p.add_argument("--image_index", type=int, default=0,
                   help="Zero-based image number to simulate")
    p.add_argument("--start_angle", type=float, default=-15.0,
                   help="Start of rotation scan (degrees)")
    p.add_argument("--end_angle", type=float, default=15.0,
                   help="End of rotation scan (degrees)")
    p.add_argument("--delta", type=float, default=1.0,
                   help="Angular width of one image (degrees)")
    p.add_argument("--n_substeps", type=int, default=100,
                   help="Sub-angles integrated per image")
    p.add_argument("--n_workers", type=int, default=1,
                   help="Parallel worker processes for substeps")
    p.add_argument("--orientation_seed", type=int, default=42,
                   help="Seed for random starting orientation")
    p.add_argument("--thickness_seed", type=int, default=0,
                   help="Seed for crystal thickness draw")
    p.add_argument(
        "--thickness_nm", type=float, default=None,
        help=(
            "Crystal thickness in nm. "
            "If given, overrides thickness_seed."
        ),
    )
    p.add_argument("--k_max", type=float, default=5.0,
                   help="Max scattering vector (1/Å)")
    p.add_argument("--sg_max", type=float, default=0.1,
                   help="Max excitation error")
    p.add_argument("--num_phonon_configs", type=int, default=10)
    p.add_argument("--phonon_sigmas", type=float, default=0.0)
    p.add_argument("--phonon_seed", type=int, default=42)
    p.add_argument("--voltage_kV", type=float, default=200.0,
                   help="Accelerating voltage (kV)")
    p.add_argument("--distance_mm", type=float, default=578.3,
                   help="Sample-to-detector distance (mm)")
    p.add_argument("--pixel_size_um", type=float, default=75.0,
                   help="Physical pixel size (µm)")
    p.add_argument("--npx", type=int, default=512,
                   help="Detector width in pixels")
    p.add_argument("--npy", type=int, default=512,
                   help="Detector height in pixels")
    p.add_argument("--psf_sigma", type=float, default=1.0,
                   help="Gaussian PSF sigma (pixels)")
    p.add_argument("--scale", type=float, default=10000.0,
                   help="Peak counts for strongest spot")
    p.add_argument("--readout_noise", type=float, default=1.0,
                   help="Readout noise sigma (counts)")
    p.add_argument("--noise_seed", type=int, default=0)
    p.add_argument("--output_path", type=str, default=None)
    p.add_argument(
        "--rocking_hkl",
        nargs="*",
        default=[],
        metavar="H,K,L",
        help=(
            "Miller indices for rocking curves,"
            " e.g. 0,0,2 1,0,1"
        ),
    )
    p.add_argument(
        "--rocking_hkl_file",
        type=str, default=None,
        metavar="FILE",
        help=(
            "Text file with one Miller index per line "
            "(h k l or h,k,l). "
            "Combined with --rocking_hkl if both given."
        ),
    )
    p.add_argument("--show", action="store_true",
                   help="Save crystal orientation plot")
    return p.parse_args()


if __name__ == "__main__":
    main()
