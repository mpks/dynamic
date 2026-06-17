"""
export_cbf.py — render integrated diffraction images to miniCBF
files readable by DIALS.

Unlike the NPZ exporter, this module rasterises the integrated
spots onto the detector grid: each spot is placed at its
intensity-weighted centroid, the image is scaled, broadened by a
Gaussian point-spread function, and given Poisson + readout
noise.  The scale, PSF width and noise parameters live here in
CbfParams because they are specific to producing a detector
image.

The header is populated from the Detector, Beam and Scan domain
objects.  Using the PILATUS_1.2 convention together with the
"Eiger Quadro" detector name routes dxtbx to
FormatCBFMiniEigerQuadroED1, which sets up an electron probe and
the simple pixel-to-millimetre strategy.  That reader requires
round(wavelength, 3) == 0.029 (about 160 kV); at other
wavelengths dxtbx falls back to the generic Eiger format.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


# ----------------------------------------------------------------
# CBF rendering parameters
# ----------------------------------------------------------------

@dataclass(frozen=True)
class CbfParams:
    """
    Parameters for rendering an image to CBF.

    scale : float
        Counts assigned to the strongest pixel after summing.
    psf_sigma : float
        Gaussian point-spread sigma in pixels.
    readout_noise : float
        Standard deviation of additive Gaussian readout noise.
    noise_seed : int
        Seed for the Poisson + readout noise generator.
    ds_sigma : float
        Sigma (pixels) of the diffuse-scatter Gaussian blob
        centred on the direct beam (0, 0, 0).  Zero disables
        the blob.
    ds_intensity : float
        Peak counts of the diffuse-scatter blob, in the same
        units as scale (added after scaling, before noise).
    """
    scale: float = 10000.0
    psf_sigma: float = 1.0
    readout_noise: float = 1.0
    noise_seed: int = 0
    ds_sigma: float = 30.0
    ds_intensity: float = 0.0


# ----------------------------------------------------------------
# Rasterisation
# ----------------------------------------------------------------

def _add_diffuse_scatter(image, detector, params):
    """
    Add a Gaussian diffuse-scatter blob centred on the direct
    beam (the detector beam centre).  The blob has peak
    params.ds_intensity counts and width params.ds_sigma
    pixels.  Returns the image unchanged if ds_intensity <= 0.
    """
    if params.ds_intensity <= 0 or params.ds_sigma <= 0:
        return image
    npy, npx = image.shape
    cx, cy = detector.beam_centre_px
    yy, xx = np.mgrid[0:npy, 0:npx]
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    blob = params.ds_intensity * np.exp(
        -r2 / (2.0 * params.ds_sigma ** 2)
    )
    return image + blob


def rasterise_image(image_result, detector, params):
    """
    Build a noisy integer detector image from an ImageResult.

    The spots are placed at their (already integrated) centroids
    px, py with their integrated intensities, broadened by the
    PSF, scaled so the strongest pixel is params.scale counts.
    A diffuse-scatter Gaussian blob is then added at the direct
    beam, and finally Poisson and readout noise are applied.

    Returns
    -------
    ndarray (npy, npx), int32
    """
    npx = detector.npx
    npy = detector.npy
    raw = np.zeros((npy, npx), dtype=np.float64)

    px = image_result.px
    py = image_result.py
    intensities = image_result.intensities

    for i in range(len(intensities)):
        ix = int(round(px[i]))
        iy = int(round(py[i]))
        if 0 <= ix < npx and 0 <= iy < npy:
            raw[iy, ix] += intensities[i]

    if params.psf_sigma > 0:
        raw = gaussian_filter(raw, sigma=params.psf_sigma)

    raw_max = raw.max()
    if raw_max > 0:
        raw = raw / raw_max * params.scale

    # Diffuse scatter around the direct beam, added after the
    # spot scaling so its level is set independently.
    raw = _add_diffuse_scatter(raw, detector, params)

    rng = np.random.default_rng(params.noise_seed)
    clipped = np.maximum(raw, 0)
    noisy = rng.poisson(clipped).astype(np.float64)
    noisy += rng.normal(0, params.readout_noise, noisy.shape)
    noisy = np.clip(np.round(noisy), 0, None)
    return noisy.astype(np.int32)


# ----------------------------------------------------------------
# Header
# ----------------------------------------------------------------

def build_header(detector, beam, start_angle_deg,
                 angle_increment_deg):
    """
    Build the _array_data.header_contents string from the
    Detector and Beam objects, in the ELDICO ED-1 /
    Eiger Quadro miniCBF format.
    """
    px_m = detector.pixel_size_mm * 1e-3
    d_m = detector.distance_mm * 1e-3
    cx, cy = detector.beam_centre_px
    wl = beam.wavelength_A
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
        f"# Wavelength {wl:.6f} A",
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


# ----------------------------------------------------------------
# Writing
# ----------------------------------------------------------------

def write_cbf(image_int32, header_contents, output_path):
    """
    Write a miniCBF using the PILATUS_1.2 convention so that
    dxtbx selects the electron-probe Eiger format.
    """
    import fabio.cbfimage
    fabio_header = {
        "_audit.creation_method": (
            "Created by dynamic.simulation export_cbf"
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


def cbf_filename(out_dir, tag, image_index):
    """Per-image CBF path with 4-digit zero-padded index."""
    name = f"image_{tag}_{image_index:04d}.cbf"
    return os.path.join(out_dir, name)


def save_image_cbf(image_result, detector, beam, scan,
                   params, out_dir, tag):
    """
    Render one ImageResult to a CBF file.

    The image window starts at centre - delta/2 and has width
    delta (taken from the Scan), matching the integration
    window used by the engine.
    """
    os.makedirs(out_dir, exist_ok=True)

    delta = scan.delta_deg
    start_angle = image_result.angle_centre_deg - delta / 2.0

    img = rasterise_image(image_result, detector, params)
    header = build_header(detector, beam, start_angle, delta)

    path = cbf_filename(out_dir, tag, image_result.image_index)
    write_cbf(img, header, path)
    return path


def save_images_cbf(images, detector, beam, scan,
                    params, out_dir, tag):
    """Render every ImageResult to CBF; return the paths."""
    paths = []
    for image in images:
        path = save_image_cbf(image, detector, beam, scan,
                              params, out_dir, tag)
        paths.append(path)
        print(
            f"  wrote {os.path.basename(path)}",
            flush=True,
        )
    return paths
