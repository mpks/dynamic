"""
projection.py — project reciprocal-space diffraction spots onto
the flat detector, in pixel coordinates.

This module is pure geometry: it converts a spot's lab-frame
reciprocal-space vector (kx, ky, kz) into a detector pixel
position (px, py) via the gnomonic projection used throughout
the pipeline,

    kz_beam = k0 + kz          (k0 = 1 / wavelength)
    dx = (kx / kz_beam) * distance_mm
    dy = -(ky / kz_beam) * distance_mm
    px = beam_centre_x + dx / pixel_size_mm
    py = beam_centre_y + dy / pixel_size_mm

It carries no intensity logic: integration, centroid averaging
and filtering happen in the engine.  The per-spot projection is
called once per substep, then the engine accumulates the
intensity-weighted px and py.
"""

from __future__ import annotations

import numpy as np


def project_position(kx, ky, kz, k0, distance_mm,
                     pixel_size_mm, beam_centre_px):
    """
    Project a single reciprocal-space vector onto the detector.

    Parameters
    ----------
    kx, ky, kz : float
        Lab-frame reciprocal-space components (A^-1).
    k0 : float
        Incident wavevector magnitude, 1 / wavelength (A^-1).
    distance_mm : float
    pixel_size_mm : float
    beam_centre_px : tuple (cx, cy)

    Returns
    -------
    (px, py) : tuple of float, or None
        Pixel coordinates, or None if the spot is behind the
        sample (kz_beam <= 0) and cannot be projected.
    """
    kz_beam = k0 + kz
    if kz_beam <= 0:
        return None
    cx, cy = beam_centre_px
    dx_mm = (kx / kz_beam) * distance_mm
    dy_mm = -(ky / kz_beam) * distance_mm
    px = cx + dx_mm / pixel_size_mm
    py = cy + dy_mm / pixel_size_mm
    return float(px), float(py)


def project_spots(spots, detector, beam):
    """
    Project all spots onto the detector.

    Parameters
    ----------
    spots : Spots
        Reciprocal-space positions (N, 3), millers, intensities.
    detector : Detector
    beam : Beam

    Returns
    -------
    list of dict, one per spot that projects onto a valid
    ray (kz_beam > 0), each with keys:
        'hkl'       : tuple (h, k, l)
        'px', 'py'  : float pixel coordinates
        'intensity' : float

    Spots that cannot be projected (kz_beam <= 0) are omitted.
    Spots are NOT filtered by detector bounds here; that is the
    caller's choice.
    """
    k0 = 1.0 / beam.wavelength_A
    distance_mm = detector.distance_mm
    pixel_size_mm = detector.pixel_size_mm
    beam_centre_px = detector.beam_centre_px

    out = []
    positions = spots.positions
    millers = spots.millers
    intensities = spots.intensities

    for i in range(len(intensities)):
        kx, ky, kz = (positions[i][0],
                      positions[i][1],
                      positions[i][2])
        proj = project_position(kx, ky, kz, k0,
                                distance_mm, pixel_size_mm,
                                beam_centre_px)
        if proj is None:
            continue
        px, py = proj
        hkl = (
            int(millers[i][0]),
            int(millers[i][1]),
            int(millers[i][2]),
        )
        out.append({
            "hkl": hkl,
            "px": px,
            "py": py,
            "intensity": float(intensities[i]),
        })
    return out


def in_detector_bounds(px, py, npx, npy):
    """True if (px, py) lies within the detector pixel grid."""
    return (0 <= px < npx) and (0 <= py < npy)
