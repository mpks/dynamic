"""
projection.py — project reciprocal-space diffraction spots onto
the detector panel, in pixel coordinates.

The abTEM simulation is taken as physical truth and is left
untouched: the beam propagates along +z in abTEM's frame and
each reflection comes out as a reciprocal-space vector
g = (kx, ky, kz) with no detector baked in.  sg_max controls
how much of the Ewald sphere is sampled; that is abTEM's
concern, not the projection's.

The projection is the only place the real detector geometry
enters.  The scattered ray for a reflection is

    s1 = s0 + g ,

where s0 = k0 * (incident beam propagation direction) and
k0 = 1 / wavelength.  The incident direction is taken from the
experiment beam (so it is NOT assumed to be exactly +z; the
slight experimental tilt is honoured), while g = (kx, ky, kz)
is passed through from abTEM unchanged.

s1 is projected onto the real, possibly tilted, detector panel
using the panel basis (fast axis, slow axis, origin) via the
dxtbx D-matrix projection:

    d = [ fast | slow | origin ]   (columns)
    D = d^{-1}

D maps the lab ray s1 to panel coordinates; the perspective
divide by the third component gives millimetres along fast and
slow, divided by the pixel sizes to get pixels.
"""

from __future__ import annotations

import numpy as np


def panel_matrix(detector):
    """
    Return D = inv([fast | slow | origin]) for the detector
    panel, the matrix that projects a lab-frame ray onto the
    panel.
    """
    d = np.column_stack([
        np.asarray(detector.fast_axis, dtype=float),
        np.asarray(detector.slow_axis, dtype=float),
        np.asarray(detector.origin, dtype=float),
    ])
    return np.linalg.inv(d)


def beam_s0(beam):
    """
    Incident wavevector s0 = k0 * (unit beam propagation
    direction), taken from the experiment beam.  k0 = 1 /
    wavelength.  The direction is normalised so only its
    orientation (the slight experimental tilt away from +z)
    matters, not its stored magnitude.
    """
    k0 = 1.0 / beam.wavelength_A
    d = np.asarray(beam.direction, dtype=float)
    d = d / np.linalg.norm(d)
    return k0 * d


def project_ray(s1, D, px_fast_mm, px_slow_mm):
    """
    Project a scattered ray s1 onto the panel using the panel
    projection matrix D.

    Returns (px, py) in pixels, or None only if the ray is
    parallel to the panel (no intersection).  The sign of the
    perspective divisor depends on the panel's origin/normal
    convention, so it is not used to reject spots.
    """
    v = D @ np.asarray(s1, dtype=float)
    if v[2] == 0:
        return None
    x_mm = v[0] / v[2]
    y_mm = v[1] / v[2]
    return x_mm / px_fast_mm, y_mm / px_slow_mm


def project_position(kx, ky, kz, beam, detector,
                     D=None, s0=None):
    """
    Project a single abTEM reciprocal-space vector onto the
    detector.

    Parameters
    ----------
    kx, ky, kz : float
        abTEM reciprocal-space components (A^-1), passed through
        unchanged (beam along +z in abTEM's frame).
    beam : Beam
        Supplies the wavelength and the experimental incident
        beam direction used for s0.
    detector : Detector
        Supplies the panel basis (fast/slow/origin) and pixel
        size.
    D : ndarray, optional
        Precomputed panel matrix; pass in a loop to avoid
        recomputing the inverse per spot.
    s0 : ndarray, optional
        Precomputed incident wavevector beam_s0(beam); pass in a
        loop to avoid recomputing it per spot.

    Returns
    -------
    (px, py) : tuple of float, or None only if the ray is
        parallel to the panel.
    """
    if D is None:
        D = panel_matrix(detector)
    if s0 is None:
        s0 = beam_s0(beam)
    s1 = (s0[0] + kx, s0[1] + ky, s0[2] + kz)
    px_mm = detector.pixel_size_mm
    return project_ray(s1, D, px_mm, px_mm)


def project_spots(spots, detector, beam):
    """
    Project all spots onto the detector.

    The incident beam direction and the detector panel basis are
    taken from the experiment; the abTEM positions (kx, ky, kz)
    are used as given.

    Parameters
    ----------
    spots : Spots
        Reciprocal-space positions (N, 3), millers, intensities.
    detector : Detector
    beam : Beam

    Returns
    -------
    list of dict, one per spot that projects onto the panel
    (only rays exactly parallel to the panel are omitted), each
    with keys:
        'hkl'       : tuple (h, k, l)
        'px', 'py'  : float pixel coordinates
        'intensity' : float

    Spots are NOT filtered by detector bounds here; that is the
    caller's choice.
    """
    D = panel_matrix(detector)
    s0 = beam_s0(beam)
    px_mm = detector.pixel_size_mm

    out = []
    positions = spots.positions
    millers = spots.millers
    intensities = spots.intensities

    for i in range(len(intensities)):
        kx = positions[i][0]
        ky = positions[i][1]
        kz = positions[i][2]
        s1 = (s0[0] + kx, s0[1] + ky, s0[2] + kz)
        proj = project_ray(s1, D, px_mm, px_mm)
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
