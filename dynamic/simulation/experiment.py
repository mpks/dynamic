"""
experiment.py — domain model for an electron diffraction
experiment and loading of geometry from a DIALS .expt file.

Defines the immutable data classes that describe what is being
measured (Detector, Beam, Geometry, Scan) and the Spots produced
by a simulation.  Also provides the cell-orientation geometry
(Geometry.oriented_cell) shared by every simulator backend, and
a reader that populates these objects from a DIALS experiment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation


# ----------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------

@dataclass(frozen=True)
class Detector:
    """Flat-panel detector geometry."""
    distance_mm: float
    npx: int
    npy: int
    pixel_size_mm: float
    beam_centre_px: tuple          # (cx, cy)
    fast_axis: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0])
    )
    slow_axis: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0])
    )


@dataclass(frozen=True)
class Beam:
    """Incident electron beam, propagating along +z."""
    wavelength_A: float
    energy_eV: float
    direction: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )


@dataclass(frozen=True)
class Geometry:
    """
    Crystal orientation and metric.

    B : 3x3  reciprocal metric (columns a*, b*, c*)
    U : 3x3  orientation matrix
    F : 3x3  goniometer fixed rotation
    S : 3x3  goniometer setting rotation
    rotation_axis : 3-vector  scan rotation axis
    orientation_seed : int or None
        Seed used to generate U (synthetic only); None for
        experiment-derived geometry.
    """
    B: np.ndarray
    U: np.ndarray
    F: np.ndarray
    S: np.ndarray
    rotation_axis: np.ndarray
    orientation_seed: int | None = None

    def oriented_cell(self, angle_deg: float) -> np.ndarray:
        """
        Return the real-space cell (ASE row convention) at the
        given scan angle.

        The construction matches the validated pipeline:

          step 1: crystal-frame cell = B^{-1}  (rows)
          step 2: rotate into the lab frame by
                  S @ R(angle) @ F @ U

        Returns
        -------
        cell_rows : ndarray (3, 3)
            Rows are the real-space vectors a, b, c, ready to
            pass to ase set_cell.
        """
        crystal_cell = np.linalg.inv(self.B)   # rows = a, b, c
        R = axis_angle_rotation_matrix(self.rotation_axis,
                                       angle_deg)
        full = self.S @ R @ self.F @ self.U
        # Apply rotation to column vectors, return to rows.
        cell_rows = (full @ crystal_cell.T).T
        return cell_rows


@dataclass(frozen=True)
class Scan:
    """
    Rotation scan description.

    angles_deg : ndarray
        Per-image start angles (the final scan point is implied,
        following the DIALS convention of omitting it).
    n_substeps : int
        Number of sub-angles integrated within each image.
    """
    angles_deg: np.ndarray
    n_substeps: int

    @property
    def delta_deg(self) -> float:
        """Oscillation width per image."""
        if len(self.angles_deg) < 2:
            raise ValueError(
                "Scan needs at least 2 angles to define a "
                "delta; got "
                f"{len(self.angles_deg)}."
            )
        return float(self.angles_deg[1]
                     - self.angles_deg[0])

    @property
    def n_images(self) -> int:
        return len(self.angles_deg)

    def substep_angles(self, image_index: int) -> np.ndarray:
        """
        Sub-angles for one image, tiling [start, start+delta)
        with endpoint excluded so that concatenating adjacent
        images produces a uniform non-overlapping series.
        """
        start = self.angles_deg[image_index]
        return np.linspace(
            start,
            start + self.delta_deg,
            self.n_substeps,
            endpoint=False,
        )


@dataclass(frozen=True)
class Spots:
    """
    Diffraction spots from one orientation.

    positions : ndarray (N, 3)  kx, ky, kz in Å^-1
    millers   : list of [h, k, l]
    intensities : ndarray (N,)
    """
    positions: np.ndarray
    millers: list
    intensities: np.ndarray


# ----------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------

def axis_angle_rotation_matrix(axis, phi_deg):
    """
    Rotation matrix for a rotation of phi_deg degrees about
    the given axis (need not be normalized).
    """
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    rotvec = axis * np.deg2rad(phi_deg)
    return Rotation.from_rotvec(rotvec).as_matrix()


def get_B_matrix(ort):
    """
    Build the reciprocal metric B from a real-space cell.

    ort : 3x3, rows are real-space cell vectors a, b, c (Å).
    Returns B : 3x3, columns are a*, b*, c* (Å^-1), in the
    Busing-Levy convention (a* along x, b* in the xy plane).
    """
    a_vec, b_vec, c_vec = ort[0], ort[1], ort[2]
    a = np.linalg.norm(a_vec)
    b = np.linalg.norm(b_vec)
    c = np.linalg.norm(c_vec)
    alpha = np.arccos(b_vec.dot(c_vec) / (b * c))
    beta = np.arccos(a_vec.dot(c_vec) / (a * c))
    gamma = np.arccos(a_vec.dot(b_vec) / (a * b))

    cos_a, cos_b, cos_g = (
        np.cos(alpha), np.cos(beta), np.cos(gamma)
    )
    sin_g = np.sin(gamma)
    V = a * b * c * np.sqrt(
        1 - cos_a**2 - cos_b**2 - cos_g**2
        + 2 * cos_a * cos_b * cos_g
    )
    B = np.array([
        [1 / a, 0, 0],
        [-cos_g / (a * sin_g), 1 / (b * sin_g), 0],
        [
            b * c * (
                cos_g * (cos_a - cos_b * cos_g) / sin_g
                - cos_b * sin_g
            ) / V,
            -(a * c * (cos_a - cos_b * cos_g)) / (V * sin_g),
            (a * b * sin_g) / V,
        ],
    ])
    return B


# ----------------------------------------------------------------
# DIALS experiment reader
# ----------------------------------------------------------------

def load_experiment(expt_file):
    """
    Read a DIALS .expt JSON file and return
    (detector, beam, geometry, scan_angles_deg).

    Assumes fixed U, B, F, S throughout the scan (no
    scan-varying crystal model).
    """
    with open(expt_file) as fh:
        data = json.load(fh)

    experiments = data["experiment"]
    if len(experiments) == 0:
        raise ValueError("No experiments found in expt file.")
    if len(experiments) > 1:
        print(
            f"WARNING: {len(experiments)} experiments found, "
            "using the first one."
        )

    exp = experiments[0]
    crystal = data["crystal"][exp["crystal"]]
    gonio = data["goniometer"][exp["goniometer"]]
    scan = data["scan"][exp["scan"]]
    beam_d = data["beam"][exp["beam"]]
    detector_d = data["detector"][exp["detector"]]

    # --- geometry (crystal orientation) -------------------
    S = np.array(gonio["setting_rotation"]).reshape(3, 3)
    F = np.array(gonio["fixed_rotation"]).reshape(3, 3)
    rotation_axis = np.array(gonio["rotation_axis"])
    angles = np.array(scan["properties"]["oscillation"])

    ar = crystal["real_space_a"]
    br = crystal["real_space_b"]
    cr = crystal["real_space_c"]
    ort = np.array([ar, br, cr])

    B = get_B_matrix(ort)
    UB = np.linalg.inv(ort)
    U = UB @ np.linalg.inv(B)

    geometry = Geometry(
        B=B, U=U, F=F, S=S,
        rotation_axis=rotation_axis,
        orientation_seed=None,
    )

    beam = _beam_from_dict(beam_d)
    detector = _detector_from_dict(detector_d, beam.direction)

    return detector, beam, geometry, angles


def _beam_from_dict(beam_d):
    """Build a Beam from the DIALS beam dictionary."""
    wavelength_A = float(beam_d["wavelength"])
    # DIALS stores the beam direction in "direction" (and the
    # equivalent "sample_to_source_direction"). The beam centre
    # projection is invariant under its sign, so we store it as
    # given without flipping.
    direction = beam_d.get("direction", None)
    if direction is None:
        direction = beam_d.get(
            "sample_to_source_direction", [0.0, 0.0, 1.0]
        )
    direction = np.array(direction, dtype=float)
    energy_eV = _wavelength_to_energy_eV(wavelength_A)
    return Beam(
        wavelength_A=wavelength_A,
        energy_eV=energy_eV,
        direction=direction,
    )


def _detector_from_dict(detector_d, beam_direction):
    """
    Build a Detector from the DIALS detector dictionary.

    DIALS detectors are a list of panels; we use panel 0.
    The panel gives fast_axis, slow_axis, origin (mm),
    pixel_size (mm), image_size (px).  The sample-to-detector
    distance is the projection of origin onto the panel
    normal, and the beam centre is where the incident beam
    (beam_direction) pierces the panel, matching the DIALS
    Panel.get_beam_centre_px(s0) calculation.
    """
    panels = detector_d["panels"]
    panel = panels[0]

    fast = np.array(panel["fast_axis"], dtype=float)
    slow = np.array(panel["slow_axis"], dtype=float)
    origin = np.array(panel["origin"], dtype=float)
    pixel_size = panel["pixel_size"]          # (fast, slow) mm
    image_size = panel["image_size"]          # (npx, npy)

    npx = int(image_size[0])
    npy = int(image_size[1])
    px_fast = float(pixel_size[0])
    px_slow = float(pixel_size[1])

    # The incident beam direction (its sign does not affect the
    # projection onto the panel).
    beam_dir = np.asarray(beam_direction, dtype=float)

    # Distance: perpendicular distance from the sample (lab
    # origin) to the panel plane, = |origin . normal|.
    normal = np.cross(fast, slow)
    normal = normal / np.linalg.norm(normal)
    distance_mm = abs(float(origin.dot(normal)))

    # Beam centre: where the incident beam pierces the panel,
    # via the dxtbx D-matrix projection (matches DIALS).
    cx_px, cy_px = _ray_intersection_px(beam_dir, fast, slow,
                                        origin, px_fast,
                                        px_slow)

    if abs(px_fast - px_slow) > 1e-9:
        print(
            "WARNING: non-square pixels "
            f"({px_fast} != {px_slow}); using fast size."
        )

    return Detector(
        distance_mm=distance_mm,
        npx=npx,
        npy=npy,
        pixel_size_mm=px_fast,
        beam_centre_px=(cx_px, cy_px),
        fast_axis=fast,
        slow_axis=slow,
    )


def _ray_intersection_px(s1, fast, slow, origin,
                         px_fast, px_slow):
    """
    Project a scattered ray onto the panel, returning the
    intersection in pixels — the dxtbx Panel approach.

    The panel basis matrix d has the fast axis, slow axis and
    origin as its columns:

        d = [ fast | slow | origin ]

    Its inverse D = d^{-1} projects a lab-frame ray s1 onto
    the panel.  The first two components of D @ s1, divided by
    the third (the perspective divide), give the position on
    the panel in millimetres along fast and slow; dividing by
    the pixel sizes converts to pixels.  No trigonometry is
    needed.

    The lab origin is at the beam-sample intersection, so the
    ray s1 is taken as a direction from that origin.
    """
    d = np.column_stack([fast, slow, origin])
    D = np.linalg.inv(d)
    v = D @ np.asarray(s1, dtype=float)
    if v[2] == 0:
        raise ValueError(
            "Ray is parallel to the panel; no intersection."
        )
    x_mm = v[0] / v[2]
    y_mm = v[1] / v[2]
    cx_px = x_mm / px_fast
    cy_px = y_mm / px_slow
    return float(cx_px), float(cy_px)


def _wavelength_to_energy_eV(wavelength_A):
    """
    Relativistic electron energy (eV) for a given de Broglie
    wavelength in Angstrom.  Inverts
        lambda = hc / sqrt(E^2 + 2 m0c2 E).
    """
    wl_m = wavelength_A * 1e-10
    m0c2 = 510998.9461                       # eV
    hc = 4.135667696e-15 * 299792458.0       # eV·m
    lhs = (hc / wl_m) ** 2
    # E^2 + 2 m0c2 E - lhs = 0
    energy_eV = -m0c2 + np.sqrt(m0c2 ** 2 + lhs)
    return float(energy_eV)
