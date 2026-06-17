"""
engine.py — integrate a single diffraction image.

The engine integrates one image of a rotation scan: it loops the
substep angles of that image, runs the Simulator at each, projects
the spots to the detector, and accumulates per Miller index the
total intensity and the intensity-weighted centroid (px, py).  It
returns one ImageResult plus that image's per-substep rocking
intensities.

The engine is backend-agnostic (it depends only on the abstract
Simulator interface and the domain data classes) and deals with a
single image only.  Looping over images, image selection, output
writing and scan-wide rocking-curve assembly all live in the
driver layer above.

Parallelism is over the substeps of the image: the substeps are
split into n_workers batches, each handled in a separate process;
the engine merges the partial accumulators.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import ase.io

from dynamic.simulation.experiment import (
    axis_angle_rotation_matrix,
)
from dynamic.simulation.projection import project_spots


# ----------------------------------------------------------------
# Engine parameters and results
# ----------------------------------------------------------------

@dataclass(frozen=True)
class EngineParams:
    """
    Engine-level (backend-independent) configuration.

    base_thickness_A : float
        Crystal thickness along the beam at zero tilt (A); the
        per-substep thickness is base_thickness_A / cos(angle).
    intensity_cut : float
        Reflections whose integrated intensity is at or below
        this value are dropped from an image after integration.
    n_workers : int
        Number of worker processes for substep parallelism.
    rocking_hkl : tuple
        Miller indices (as tuples) to record rocking curves for.
        Empty means no rocking curves.
    """
    base_thickness_A: float
    intensity_cut: float = 0.0
    n_workers: int = 1
    rocking_hkl: tuple = ()


@dataclass(frozen=True)
class ImageResult:
    """
    Integrated diffraction for one image.

    millers : list of (h, k, l)
    px, py : ndarray
        Intensity-weighted detector centroids (pixels).
    intensities : ndarray
        Integrated intensities (sum over substeps).
    angle_centre_deg : float
        Centre angle of the image (start + delta/2).
    image_index : int
    """
    millers: list
    px: np.ndarray
    py: np.ndarray
    intensities: np.ndarray
    angle_centre_deg: float
    image_index: int


# ----------------------------------------------------------------
# Atom orientation
# ----------------------------------------------------------------

def make_base_atoms(cif_file, geometry):
    """
    Read the CIF and place the atoms into the B^{-1} crystal
    cell, preserving the CIF fractional coordinates.

    Uses scale_atoms=True so the fractional coordinates are
    kept (convention-independent) while the cell becomes
    B^{-1}.  A volume check (10% tolerance) guards against a
    CIF cell whose metric grossly mismatches the experiment B;
    small differences (e.g. CIF and ED data at different
    temperatures) are expected and allowed.
    """
    atoms = ase.io.read(cif_file)
    v_before = atoms.get_volume()
    crystal_cell = np.linalg.inv(geometry.B)
    atoms.set_cell(crystal_cell, scale_atoms=True)
    v_after = atoms.get_volume()
    if not np.isclose(v_before, v_after, rtol=0.10):
        raise ValueError(
            "Cell volume changed on alignment: "
            f"{v_before:.2f} -> {v_after:.2f} A^3. "
            "CIF cell and experiment B metric differ by "
            "more than 10%."
        )
    return atoms


def orient_atoms(base_atoms, geometry, angle_deg):
    """
    Rotate the base atoms into the lab frame for a scan angle.

    Applies S @ R(angle) @ F @ U to the B^{-1} cell with
    scale_atoms=True so the atoms rotate rigidly, then wraps.
    """
    atoms = base_atoms.copy()
    cell = np.array(atoms.get_cell())
    R = axis_angle_rotation_matrix(geometry.rotation_axis,
                                   angle_deg)
    full = geometry.S @ R @ geometry.F @ geometry.U
    new_cell = (full @ cell.T).T
    atoms.set_cell(new_cell, scale_atoms=True)
    atoms.wrap()
    return atoms


# ----------------------------------------------------------------
# Worker: process a batch of substeps for one image
# ----------------------------------------------------------------

def _process_substep_batch(args):
    """
    Process a batch of substep angles for a single image.

    Runs in a worker process.  Returns partial accumulators:
      sum_I, sum_I_px, sum_I_py : dict (h,k,l) -> float
      rocking : dict (h,k,l) -> dict(angle -> intensity)

    args is a tuple (picklable):
      (cif_file, geometry, simulator, detector, beam,
       base_thickness_A, angle_batch, rocking_hkl_set)
    """
    (cif_file, geometry, simulator, detector, beam,
     base_thickness_A, angle_batch, rocking_hkl_set) = args

    base_atoms = make_base_atoms(cif_file, geometry)

    sum_I = {}
    sum_I_px = {}
    sum_I_py = {}
    rocking = {}

    for angle in angle_batch:
        atoms = orient_atoms(base_atoms, geometry, angle)
        tilt_rad = np.deg2rad(angle)
        thickness_A = base_thickness_A / np.cos(tilt_rad)

        spots = simulator.simulate(atoms, thickness_A)
        projected = project_spots(spots, detector, beam)

        for s in projected:
            hkl = s["hkl"]
            inten = s["intensity"]
            px = s["px"]
            py = s["py"]
            sum_I[hkl] = sum_I.get(hkl, 0.0) + inten
            sum_I_px[hkl] = (
                sum_I_px.get(hkl, 0.0) + inten * px
            )
            sum_I_py[hkl] = (
                sum_I_py.get(hkl, 0.0) + inten * py
            )
            if hkl in rocking_hkl_set:
                if hkl not in rocking:
                    rocking[hkl] = {}
                rocking[hkl][float(angle)] = inten

    return sum_I, sum_I_px, sum_I_py, rocking


# ----------------------------------------------------------------
# Merging partial accumulators
# ----------------------------------------------------------------

def _merge_sum_dicts(partials):
    """
    Merge a list of (sum_I, sum_I_px, sum_I_py, rocking)
    tuples into combined dictionaries.
    """
    sum_I = {}
    sum_I_px = {}
    sum_I_py = {}
    rocking = {}
    for p_I, p_px, p_py, p_rock in partials:
        for hkl, v in p_I.items():
            sum_I[hkl] = sum_I.get(hkl, 0.0) + v
        for hkl, v in p_px.items():
            sum_I_px[hkl] = sum_I_px.get(hkl, 0.0) + v
        for hkl, v in p_py.items():
            sum_I_py[hkl] = sum_I_py.get(hkl, 0.0) + v
        for hkl, ang_map in p_rock.items():
            if hkl not in rocking:
                rocking[hkl] = {}
            rocking[hkl].update(ang_map)
    return sum_I, sum_I_px, sum_I_py, rocking


# ----------------------------------------------------------------
# Per-image integration
# ----------------------------------------------------------------

def _integrate_image(sum_I, sum_I_px, sum_I_py,
                     intensity_cut, angle_centre_deg,
                     image_index):
    """
    Turn the merged accumulators into an ImageResult, computing
    intensity-weighted centroids and applying the intensity
    cut on integrated intensity.
    """
    millers = []
    px_list = []
    py_list = []
    int_list = []

    for hkl in sorted(sum_I.keys()):
        total = sum_I[hkl]
        if total <= intensity_cut:
            continue
        px = sum_I_px[hkl] / total
        py = sum_I_py[hkl] / total
        millers.append(hkl)
        px_list.append(px)
        py_list.append(py)
        int_list.append(total)

    return ImageResult(
        millers=millers,
        px=np.array(px_list),
        py=np.array(py_list),
        intensities=np.array(int_list),
        angle_centre_deg=angle_centre_deg,
        image_index=image_index,
    )


# ----------------------------------------------------------------
# Top-level driver
# ----------------------------------------------------------------

def run_image(cif_file, detector, beam, geometry, scan,
              image_index, simulator, params):
    """
    Integrate a single image of the scan.

    Parameters
    ----------
    cif_file : str
    detector : Detector
    beam : Beam
    geometry : Geometry
    scan : Scan
    image_index : int
        Which image of the scan to integrate.
    simulator : Simulator
    params : EngineParams

    Returns
    -------
    (image_result, image_rocking)
      image_result : ImageResult
          Integrated spots for this image.
      image_rocking : dict (h, k, l) -> dict(angle -> intensity)
          Per-substep intensities for the tracked reflections
          of this image.  The driver places these on the full
          scan axis.
    """
    if scan.n_substeps % 2 != 0:
        print(
            "WARNING: n_substeps is odd; the image centre "
            "angle is not sampled by a substep."
        )

    rocking_set = set(params.rocking_hkl)
    sub_angles = scan.substep_angles(image_index)
    start = scan.angles_deg[image_index]
    centre = start + 0.5 * scan.delta_deg

    partials = _run_image_substeps(
        cif_file, detector, beam, geometry, simulator,
        params, sub_angles, rocking_set,
    )

    (sum_I, sum_I_px, sum_I_py,
     image_rocking) = _merge_sum_dicts(partials)

    image_result = _integrate_image(
        sum_I, sum_I_px, sum_I_py,
        params.intensity_cut, centre, image_index,
    )
    return image_result, image_rocking


def _run_image_substeps(cif_file, detector, beam, geometry,
                        simulator, params, sub_angles,
                        rocking_set):
    """
    Dispatch one image's substeps across worker processes and
    return the list of partial accumulators.
    """
    batches = _split_batches(sub_angles, params.n_workers)
    work = [
        (cif_file, geometry, simulator, detector, beam,
         params.base_thickness_A, batch, rocking_set)
        for batch in batches
    ]

    if params.n_workers == 1:
        return [_process_substep_batch(w) for w in work]

    with ProcessPoolExecutor(
        max_workers=params.n_workers
    ) as executor:
        results = list(
            executor.map(_process_substep_batch, work)
        )
    return results


def _split_batches(angles, n_workers):
    """Split an array of angles into n_workers contiguous
    batches (some may be empty if there are fewer angles
    than workers)."""
    return [
        batch for batch in np.array_split(angles, n_workers)
        if len(batch) > 0
    ]
