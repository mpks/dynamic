"""
export_npz.py — default exporter writing the engine outputs to
NumPy NPZ files, with no noise and no point-spread broadening.

Two kinds of file are written:

  * one NPZ per image, holding that image's integrated spots
    (Miller indices, detector centroids px/py, intensities)
    together with the image index and centre angle;

  * one NPZ for the whole scan, holding the rocking curves
    (the substep angle axis and the intensity of each tracked
    reflection at every substep).

The per-image files follow the convention used by the plotting
tools, which read several of them and concatenate by image
index.
"""

from __future__ import annotations

import os

import numpy as np


def _hkl_array(millers):
    """Stack a list of (h, k, l) into an (N, 3) int array."""
    if len(millers) == 0:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(millers, dtype=np.int32)


def image_filename(out_dir, tag, image_index):
    """
    Build the per-image NPZ path.

    tag is an arbitrary run label; image_index is zero padded
    to four digits to sort correctly across a scan (scans are
    typically fewer than 1000 images).
    """
    name = f"image_{tag}_{image_index:04d}.npz"
    return os.path.join(out_dir, name)


def save_image(image, out_dir, tag):
    """
    Save one ImageResult to an NPZ file.

    Stored arrays:
      hkl_indices : (N, 3) int
      px, py      : (N,) float   detector centroids
      intensities : (N,) float   integrated intensities
      image_index : int
      angle_centre_deg : float
    """
    os.makedirs(out_dir, exist_ok=True)
    path = image_filename(out_dir, tag, image.image_index)
    np.savez(
        path,
        hkl_indices=_hkl_array(image.millers),
        px=image.px,
        py=image.py,
        intensities=image.intensities,
        image_index=image.image_index,
        angle_centre_deg=image.angle_centre_deg,
    )
    return path


def save_images(images, out_dir, tag):
    """Save every ImageResult; return the list of paths."""
    paths = []
    for image in images:
        path = save_image(image, out_dir, tag)
        paths.append(path)
        print(
            f"  wrote {os.path.basename(path)} "
            f"({len(image.millers)} spots)",
            flush=True,
        )
    return paths


def rocking_filename(out_dir, tag):
    """Build the scan-wide rocking-curve NPZ path."""
    return os.path.join(out_dir, f"rocking_{tag}.npz")


def save_rocking(rocking, out_dir, tag):
    """
    Save the RockingResult to a single NPZ for the whole scan.

    Stored arrays:
      angles_deg : (M,) float   substep angle axis
      hkl_indices : (K, 3) int  tracked reflections
      curves : (K, M) float     intensity per reflection per
                                substep, row-aligned to
                                hkl_indices
    """
    os.makedirs(out_dir, exist_ok=True)
    path = rocking_filename(out_dir, tag)

    hkls = sorted(rocking.curves.keys())
    hkl_arr = _hkl_array(hkls)
    if len(hkls) == 0:
        curves = np.zeros(
            (0, len(rocking.angles_deg)), dtype=float
        )
    else:
        curves = np.array(
            [rocking.curves[hkl] for hkl in hkls]
        )

    np.savez(
        path,
        angles_deg=rocking.angles_deg,
        hkl_indices=hkl_arr,
        curves=curves,
    )
    print(
        f"  wrote {os.path.basename(path)} "
        f"({len(hkls)} rocking curves, "
        f"{len(rocking.angles_deg)} substeps)",
        flush=True,
    )
    return path
