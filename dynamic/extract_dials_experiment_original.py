"""
extract_experiment.py — extract per-image geometry from a DIALS
experiment into a self-contained NPZ file for use without DIALS.

Run this inside the DIALS environment:

    python extract_experiment.py refined.expt refined.refl \
        -o geometry.npz

The NPZ contains:
    n_images        : int
    image_range     : (first, last) 1-based image numbers
    start_angle_deg : scan start angle in degrees
    delta_deg       : oscillation width per image in degrees
    wavelength_A    : beam wavelength in Angstrom
    distance_mm     : sample-to-detector distance in mm
    beam_centre_px  : (cx, cy) beam centre in pixels
    pixel_size_mm   : pixel size in mm (assumed square)
    npx, npy        : detector dimensions in pixels
    U_frames        : (n_images, 3, 3) orientation matrices
    B_frames        : (n_images, 3, 3) metric matrices
    cell_vectors    : (n_images, 3, 3) real-space cell vectors
                      in lab frame (rows = a, b, c) in Angstrom
    scan_angles_deg : (n_images,) angle at centre of each image
"""

from __future__ import annotations

import argparse
import sys
from itertools import tee

import numpy as np
from scitbx import matrix
from dxtbx.model import ExperimentList


def main(argv=None):

    desc = ("Extract per-image geometry from DIALS experiment "
            "into NPZ for use without DIALS.")
    p = argparse.ArgumentParser(description=desc)

    p.add_argument("expt_file", help=".expt file")
    p.add_argument("-o", "--output", default="experiment_geometry.npz",
                   help="Output NPZ file")
    args = p.parse_args(argv)

    experiments = ExperimentList.from_file(args.expt_file,
                                           check_format=False)
    if len(experiments) == 0:
        sys.exit("No experiments found.")
    if len(experiments) > 1:
        msg = f"WARNING: {len(experiments)} experiments found, "
        msg += "using the first one."
        print(msg)

    exp = experiments[0]
    print("Extracting per-image orientation data…")
    dat = extract_experiment_data(exp)
    print("Extracting detector geometry…")
    geo = extract_detector_geometry(exp)

    n_images = len(dat["images"])
    image_range = (dat["images"][0], dat["images"][-1])

    # Scan parameters
    scan = exp.scan
    start_angle = scan.get_oscillation()[0]
    delta = scan.get_oscillation()[1]

    # Pack matrices into arrays
    U_arr = np.array(
        [np.array(U.elems).reshape(3, 3) for U in dat["U_frames"]]
    )
    B_arr = np.array(
        [np.array(B.elems).reshape(3, 3) for B in dat["B_frames"]]
    )
    # cell_vectors: rows are a, b, c vectors in lab frame
    cv_arr = np.array([
        [
            list(a_vec.elems),
            list(b_vec.elems),
            list(c_vec.elems),
        ]
        for a_vec, b_vec, c_vec in dat["cell_vectors"]
    ])

    # initial_orient: crystal-frame cell vectors from B^{-T}
    # shape (3, 3), rows = a, b, c  — same as initial_orientations
    # in your old euler_angles NPZ files
    a0, b0, c0 = dat["initial_orient"]
    initial_orient_arr = np.array([
        list(a0.elems),
        list(b0.elems),
        list(c0.elems),
    ])

    np.savez(
        args.output,
        n_images=n_images,
        image_range=image_range,
        start_angle_deg=start_angle,
        delta_deg=delta,
        wavelength_A=geo["wavelength_A"],
        distance_mm=geo["distance_mm"],
        pixel_size_mm=geo["pixel_size_mm"],
        npx=geo["npx"],
        npy=geo["npy"],
        beam_centre_px=geo["beam_centre_px"],
        U_frames=U_arr,
        B_frames=B_arr,
        cell_vectors=cv_arr,
        scan_angles_deg=np.array(dat["scan_angles_deg"]),
        initial_orient=initial_orient_arr,
    )

    print(f"Saved geometry for {n_images} images to {args.output}")
    print(f"  Image range  : {image_range}")
    print(f"  Start angle  : {start_angle:.3f} deg")
    print(f"  Delta        : {delta:.4f} deg")
    print(f"  Wavelength   : {geo['wavelength_A']:.6f} A")
    print(f"  Distance     : {geo['distance_mm']:.2f} mm")
    print(f"  Pixel size   : {geo['pixel_size_mm']*1000:.1f} um")
    print(f"  Detector     : {geo['npx']} x {geo['npy']} px")
    print(
        f"  Beam centre  : "
        f"({geo['beam_centre_px'][0]:.2f}, "
        f"{geo['beam_centre_px'][1]:.2f}) px"
    )


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def extract_experiment_data(exp):
    """
    Extract per-image orientation and metric matrices.
    Adapted from dials.frame_orientations.
    Returns a dict with n_images entries for each quantity.
    """
    crystal = exp.crystal
    scan = exp.scan
    gonio = exp.goniometer

    image_range = scan.get_image_range()
    images = list(range(image_range[0], image_range[1] + 1))
    num_scan_points = scan.get_num_images() + 1

    # Goniometer matrices at each scan point
    if gonio.num_scan_points > 0:
        S_mats = [matrix.sqr(gonio.get_setting_rotation_at_scan_point(i))
                  for i in range(gonio.num_scan_points)]
    else:
        S_mats = [matrix.sqr(gonio.get_setting_rotation())
                  for _ in range(num_scan_points)]

    F_mats = [matrix.sqr(gonio.get_fixed_rotation())
              for _ in range(num_scan_points)]

    start, stop = scan.get_array_range()
    R_mats = []
    axis = matrix.col(gonio.get_rotation_axis_datum())
    for i in range(start, stop + 1):
        phi = scan.get_angle_from_array_index(i, deg=False)
        aa_as_r3 = axis.axis_and_angle_as_r3_rotation_matrix
        R = matrix.sqr(aa_as_r3(phi, deg=False))
        R_mats.append(R)

    # Crystal U and B at each scan point
    if crystal.num_scan_points > 0:
        U_mats = [matrix.sqr(crystal.get_U_at_scan_point(i))
                  for i in range(crystal.num_scan_points)]
        B_mats = [matrix.sqr(crystal.get_B_at_scan_point(i))
                  for i in range(crystal.num_scan_points)]
    else:
        U_mats = [matrix.sqr(crystal.get_U())
                  for _ in range(num_scan_points)]
        B_mats = [matrix.sqr(crystal.get_B())
                  for _ in range(num_scan_points)]

    # Full lab-frame orientation at each scan-point boundary
    SRFU = [S*R*F*U for S, R, F, U in zip(S_mats, R_mats, F_mats, U_mats)]

    # Interpolate to image centres using half-step slerp
    U_frames = []
    for U1, U2 in pairwise(SRFU):
        M = U2 * U1.transpose()

        M_unit_quat = M.r3_rotation_matrix_as_unit_quaternion
        M_axis_and_angle = M_unit_quat().unit_quaternion_as_axis_and_angle
        angle, ax = M_axis_and_angle(deg=False)

        M_half = ax.axis_and_angle_as_r3_rotation_matrix(angle / 2,
                                                         deg=False)
        U_frames.append(M_half * U1)

    # B at image centres: average of adjacent scan-point values
    B_frames = []
    for B1, B2 in pairwise(B_mats):
        B_frames.append((B1 + B2) / 2)

    UB_frames = [U * B for U, B in zip(U_frames, B_frames)]

    # Real-space cell vectors at each image centre
    # orthog = (UB^T)^{-1} = (frac)^{-1}
    h = matrix.col((1, 0, 0))
    k = matrix.col((0, 1, 0))
    l = matrix.col((0, 0, 1))     # noqa: E741
    cell_vectors = []
    for UB in UB_frames:
        frac = UB.transpose()
        orth = frac.inverse()
        a_vec = orth * h
        b_vec = orth * k
        c_vec = orth * l
        cell_vectors.append((a_vec, b_vec, c_vec))

    # Crystal-frame cell vectors from B^{-T} (constant metric,
    # no orientation — equivalent to initial_orient in your
    # old pipeline).  B is the reciprocal-space metric matrix
    # so B^{-T} columns are the real-space cell vectors in the
    # crystal orthogonal frame, independent of U.
    # We use the first B_frame as representative; for a well-
    # refined dataset B barely changes across the scan.
    B0 = B_frames[0]
    B0_inv_T = B0.transpose().inverse()
    initial_orient = (
        B0_inv_T * h,
        B0_inv_T * k,
        B0_inv_T * l,
    )

    # Scan angles at image centres
    scan_angles = []
    for img in images:
        # get_angle_from_array_index uses 0-based array index
        arr_idx = img - image_range[0]
        centre = scan.get_angle_from_array_index(
            arr_idx + 0.5, deg=True
        )
        scan_angles.append(centre)

    return {
        "images": images,
        "U_frames": U_frames,
        "B_frames": B_frames,
        "cell_vectors": cell_vectors,
        "scan_angles_deg": scan_angles,
        "initial_orient": initial_orient,
    }


def extract_detector_geometry(exp):
    """Extract flat-panel detector geometry from experiment."""
    detector = exp.detector
    beam = exp.beam
    panel = detector[0]

    # Pixel size — assume square
    px_mm, py_mm = panel.get_pixel_size()
    if abs(px_mm - py_mm) > 1e-6:
        raise ValueError("Non-square pixels not supported")

    npx, npy = panel.get_image_size()
    distance_mm = panel.get_distance()

    # Beam centre in pixels
    s0 = beam.get_s0()
    bc_mm = panel.get_beam_centre(s0)
    cx = bc_mm[0] / px_mm
    cy = bc_mm[1] / py_mm

    wavelength_A = beam.get_wavelength()

    return {
        "wavelength_A": wavelength_A,
        "distance_mm": distance_mm,
        "pixel_size_mm": px_mm,
        "npx": npx,
        "npy": npy,
        "beam_centre_px": (cx, cy),
    }


if __name__ == "__main__":
    main()
