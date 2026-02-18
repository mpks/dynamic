"""Used to extract and manipulate unit cell orientation"""
from scipy.spatial.transform import Rotation
from dxtbx.model.experiment_list import ExperimentListFactory as ExpList
from scitbx import matrix
from itertools import tee
import numpy as np


def get_euler_angles(expt_file, exp_index=0, npz_file=None):

    output = extract_unit_cell_orientation(expt_file,
                                           exp_index)

    exps = ExpList.from_json_file(expt_file, check_format=False)
    exp = exps[exp_index]
    path = exp.imageset.get_path(0)
    print("PATH", path)

    real_space_axes, zone_axes, initial_orientations = output

    angles = []
    for start_uc, end_uc in zip(initial_orientations, real_space_axes):

        a0, b0, c0 = start_uc
        a1, b1, c1 = end_uc

        alpha, beta, gamma = get_angles_xyz(a0, b0, c0, a1, b1, c1,
                                            degrees=True)
        angles.append((alpha, beta, gamma))

    out = f'euler_angles_exp_{exp_index:04d}.npz'
    if npz_file:
        out = npz_file
    np.savez(out, angles=angles, real_space_axes=real_space_axes,
             initial_orientations=initial_orientations,
             zone_axes=zone_axes, original_file=path)


def compare_unit_cells(a0, b0, c0, a1, b1, c1,
                       length_tolerance_percent=0.1,
                       angle_tolerance_deg=0.1):
    """Compare two unit cells by lenght of each vector and by angles"""
    norm = np.linalg.norm
    lt = length_tolerance_percent / 100.
    assert (norm(a0) - norm(a1)) / norm(a0) <= lt
    assert (norm(b0) - norm(b1)) / norm(b0) <= lt
    assert (norm(c0) - norm(c1)) / norm(c0) <= lt

    alpha = angle_between_vectors(a0, b0)
    beta = angle_between_vectors(a0, c0)
    gamma = angle_between_vectors(b0, c0)

    alpha1 = angle_between_vectors(a1, b1)
    beta1 = angle_between_vectors(a1, c1)
    gamma1 = angle_between_vectors(b1, c1)

    assert abs(alpha - alpha1) < angle_tolerance_deg
    assert abs(beta - beta1) < angle_tolerance_deg
    assert abs(gamma - gamma1) < angle_tolerance_deg


def get_angles_xyz(a0, b0, c0, a1, b1, c1, degrees=False):
    """
    Given two unit cell orientations initial_cell = [a0, b0, c0] and
    final_cell = [a1, b1, c1], compute rotation angles alpha, beta,
    and gamma (around x, y, and z) that rotate initial_cell into final_cell
    """

    compare_unit_cells(a0, b0, c0, a1, b1, c1)
    A0 = np.column_stack([a0, b0, c0])
    A = np.column_stack([a1,  b1,  c1])
    R = A @ np.linalg.inv(A0)
    rot = Rotation.from_matrix(R)
    alpha, beta, gamma = rot.as_euler('xyz', degrees=True)
    return alpha, beta, gamma


def extract_unit_cell_orientation(expt_file, exp_index=0):

    exps = ExpList.from_json_file(expt_file, check_format=False)
    exp = exps[exp_index]
    path = exp.imageset.get_path(0)

    print(f"Extracting unit cell orientation for exp={exp_index}")
    print(f"Path: {path}")

    crystal = exp.crystal
    beam = exp.beam
    scan = exp.scan
    gonio = exp.goniometer

    num_pts = scan.get_num_images() + 1

    if beam.num_scan_points > 0:
        us0 = []
        for i in range(beam.num_scan_points):
            s0 = matrix.col(beam.get_s0_at_scan_point(i))
            us0.append(s0.normalize())
    else:
        us0 = [matrix.col(beam.get_unit_s0()) for _ in range(num_pts)]

    if gonio.num_scan_points > 0:
        get_set = gonio.get_setting_rotation_at_scan_point
        nscn = gonio.num_scan_points
        S_mats = [matrix.sqr(get_set(i)) for i in range(nscn)]
    else:
        S_mats = [matrix.sqr(gonio.get_setting_rotation()) for _ in
                  range(num_pts)]

    F_mats = [matrix.sqr(gonio.get_fixed_rotation()) for _ in
              range(num_pts)]
    start, stop = scan.get_array_range()
    R_mats = []
    axis = matrix.col(gonio.get_rotation_axis_datum())
    for i in range(start, stop + 1):
        phi = scan.get_angle_from_array_index(i, deg=False)
        ang_as_r3 = axis.axis_and_angle_as_r3_rotation_matrix
        R = matrix.sqr(ang_as_r3(phi, deg=False))
        R_mats.append(R)

    if crystal.num_scan_points > 0:
        U_mats = [
            matrix.sqr(crystal.get_U_at_scan_point(i))
            for i in range(crystal.num_scan_points)
        ]
        B_mats = [
            matrix.sqr(crystal.get_B_at_scan_point(i))
            for i in range(crystal.num_scan_points)
        ]
    else:
        U_mats = [matrix.sqr(crystal.get_U()) for _ in range(num_pts)]
        B_mats = [matrix.sqr(crystal.get_B()) for _ in range(num_pts)]

    check = {len(x) for x in (us0, S_mats, F_mats, R_mats, U_mats)}
    assert len(check) == 1

    SRFU = (S * R * F * U for S, R, F, U in
            zip(S_mats, R_mats, F_mats, U_mats))

    U_frames = []
    for U1, U2 in pairwise(SRFU):
        M = U2 * U1.transpose()
        qt = M.r3_rotation_matrix_as_unit_quaternion()
        angle, axis = qt.unit_quaternion_as_axis_and_angle(deg=False)
        M_half = axis.axis_and_angle_as_r3_rotation_matrix(angle/2,
                                                           deg=False)
        U_frames.append(M_half * U1)

    B_frames = []
    for B1, B2 in pairwise(B_mats):
        B_frames.append((B1 + B2) / 2)

    UB_frames = [U * B for U, B in zip(U_frames, B_frames)]

    frac_mats = [m.transpose() for m in UB_frames]

    # Calculate zone axes, which also requires the beam directions
    # at the framecentres
    us0_frames = []
    for d1, d2 in pairwise(us0):
        us0_frames.append(((d1 + d2) / 2).normalize())

    scale = 1   # I added this line (see original func for more detail)
    zone_axes = [frac * (d * scale) for frac, d in
                 zip(frac_mats, us0_frames)]

    orthog_mats = (frac.inverse() for frac in frac_mats)
    h = matrix.col((1, 0, 0))
    k = matrix.col((0, 1, 0))
    l = matrix.col((0, 0, 1))      # noqa: E741
    real_space_axes = [(o * h, o * k, o * l) for o in orthog_mats]

    # Vectors in the crystal frame
    Os = [b.transpose().inverse() for b in B_frames]
    initial_orient = [(o * h, o * k, o * l) for o in Os]

    return real_space_axes, zone_axes, initial_orient


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def angle_between_vectors(v1, v2, degrees=True):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    # Normalize vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v1_norm == 0 or v2_norm == 0:
        raise ValueError("One of the vectors has zero length")

    # Dot product of unit vectors
    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)

    # Numerical safety (avoid arccos domain error)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)

    if degrees:
        return np.degrees(angle_rad)
    return angle_rad
