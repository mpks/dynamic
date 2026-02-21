#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import sys


def main():

    input_files = sys.argv[1:]

    for file in input_files:
        interpolate(file)


def interpolate(input_file, subsamples=50):

    data = np.load(input_file)
    print(f"Interpolating for file: {input_file}")
    angles = data['angles']  # (N, 3) Euler angles
    if 'original_file' in data:
        original_file = data['original_file']
    else:
        original_file = 'unknown'
    initial_orientations = data['initial_orientations']  # (N, 3, 3)

    n_original = len(angles)
    n_interpolated = subsamples * n_original

    # Original and new frame indices
    original_indices = np.arange(n_original)
    new_indices = np.linspace(0, n_original - 1, n_interpolated)
    print('New indices', new_indices[0:subsamples+1], " ...")

    # ===== 1. Interpolate Euler angles using SLERP =====
    rotations = Rotation.from_euler('xyz', angles, degrees=True)
    slerp = Slerp(original_indices, rotations)
    interpolated_rotations = slerp(new_indices)
    interpolated_angles = interpolated_rotations.as_euler('xyz',
                                                          degrees=True)
    (a_lens, b_lens, c_lens, alphas,
     betas, gammas) = extract_cell_params(initial_orientations)

    # Interpolate each parameter
    interp_a = interp1d(original_indices, a_lens, kind='cubic')
    interp_b = interp1d(original_indices, b_lens, kind='cubic')
    interp_c = interp1d(original_indices, c_lens, kind='cubic')
    interp_alpha = interp1d(original_indices, alphas, kind='cubic')
    interp_beta = interp1d(original_indices, betas, kind='cubic')
    interp_gamma = interp1d(original_indices, gammas, kind='cubic')

    new_a_lens = interp_a(new_indices)
    new_b_lens = interp_b(new_indices)
    new_c_lens = interp_c(new_indices)
    new_alphas = interp_alpha(new_indices)
    new_betas = interp_beta(new_indices)
    new_gammas = interp_gamma(new_indices)

    # Build interpolated orientations
    interpolated_orientations = np.zeros((n_interpolated, 3, 3))

    for i in range(n_interpolated):
        # Create unrotated unit cell from interpolated parameters
        a_vec, b_vec, c_vec = unit_cell_to_vectors(
            new_a_lens[i], new_b_lens[i], new_c_lens[i],
            new_alphas[i], new_betas[i], new_gammas[i]
        )

        # Stack into matrix (rows are a, b, c)
        unit_cell = np.array([a_vec, b_vec, c_vec])

        # Apply interpolated rotation
        R = rotation_matrix_from_angles(
            interpolated_angles[i, 0],
            interpolated_angles[i, 1],
            interpolated_angles[i, 2],
            degrees=True
        )

        # Rotated unit cell
        interpolated_orientations[i] = (R @ unit_cell.T).T

    output = input_file.replace('.npz', '_interpolated.npz')
    np.savez(output,
             angles=interpolated_angles,
             initial_orientations=interpolated_orientations,
             original_file=original_file,
             subsamples=subsamples,
             indices=new_indices)


def extract_cell_params(orientations):

    """Extract a, b, c, alpha, beta, gamma from unit cell vectors"""
    n = len(orientations)

    a_lengths = np.linalg.norm(orientations[:, 0, :], axis=1)
    b_lengths = np.linalg.norm(orientations[:, 1, :], axis=1)
    c_lengths = np.linalg.norm(orientations[:, 2, :], axis=1)

    alphas = np.zeros(n)
    betas = np.zeros(n)
    gammas = np.zeros(n)

    for i in range(n):
        a = orientations[i, 0, :]
        b = orientations[i, 1, :]
        c = orientations[i, 2, :]

        # Normalize to get unit vectors
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        c_norm = c / np.linalg.norm(c)

        # Angles between vectors (in degrees)
        alphas[i] = np.degrees(np.arccos(np.clip(np.dot(b_norm, c_norm),
                                                 -1, 1)))
        betas[i] = np.degrees(np.arccos(np.clip(np.dot(a_norm, c_norm),
                                                -1, 1)))
        gammas[i] = np.degrees(np.arccos(np.clip(np.dot(a_norm, b_norm),
                                                 -1, 1)))

    return a_lengths, b_lengths, c_lengths, alphas, betas, gammas


def unit_cell_to_vectors(a, b, c, alpha, beta, gamma):
    """
    Convert unit cell parameters to vectors using PDB convention.
    Angles in degrees.
    """
    alpha_r = np.radians(alpha)
    beta_r = np.radians(beta)
    gamma_r = np.radians(gamma)

    # a along x
    a_vec = np.array([a, 0, 0])

    # b in xy plane
    b_vec = np.array([
        b * np.cos(gamma_r),
        b * np.sin(gamma_r),
        0
    ])

    # c fully determined
    cx = c * np.cos(beta_r)
    cy = c * (np.cos(alpha_r) - np.cos(beta_r) *
              np.cos(gamma_r)) / np.sin(gamma_r)
    cz = np.sqrt(c**2 - cx**2 - cy**2)
    c_vec = np.array([cx, cy, cz])

    return a_vec, b_vec, c_vec


def rotation_matrix_from_angles(alpha, beta, gamma, degrees=True):
    """Build rotation matrix from Euler angles"""
    if degrees:
        alpha, beta, gamma = np.deg2rad([alpha, beta, gamma])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])
    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    Rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx


if __name__ == '__main__':
    main()
