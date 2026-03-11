#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import sys


def main():

    input_files = sys.argv[2:]
    sub_samples = int(sys.argv[1])

    for file in input_files:
        interpolate(file, subsamples=sub_samples)


def interpolate(input_file, subsamples=50):

    data = np.load(input_file)
    print(f"Interpolating for file: {input_file}")
    angles = data['angles']
    if 'original_file' in data:
        original_file = data['original_file']
    else:
        original_file = 'unknown'
    initial_orientations = data['initial_orientations']

    n_original = len(angles)
    n_interpolated = subsamples * (n_original - 1) + 1

    # Original and new frame indices
    original_indices = np.arange(n_original)
    new_indices = np.linspace(0, n_original - 1, n_interpolated)

    rotations = Rotation.from_euler('xyz', angles, degrees=True)
    slerp = Slerp(original_indices, rotations)
    interpolated_rotations = slerp(new_indices)
    interpolated_angles = interpolated_rotations.as_euler('xyz',
                                                          degrees=True)

    output = input_file.replace('.npz', '_interpolated.npz')
    np.savez(output,
             angles=interpolated_angles,
             initial_orientations=initial_orientations[0],
             original_file=original_file,
             subsamples=subsamples,
             indices=new_indices)


if __name__ == '__main__':
    main()
