#!/usr/bin/env python3
import numpy as np
import argparse


def main():

    msg = 'Integrate the rocking curve '
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('rc_npz_file', type=str,
                        help='Input rocking curve npz file')
    parser.add_argument('-o', '--out', type=str, default='integrated.hkl',
                        help='Output hkl file')
    parser.add_argument('-s', '--scale', type=float,
                        default=1, help='Scaling factor')
    parser.add_argument('--cut',  type=float,
                        default=1.e-15, help='Trim values below this')

    args = parser.parse_args()

    rc_file = args.rc_npz_file
    scale = args.scale
    rc_dict, angles = get_dict(rc_file)

    max_val = 0
    with open(args.out, 'w') as f:
        for index in rc_dict.keys():
            H, K, L = index
            out_str = f"{H:>4d}{K:>4d}{L:>4d}"
            intensity = scale * rc_dict[index] * 100000
            if intensity > max_val:
                max_val = intensity
            error = intensity * 0.01

            if intensity > args.cut:
                out_str += f"{intensity:>10.2f} {error:>9.2f}\n"
                f.write(out_str)
    print(f"Written hkl file: {args.out}")
    if max_val > 1000000:
        print("WARNNING: Probably a problem with output format")
    print("MAX", max_val)


def plot_miller(miller, curve, angles):

    H, K, L = miller


def get_dict(rc_file):

    data = np.load(rc_file)
    hkl_indices = data['hkl_indices']

    curves = data['curves']
    angles = data['angles_deg']

    delta_phi = angles[1] - angles[0]

    rc = {}

    for hkl, curve in zip(hkl_indices, curves):
        H, K, L = hkl
        m = (int(H), int(K), int(L))
        rc[m] = np.sum(curve) * delta_phi

    return rc, angles


if __name__ == '__main__':
    main()
