#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse


def main():

    msg = 'Module used to plot rocking curves in the npz files'
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('npz', type=str,
                        help='NPZ file with rocking curves')
    parser.add_argument("--miller", type=triple_ints,
                        metavar="H,K,L",
                        help="Miller indices separated by commas")

    parser.add_argument('--scale', type=float, default=None,
                        help='Scale for the y-axis')
    parser.add_argument('--xmin', type=float, default=None,
                        help='x lim min')
    parser.add_argument('--xmax', type=float, default=None,
                        help='x lim min')
    args = parser.parse_args()

    xs, rc_dict = get_miller_index(args.npz)

    if args.miller in rc_dict:
        ys = rc_dict[args.miller]

        H, K, L = args.miller

        out_str = f'rocking_curve_{H:+03d}_{K:03d}_{L:03d}.png'
        label = f'{H} {K} {L}'

        fig = plt.figure(figsize=(3.375, 3.0))
        ax = fig.add_axes([0.16, 0.15, 0.79, 0.80])

        ax.plot(xs, ys, marker='o', ms=1, mew=0, lw=0, c='C3',
                label=label)

        plt.legend(loc=(0.1, 0.8), labelspacing=0.5, borderpad=0,
                   columnspacing=3.5, handletextpad=0.4, fontsize=9,
                   handlelength=2.0, handleheight=0.7,
                   frameon=False)

        print("RC Scale", ys.min(), ys.max())

        if args.scale:
            ax.set_ylim([-1.e-10, args.scale])
        ax.ticklabel_format(style='sci', axis='y',
                            scilimits=(0, 0))
        text = ax.yaxis
        text.OFFSETTEXTPAD = -10
        t = text.get_offset_text()
        t.set_position((0.1, 0))

        if args.xmin and args.xmax:
            ax.set_xlim(args.xmin, args.xmax)

        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Intensity (a.u.)')
        plt.savefig(out_str, dpi=400)
    else:
        print("Miller index not in the rocking curve file. No plot")


def triple_ints(value):
    try:
        parts = value.split(",")
        if len(parts) != 3:
            raise ValueError
        return tuple(int(x) for x in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Expected three integers separated by commas, e.g. 1,2,-1"
        )


def get_miller_index(npz_file):

    data = np.load(npz_file)

    angles_deg = data['angles_deg']

    hkls = data['hkl_indices']
    rcs = data['curves']

    out_dict = {}

    for hkl, rc in zip(hkls, rcs):
        H, K, L = hkl
        mm = (int(H), int(K), int(L))
        out_dict[mm] = rc

    return angles_deg, out_dict


if __name__ == '__main__':
    main()
