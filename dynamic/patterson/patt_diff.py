"""
patt_diff.py

Subtract two .patt files and write the signed difference as a new .patt file.

    result = obs - cal   (both normalised to [0,1] before subtraction)

The output is normalised to [0,1] so the viewer can display it.
Positive regions (obs > cal) appear as peaks; negative regions are clipped to 0
unless --signed is used (which shifts so that 0 maps to 0.5).

Usage:
    python patt_diff.py obs.patt cal.patt diff.patt
    python patt_diff.py obs.patt cal.patt diff.patt --signed
"""

import struct, sys, argparse
import numpy as np


def read_patt(path):
    with open(path, 'rb') as f:
        buf = f.read()
    dv = memoryview(buf)
    hdr_len = struct.unpack_from('<I', buf, 0)[0]
    hdr = buf[4:4+hdr_len].decode('ascii').strip().split('\n')
    if hdr[0].strip() != 'PATT1':
        raise ValueError(f"{path} is not a valid .patt file")
    nx, ny, nz = map(int, hdr[1].strip().split())
    label = hdr[2].strip() if len(hdr) > 2 else 'unknown'
    data = np.frombuffer(buf, dtype='<f4', offset=4+hdr_len,
                         count=nx*ny*nz).copy()
    return data.reshape(nx, ny, nz), label


def write_patt(grid, path, label):
    nx, ny, nz = grid.shape
    header = f"PATT1\n{nx} {ny} {nz}\n{label}\n"
    hdr_bytes = header.encode('ascii')
    pad = (-( 4 + len(hdr_bytes))) % 4
    hdr_bytes += b'\x00' * pad
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', len(hdr_bytes)))
        f.write(hdr_bytes)
        f.write(grid.astype('<f4').tobytes())
    mb = (4 + len(hdr_bytes) + nx*ny*nz*4) / 1e6
    print(f"Wrote {path}  ({nx}×{ny}×{nz}, {mb:.2f} MB)")


def main():
    p = argparse.ArgumentParser(description="Subtract two .patt files.")
    p.add_argument('obs',  help="Observed .patt file")
    p.add_argument('cal',  help="Calculated .patt file")
    p.add_argument('out',  nargs='?', default=None,
                   help="Output .patt file (default: diff.patt)")
    p.add_argument('--signed', action='store_true',
                   help="Keep signed difference: 0→0.5, positive→>0.5, negative→<0.5")
    p.add_argument('--no-prenorm', action='store_true',
                   help="Skip pre-normalisation (use if both maps already share a scale)")
    args = p.parse_args()

    if args.out is None:
        args.out = 'diff.patt'

    print(f"Reading {args.obs}...")
    obs, lbl_obs = read_patt(args.obs)
    print(f"Reading {args.cal}...")
    cal, lbl_cal = read_patt(args.cal)

    if obs.shape != cal.shape:
        sys.exit(f"Shape mismatch: {obs.shape} vs {cal.shape}. "
                 f"Recompute both maps with the same grid_size.")

    # Normalise each to [0,1] so they're on the same scale
    if not args.no_prenorm:
        obs = (obs - obs.min()) / (obs.max() - obs.min() + 1e-12)
        cal = (cal - cal.min()) / (cal.max() - cal.min() + 1e-12)

    diff = obs.astype(np.float32) - cal.astype(np.float32)

    if args.signed:
        # Map so that 0 → 0.5, full range fits [0,1]
        abs_max = np.abs(diff).max()
        if abs_max > 0:
            diff = diff / (2 * abs_max) + 0.5
        else:
            diff = diff + 0.5
        label = f"signed_diff ({lbl_obs}) - ({lbl_cal})"
        print(f"Signed mode: 0.5=no difference, >0.5=obs>cal, <0.5=obs<cal")
    else:
        # Clip negative, normalise positive to [0,1]
        diff = np.clip(diff, 0, None)
        mx = diff.max()
        if mx > 0:
            diff = diff / mx
        label = f"diff ({lbl_obs}) - ({lbl_cal})"
        print(f"Positive-only mode: only regions where obs > cal are shown")

    print(f"Diff range: [{diff.min():.3f}, {diff.max():.3f}]")
    write_patt(diff.astype(np.float32), args.out, label)


if __name__ == '__main__':
    main()
