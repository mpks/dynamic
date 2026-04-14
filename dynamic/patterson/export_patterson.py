"""
export_patterson.py

Export a 3D Patterson map from a SpotsList to a compact binary .patt file
that can be loaded by the standalone HTML viewer.

File format (.patt):
    Header (ASCII, newline terminated):
        PATT1                   magic
        nx ny nz                grid dimensions
        label                   e.g. "paracetamol_obs"
    Body (little-endian float32):
        nx*ny*nz floats, C-order (i fastest, k slowest)
        i.e. index = k*ny*nx + j*nx + i

Usage:
    from export_patterson import export_patterson, export_patterson_pair
    from dynamic.spots import SpotsList

    # Single map
    spots = SpotsList.from_npz('dataset_001.npz')
    spots.compute_Fo_and_bulk_scale()
    export_patterson(spots, 'obs', grid_size=64, outfile='my_map_obs.patt')

    # Observed + calculated pair
    export_patterson_pair(spots, grid_size=64, prefix='my_map')
    # writes my_map_obs.patt and my_map_cal.patt
"""

import numpy as np
import struct
from typing import Literal


def compute_patterson_3d(spots_list,
                         intensity_attr: str = 'Fo_scaled',
                         grid_size: int = 64,
                         normalise: bool = True) -> np.ndarray:
    """
    Compute a 3D Patterson map from a SpotsList.

    P(u, v, w) = sum_{hkl} I_hkl * cos(2*pi*(h*u + k*v + l*w))

    Sampled on a regular grid of shape (grid_size, grid_size, grid_size)
    over fractional coordinates u,v,w in [0, 1).

    Parameters
    ----------
    spots_list     : SpotsList object (must have Fo_scaled and Fc computed)
    intensity_attr : 'Fo_scaled' for observed, 'Fc' for calculated
    grid_size      : N — output grid is (N, N, N)
    normalise      : normalise output to [0, 1]

    Returns
    -------
    np.ndarray of shape (grid_size, grid_size, grid_size), float32
    """
    N = grid_size

    # Grid coordinates in [0, 1)
    u = np.linspace(0, 1, N, endpoint=False, dtype=np.float32)
    v = np.linspace(0, 1, N, endpoint=False, dtype=np.float32)
    w = np.linspace(0, 1, N, endpoint=False, dtype=np.float32)

    patterson = np.zeros((N, N, N), dtype=np.float32)

    n_used = 0
    for spot in spots_list.spots:

        I = getattr(spot, intensity_attr, None)
        if I is None or np.isnan(I):
            continue
        if intensity_attr == 'Fo_scaled':
            coeff = float(I) ** 2
        else:
            coeff = float(I) ** 2   # Fc^2

        if coeff <= 0:
            continue

        h, k, l = spot.H, spot.K, spot.L

        # Vectorised: compute cos(2pi(hu + kv + lw)) on the full grid
        # phase_u[i] = h*u[i], phase_v[j] = k*v[j], phase_w[m] = l*w[m]
        phase_u = (2 * np.pi * h * u).astype(np.float32)   # (N,)
        phase_v = (2 * np.pi * k * v).astype(np.float32)   # (N,)
        phase_w = (2 * np.pi * l * w).astype(np.float32)   # (N,)

        # Outer sum: phase[i,j,m] = phase_u[i] + phase_v[j] + phase_w[m]
        cos_uvw = (np.cos(phase_u)[:, None, None] *
                   np.cos(phase_v)[None, :, None] *
                   np.cos(phase_w)[None, None, :] -
                   np.sin(phase_u)[:, None, None] *
                   np.sin(phase_v)[None, :, None] *
                   np.cos(phase_w)[None, None, :] -
                   np.sin(phase_u)[:, None, None] *
                   np.cos(phase_v)[None, :, None] *
                   np.sin(phase_w)[None, None, :] -
                   np.cos(phase_u)[:, None, None] *
                   np.sin(phase_v)[None, :, None] *
                   np.sin(phase_w)[None, None, :])
        # This is cos(phase_u + phase_v + phase_w) via angle addition

        patterson += coeff * cos_uvw
        n_used += 1

    print(f"  Used {n_used} reflections")

    if normalise and patterson.max() != patterson.min():
        mn, mx = patterson.min(), patterson.max()
        patterson = (patterson - mn) / (mx - mn)

    return patterson.astype(np.float32)


def write_patt_file(grid: np.ndarray,
                    outfile: str,
                    label: str = 'patterson') -> None:
    """
    Write a .patt binary file readable by patterson_viewer.html.

    Parameters
    ----------
    grid    : (nx, ny, nz) float32 array
    outfile : output filename (should end in .patt)
    label   : descriptive label embedded in header
    """

    nx, ny, nz = grid.shape
    header = f"PATT1\n{nx} {ny} {nz}\n{label}\n"
    header_bytes = header.encode('ascii')

    # Compute padding so (4 + header_len + padding) % 4 == 0
    pad = (- (4 + len(header_bytes))) % 4
    header_bytes_padded = header_bytes + b'\x00' * pad

    with open(outfile, 'wb') as f:
        # write padded length
        f.write(struct.pack('<I', len(header_bytes_padded)))
        f.write(header_bytes_padded)
        f.write(grid.astype('<f4').tobytes())
    size_mb = (len(header_bytes) + 4 + nx*ny*nz*4) / 1e6
    print(f"Wrote {outfile}  ({nx}x{ny}x{nz}, {size_mb:.2f} MB)")

    # nx, ny, nz = grid.shape
    # header = f"PATT1\n{nx} {ny} {nz}\n{label}\n"
    # header_bytes = header.encode('ascii')

    # with open(outfile, 'wb') as f:
    #    Write header length as uint32, then header, then data
    #    f.write(struct.pack('<I', len(header_bytes)))
    #    f.write(header_bytes)

    #    little-endian float32, C-order
    #    f.write(grid.astype('<f4').tobytes())

    # size_mb = (len(header_bytes) + 4 + nx*ny*nz*4) / 1e6
    # print(f"Wrote {outfile}  ({nx}x{ny}x{nz}, {size_mb:.2f} MB)")


def export_patterson(spots_list,
                     mode: Literal['obs', 'cal'] = 'obs',
                     grid_size: int = 64,
                     outfile: str = None,
                     normalise: bool = True) -> str:
    """
    Compute and export a single Patterson map.

    Parameters
    ----------
    spots_list : SpotsList (Fo_scaled and Fc must be populated)
    mode       : 'obs' uses Fo_scaled^2, 'cal' uses Fc^2
    grid_size  : grid size N (output is NxNxN)
    outfile    : output path; auto-generated if None
    normalise  : normalise to [0,1]

    Returns
    -------
    outfile path
    """
    if outfile is None:
        outfile = f"{spots_list.output_prefix}_patterson_{mode}_{grid_size}.patt"

    attr = 'Fo_scaled' if mode == 'obs' else 'Fc'
    label = f"{spots_list.output_prefix}_{mode}"

    print(f"Computing {mode} Patterson map (grid={grid_size})...")
    grid = compute_patterson_3d(spots_list, intensity_attr=attr,
                                grid_size=grid_size, normalise=normalise)
    write_patt_file(grid, outfile, label=label)
    return outfile


def export_patterson_pair(spots_list,
                          grid_size: int = 64,
                          prefix: str = None,
                          normalise: bool = True):
    """
    Export both observed and calculated Patterson maps.
    Writes <prefix>_obs.patt and <prefix>_cal.patt.
    """
    if prefix is None:
        prefix = f"{spots_list.output_prefix}_patterson_{grid_size}"

    export_patterson(spots_list, mode='obs', grid_size=grid_size,
                     outfile=f"{prefix}_obs.patt", normalise=normalise)
    export_patterson(spots_list, mode='cal', grid_size=grid_size,
                     outfile=f"{prefix}_cal.patt", normalise=normalise)
    print(f"Done. Open patterson_viewer.html and load these files.")


def export_diff_patterson(spots_list,
                          grid_size: int = 64,
                          outfile: str = None) -> str:
    """
    Export a difference Patterson map (obs - cal), normalised to [0,1].
    """
    if outfile is None:
        outfile = f"{spots_list.output_prefix}_patterson_diff_{grid_size}.patt"

    print("Computing difference Patterson map...")
    grid_obs = compute_patterson_3d(spots_list, intensity_attr='Fo_scaled',
                                    grid_size=grid_size, normalise=False)
    grid_cal = compute_patterson_3d(spots_list, intensity_attr='Fc',
                                    grid_size=grid_size, normalise=False)
    diff = grid_obs - grid_cal
    mn, mx = diff.min(), diff.max()
    if mx != mn:
        diff = (diff - mn) / (mx - mn)
    write_patt_file(diff.astype(np.float32), outfile,
                    label=f"{spots_list.output_prefix}_diff")
    return outfile


# ── Quick test with synthetic data ────────────────────────────────────
if __name__ == '__main__':
    import sys

    # Generate a synthetic 64^3 Patterson map for testing
    N = 64
    grid = np.zeros((N, N, N), dtype=np.float32)

    def gauss3d(cx, cy, cz, sx, grid):
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    dx, dy, dz = i/N-cx, j/N-cy, k/N-cz
                    grid[i,j,k] += np.exp(-(dx*dx+dy*dy+dz*dz)/(2*sx*sx))

    print("Generating synthetic Patterson map...")
    peaks = [(0.5,0.5,0.5,0.04), (0.6,0.55,0.5,0.03),
             (0.4,0.45,0.5,0.03), (0.55,0.4,0.6,0.025)]
    for cx,cy,cz,s in peaks:
        gauss3d(cx,cy,cz,s,grid)

    mn, mx = grid.min(), grid.max()
    grid = (grid - mn) / (mx - mn)
    write_patt_file(grid, 'test_patterson.patt', label='synthetic_test')
    print("Done. Open patterson_viewer.html and load test_patterson.patt")
