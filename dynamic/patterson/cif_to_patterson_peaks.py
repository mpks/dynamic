"""
cif_to_patterson_peaks.py

Compute all interatomic Patterson vectors from a CIF file and export
them as a JSON file loadable by patterson_viewer.html.

Each Patterson peak is a vector u = r_j - r_i (mod 1) between any two
atoms in the unit cell (including symmetry-expanded copies).

Output JSON format (list of dicts):
    {
        "atom1":      "C1",
        "atom2":      "O1",
        "u":          0.513,    # fractional coord in [0, 1)
        "v":          0.487,
        "w":          0.502,
        "distance":   1.375,    # Angstrom (real-space distance)
        "involves_H": false,    # true if either atom is hydrogen
        "label":      "C1-O1"  # display label
    }

The origin peak (u=v=w=0.5 in viewer coords) is included with label "origin".

Usage:
    python cif_to_patterson_peaks.py paracetamol.cif paracetamol_peaks.json
    python cif_to_patterson_peaks.py paracetamol.cif --no-HH --max-dist 5.0

    # or from Python:
    from cif_to_patterson_peaks import cif_to_peaks
    peaks = cif_to_peaks("paracetamol.cif")
"""

import json
import argparse
import sys
from itertools import product as iproduct
from typing import Optional

try:
    import gemmi
except ImportError:
    sys.exit("gemmi is required: pip install gemmi")

import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────

def frac_to_cart(frac_vec, cell: gemmi.UnitCell) -> np.ndarray:
    """Convert fractional vector to Cartesian (Angstrom)."""
    pos = gemmi.Fractional(*frac_vec)
    cart = cell.orthogonalize(pos)
    return np.array([cart.x, cart.y, cart.z])


def wrap01(x: float) -> float:
    """Wrap to [0, 1)."""
    return x % 1.0


# ── Main function ─────────────────────────────────────────────────────────

def cif_to_peaks(cif_file: str,
                 include_HH: bool = False,
                 max_distance: Optional[float] = None,
                 min_distance: float = 0.01
                 ) -> list:
    """
    Compute Patterson peaks from a CIF file.

    Parameters
    ----------
    cif_file     : path to the .cif file
    include_HH   : if False, skip H-H vectors (very weak, clutter)
    max_distance : only include vectors shorter than this (Å); None = all
    min_distance : skip vectors shorter than this (Å); filters near-zero

    Returns
    -------
    list of peak dicts (see module docstring)
    """

    # gemmi small-molecule reader — works for CSD-style CIFs
    doc = gemmi.cif.read(cif_file)
    block = doc.sole_block()
    cs = gemmi.make_small_structure_from_block(block)
    cell = cs.cell

    # Build label → element map from ASU
    label_to_elem = {}
    for site in cs.sites:
        label_to_elem[site.label] = site.element.name

    # get_all_unit_cell_sites() expands ASU through all symmetry ops
    all_unit_cell = cs.get_all_unit_cell_sites()

    all_atoms = []  # (label, element, fx, fy, fz)
    for site in all_unit_cell:
        fx = wrap01(site.fract.x)
        fy = wrap01(site.fract.y)
        fz = wrap01(site.fract.z)
        label = site.label
        elem = label_to_elem.get(label, site.element.name)
        all_atoms.append((label, elem, fx, fy, fz))

    n = len(all_atoms)
    print(f"  {n} atoms in unit cell (after symmetry expansion)")
    if n == 0:
        raise ValueError("No atoms found in CIF. Check the file format.")

    # ── Compute all pairwise difference vectors ───────────────────────────
    peaks = []

    # Origin peak — stored as 0.5,0.5,0.5 so viewer shift puts it at (0,0,0)
    peaks.append({
        "atom1": "origin",
        "atom2": "origin",
        "u": 0.5, "v": 0.5, "w": 0.5,
        "distance": 0.0,
        "involves_H": False,
        "label": "origin"
    })

    seen_vectors = set()

    for i in range(n):
        lbl_i, elem_i, xi, yi, zi = all_atoms[i]
        for j in range(n):
            if i == j:
                continue
            lbl_j, elem_j, xj, yj, zj = all_atoms[j]

            # Skip H-H if not requested
            if not include_HH and elem_i == 'H' and elem_j == 'H':
                continue

            du = wrap01(xj - xi)
            dv = wrap01(yj - yi)
            dw = wrap01(zj - zi)

            # u and -u give the same Patterson peak — deduplicate
            neg_du = wrap01(-du)
            neg_dv = wrap01(-dv)
            neg_dw = wrap01(-dw)
            key_pos = (round(du, 4), round(dv, 4), round(dw, 4))
            key_neg = (round(neg_du, 4), round(neg_dv, 4), round(neg_dw, 4))
            canon = min(key_pos, key_neg)
            if canon in seen_vectors:
                continue
            seen_vectors.add(canon)

            # Shortest-image distance (check periodic images ±1)
            best_d = None
            for dx, dy, dz in iproduct([-1, 0, 1], repeat=3):
                v = frac_to_cart((du + dx, dv + dy, dw + dz), cell)
                d = float(np.linalg.norm(v))
                if best_d is None or d < best_d:
                    best_d = d

            if best_d < min_distance:
                continue
            if max_distance is not None and best_d > max_distance:
                continue

            involves_H = (elem_i == 'H' or elem_j == 'H')
            label = f"{lbl_i}-{lbl_j}"

            peaks.append({
                "atom1": lbl_i,
                "atom2": lbl_j,
                "u": round(du, 5),
                "v": round(dv, 5),
                "w": round(dw, 5),
                "distance": round(best_d, 4),
                "involves_H": involves_H,
                "label": label
            })

    peaks.sort(key=lambda p: p["distance"])
    print(f"  {len(peaks)} unique Patterson vectors (including origin)")
    return peaks


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export Patterson peaks from a CIF file to JSON.")
    parser.add_argument("cif_file", help="Input CIF file")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output JSON file (default: <cif_stem>_peaks.json)")
    parser.add_argument("--no-HH", dest="no_HH", action="store_true",
                        help="Skip hydrogen-hydrogen vectors")
    parser.add_argument("--max-dist", type=float, default=None,
                        help="Maximum vector distance in Å to include")
    parser.add_argument("--min-dist", type=float, default=0.01,
                        help="Minimum vector distance in Å (default: 0.01)")
    args = parser.parse_args()

    if args.output is None:
        stem = args.cif_file.rsplit('.', 1)[0]
        args.output = stem + "_peaks.json"

    print(f"Reading {args.cif_file}...")
    peaks = cif_to_peaks(
        args.cif_file,
        include_HH=not args.no_HH,
        max_distance=args.max_dist,
        min_distance=args.min_dist,
    )

    with open(args.output, 'w') as f:
        json.dump(peaks, f, indent=2)
    print(f"Wrote {len(peaks)} peaks to {args.output}")


if __name__ == "__main__":
    main()
