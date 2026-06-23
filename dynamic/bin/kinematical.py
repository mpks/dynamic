#!/usr/bin/env python3
"""
kinematical_hkl.py — compute a kinematical HKL file from a CIF
using electron structure factors (gemmi StructureFactorCalculatorE).

For each Miller index the intensity is:

    I = |F(hkl)|^2

and sigma is a fraction of I (default 10%).

The output is a standard SHELX free-format HKL file:

    h  k  l  I  sigma

Miller indices are read from a text file, one per line, in the
same format as the rocking curve HKL files (space- or
comma-separated h k l).

Usage
-----
    python kinematical_hkl.py structure.cif millers.txt -o kin.hkl

    # Custom sigma fraction (5%)
    python kinematical_hkl.py structure.cif millers.txt \\
        -o kin.hkl --sigma_frac 0.05

    # Auto-generate all reflections up to d_min resolution
    python kinematical_hkl.py structure.cif \\
        --d_min 0.8 -o kin.hkl

    # Print summary table
    python kinematical_hkl.py structure.cif millers.txt \\
        -o kin.hkl --summary
"""

import argparse
import sys

import gemmi
import numpy as np


# ---------------------------------------------------------------------------
# CIF helpers (from your original module)
# ---------------------------------------------------------------------------

def _afloat(value):
    """Parse a CIF numeric value,
    stripping uncertainty e.g. 7.09(2).
    """
    return float(value.split('(')[0])


def load_unit_cell_from_cif(cif_file):
    """Return a gemmi.UnitCell from a CIF file."""
    doc = gemmi.cif.read_file(str(cif_file))
    block = doc.sole_block()
    a = _afloat(block.find_value('_cell_length_a'))
    b = _afloat(block.find_value('_cell_length_b'))
    c = _afloat(block.find_value('_cell_length_c'))
    alpha = _afloat(block.find_value('_cell_angle_alpha'))
    beta = _afloat(block.find_value('_cell_angle_beta'))
    gamma = _afloat(block.find_value('_cell_angle_gamma'))
    return gemmi.UnitCell(a, b, c, alpha, beta, gamma)


# ---------------------------------------------------------------------------
# Structure factor calculator
# ---------------------------------------------------------------------------

class StructureFactorCalculator:
    """
    Electron structure factor calculator using gemmi.

    Uses StructureFactorCalculatorE (electron scattering)
    rather than X-ray, which is correct for ED data.
    """

    def __init__(self, cif_file):
        self.cif_file = str(cif_file)
        self._st = None
        self._calc = None

    def _setup(self):
        if self._calc is not None:
            return
        st = gemmi.read_small_structure(self.cif_file)
        st.change_occupancies_to_crystallographic()
        self._st = st
        self._calc = gemmi.StructureFactorCalculatorE(st.cell)

    def structure_factor(self, hkl):
        """
        Return the complex structure factor F(hkl).

        Parameters
        ----------
        hkl : tuple of int  (h, k, l)

        Returns
        -------
        F : complex
        """
        self._setup()
        return self._calc.calculate_sf_from_small_structure(
            self._st, hkl
        )

    def intensity(self, hkl):
        """Return |F(hkl)|^2."""
        F = self.structure_factor(hkl)
        return abs(F) ** 2

    def resolution(self, hkl):
        """Return d-spacing in Angstrom for hkl."""
        self._setup()
        return self._st.cell.calculate_d(list(hkl))


# ---------------------------------------------------------------------------
# Miller index generation
# ---------------------------------------------------------------------------

def read_hkl_file(path):
    """
    Read Miller indices from a text file, one per line.
    Blank lines and # comments are ignored.
    Accepted formats: 'h k l' or 'h,k,l'.
    """
    result = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            line = line.replace(',', ' ')
            parts = line.split()
            H, K, L = int(parts[0]), int(parts[1]), int(parts[2])
            result.append((H, K, L))
    return result


def generate_hkl_to_resolution(cell, d_min):
    """
    Generate all Miller indices with d-spacing >= d_min.

    Parameters
    ----------
    cell : gemmi.UnitCell
    d_min : float  (Angstrom)

    Returns
    -------
    list of (h, k, l) tuples
    """
    # Estimate max indices from cell parameters
    h_max = int(np.ceil(cell.a / d_min)) + 1
    k_max = int(np.ceil(cell.b / d_min)) + 1
    l_max = int(np.ceil(cell.c / d_min)) + 1

    result = []
    for h in range(-h_max, h_max + 1):
        for k in range(-k_max, k_max + 1):
            for l in range(-l_max, l_max + 1):  # noqa: E741
                if h == 0 and k == 0 and l == 0:
                    continue
                d = cell.calculate_d([h, k, l])
                if d >= d_min:
                    result.append((h, k, l))
    return result


# ---------------------------------------------------------------------------
# HKL writer
# ---------------------------------------------------------------------------

def compute_and_write_hkl(
    cif_file,
    hkl_list,
    output_path,
    sigma_frac=0.10,
    scale=None,
    min_sigma=1.0,
):
    """
    Compute |F(hkl)|^2 for each Miller index and write a
    SHELX free-format HKL file.

    Parameters
    ----------
    cif_file : str
        Path to the CIF file.
    hkl_list : list of (h, k, l)
        Miller indices to compute.
    output_path : str
        Output .hkl file path.
    sigma_frac : float
        Sigma as a fraction of I.  Default 0.10 (10%).
    scale : float or None
        Scale factor applied to all intensities before writing.
        If None, auto-scale so the strongest reflection = 10000.
    min_sigma : float
        Minimum sigma value (absolute, after scaling).

    Returns
    -------
    results : list of (h, k, l, I, sigma)
        Computed values before writing.
    """
    calc = StructureFactorCalculator(cif_file)

    print(
        f"Computing |F|^2 for {len(hkl_list)} reflections…"
    )
    results = []
    for hkl in hkl_list:
        i_kin = calc.intensity(hkl)
        results.append((hkl[0], hkl[1], hkl[2], i_kin))

    # Auto-scale
    intensities = np.array([r[3] for r in results])
    if scale is None:
        i_max = intensities.max() if intensities.max() > 0 else 1.0
        scale = 10000.0 / i_max

    # Write
    with open(output_path, 'w') as fh:
        out = []
        for h, k, l, i_raw in results:
            i_scaled = i_raw * scale
            sigma = max(i_scaled * sigma_frac, min_sigma)
            fh.write(
                f"{h:4d}{k:4d}{l:4d}"
                f"{i_scaled:12.2f}"
                f"{sigma:12.2f}\n"
            )
            out.append((h, k, l, i_scaled, sigma))
        # SHELX terminator
        fh.write(
            f"{'0':>4}{'0':>4}{'0':>4}"
            f"{'0.00':>12}"
            f"{'0.00':>12}\n"
        )

    print(
        f"Written {len(results)} reflections to {output_path}"
    )
    print(f"  Scale factor: {scale:.4e}")
    print(f"  Sigma fraction: {sigma_frac*100:.1f}%")
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results, n_top=20):
    """Print the n_top strongest reflections."""
    sorted_r = sorted(
        results, key=lambda x: x[3], reverse=True
    )
    col_hkl = 16
    col_i = 14
    col_s = 14
    header = (
        f"{'h  k  l':>{col_hkl}}"
        f"{'I (scaled)':>{col_i}}"
        f"{'sigma':>{col_s}}"
    )
    sep = "-" * (col_hkl + col_i + col_s)
    print(f"\n--- Top {n_top} reflections by intensity ---")
    print(header)
    print(sep)
    for h, k, l, i_sc, sigma in sorted_r[:n_top]:
        hkl_str = f"({h:3d},{k:3d},{l:3d})"
        print(
            f"{hkl_str:>{col_hkl}}"
            f"{i_sc:>{col_i}.2f}"
            f"{sigma:>{col_s}.2f}"
        )
    print(sep)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compute kinematical HKL file from electron "
            "structure factors (gemmi)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "cif_file",
        help="Crystal structure CIF file",
    )
    p.add_argument(
        "hkl_file",
        nargs="?",
        default=None,
        help=(
            "Text file with Miller indices (h k l per line). "
            "Required unless --d_min is given."
        ),
    )
    p.add_argument(
        "-o", "--output",
        default="kinematical.hkl",
        help="Output HKL file path",
    )
    p.add_argument(
        "--d_min",
        type=float,
        default=None,
        help=(
            "If given, auto-generate all reflections with "
            "d >= d_min (Angstrom) instead of reading a file."
        ),
    )
    p.add_argument(
        "--sigma_frac",
        type=float,
        default=0.10,
        help="Sigma as fraction of I (e.g. 0.10 = 10%%)",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=None,
        help=(
            "Scale factor for intensities. "
            "Default: auto-scale so strongest = 10000."
        ),
    )
    p.add_argument(
        "--min_sigma",
        type=float,
        default=1.0,
        help="Minimum sigma value (after scaling)",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="Print a summary table of the top 20 reflections.",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    if args.hkl_file is None and args.d_min is None:
        sys.exit(
            "Provide either a Miller index file or --d_min."
        )

    if args.d_min is not None:
        cell = load_unit_cell_from_cif(args.cif_file)
        hkl_list = generate_hkl_to_resolution(
            cell, args.d_min
        )
        print(
            f"Generated {len(hkl_list)} reflections "
            f"with d >= {args.d_min} A"
        )
    else:
        hkl_list = read_hkl_file(args.hkl_file)
        print(
            f"Read {len(hkl_list)} Miller indices "
            f"from {args.hkl_file}"
        )

    results = compute_and_write_hkl(
        cif_file=args.cif_file,
        hkl_list=hkl_list,
        output_path=args.output,
        sigma_frac=args.sigma_frac,
        scale=args.scale,
        min_sigma=args.min_sigma,
    )

    if args.summary:
        print_summary(results)


if __name__ == "__main__":
    main()
