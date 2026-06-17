from scipy.spatial.transform import Rotation


def slerp_R(R1, R2, t):
    """
    Interpolate between rotation matrices R1 (t=0) and R2 (t=1).
    t may be outside [0, 1] for extrapolation.

    Uses the geodesic:  R(t) = (R2 * R1^T)^t * R1
    The fractional power is computed via axis-angle decomposition.
    """
    M = R2 @ R1.T
    rv = _mat_to_rotvec(M)   # axis * angle
    M_t = _rotvec_to_mat(rv * t)
    return M_t @ R1


def _mat_to_rotvec(R):
    """Convert 3x3 rotation matrix (numpy) to rotation vector."""
    return Rotation.from_matrix(R).as_rotvec()


def _rotvec_to_mat(rv):
    """Convert rotation vector to 3x3 rotation matrix (numpy)."""
    return Rotation.from_rotvec(rv).as_matrix()


def _parse_hkl_list(hkl_strings):
    """Parse 'h,k,l' strings into (h, k, l) tuples."""
    if not hkl_strings:
        return []
    result = []
    for s in hkl_strings:
        parts = s.split(",")
        result.append(tuple(int(x) for x in parts))
    return result


def _read_hkl_file(path):
    """
    Read Miller indices from a text file, one per line.
    Blank lines and lines starting with # are ignored.
    Accepted formats per line:
        h k l        (space-separated)
        h,k,l        (comma-separated)
        h, k, l      (comma+space)
    Returns a list of (h, k, l) tuples.
    """
    result = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # normalise separators
            line = line.replace(",", " ")
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Cannot parse HKL line: {line!r}"
                )
            result.append(tuple(int(x) for x in parts))
    return result
