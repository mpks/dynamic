from typing import List
from dxtbx.model.experiment_list import ExperimentListFactory
from scitbx import matrix
import numpy as np


def compute_kxyz(miller_list: list[tuple[int, int, int]],
                 expt_file: str,
                 exp_id: int):
    """
    Compute a sequence of positions in the reciprocal space from
    a sequence of Miller indices. The positions are computed in the
    crystal frame of reference (they do not depend on the goniometer
    orientation, or experiment setup).

    Parameters
    ----------
    miller_list : a list of Miller indices (3-int tuples or lists)
        A list of Miller indices for which to compute the positions
        in the reciprocal space.
    expt_file : str or Path
        The DIALS expt file.
    exp_id : integer
        The integer of the experiment for which to compute the
        reciprocal space position.

    Returns
    -------
    kxyz_list : a list of positions in the reciprocal space (kx, ky, kz).
        The order of positions is the same as the order of the input
        Miller indices.
    """

    experiments = ExperimentListFactory.from_json_file(expt_file)
    experiment = experiments[exp_id]

    crystal = experiment.crystal

    B = matrix.sqr(crystal.get_B())

    k_vecs = []

    for miller in miller_list:

        H, K, L = miller
        hkl = matrix.col((int(H), int(K), int(L)))
        k_vec = np.array(B * hkl)

        k_vecs.append(k_vec)

    return k_vecs
