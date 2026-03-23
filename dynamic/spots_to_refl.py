"""
Utilities for writing SpotsList intensities back into a DIALS .refl file.
"""
from dials.array_family import flex
from typing import Literal
import numpy as np

from dynamic.spots import SpotsList


def write_spots_to_refl(spots: SpotsList,
                        refl_file: str,
                        output_file: str,
                        mode: Literal["sum", "prf"] = "prf",
                        unmatched_policy: Literal["keep", 
                                                  "zero", 
                                                  "warn"] = "keep",
                        intensity_mode: Literal["obs", 
                                                "cal", 
                                                "fit"] = "fit",) -> dict:
    """
    Update intensity values in a DIALS reflection table from a list of
    SpotsList objects (one per experiment index) and write to a new file.

    Parameters
    ----------
    spots : SpotsList

        One SpotsList contains the spots for a single experiment.
        The function assumes there is only one experiment in the
        experiment list.

    refl_file : str

        Path to the input cluster_1.refl file.

    output_file : str

        Path to write the modified reflection table.

    mode : "sum" or "prf"

        Which intensity column to modify:
        "prf" -> intensity.prf.value / intensity.prf.variance
        "sum" -> intensity.sum.value / intensity.sum.variance

    unmatched_policy : "keep", "zero", or "warn"

        What to do with reflections in the refl file that have no matching
        spot in the SpotsList:
        "keep" -> leave the original value unchanged (default)
        "zero" -> set intensity to 0.0 and variance to 1.0
        "warn" -> same as keep but print a warning for each unmatched refl

    intensity_mode : "obs", "cal", "fit"

        If "obs" it will save the observed intensity.
        If "cal" it will save scaled kinematic intensity.
        If "fit" it will save scaled corrected intensity.

    Returns
    -------
    stats : dict

        Summary counts: total, matched, unmatched, skipped_exps.

    Notes
    -----
    This function is written to modify integrated.refl files for individual
    (single) experiments, before they are grouped using clustering. So
    the assumption is that the same SpotList is previously read from
    this refl file, we performed ML fitting on the intensites, and now
    we want to overwrite one of the fields with the fitted data.
    """

    if mode not in ("sum", "prf"):
        raise ValueError(f"mode must be 'sum' or 'prf', got '{mode}'")

    refl = flex.reflection_table.from_file(refl_file)

    val_key = f"intensity.{mode}.value"
    var_key = f"intensity.{mode}.variance"

    if val_key not in refl:
        raise KeyError(
            f"Column '{val_key}' not found in {refl_file}. "
            f"Available columns: {list(refl.keys())}"
        )

    # Extract columns as mutable numpy arrays, edit, then push back
    values = np.array(refl[val_key])
    variances = np.array(refl[var_key])
    exp_ids = np.array(refl["id"])
    millers = refl["miller_index"]  # flex array of (H, K, L) tuples

    stats = {"total": len(refl), "matched": 0, "unmatched": 0}

    exp_id = 0

    spot_lookup: dict = {}
    for spot in spots:
        key = (spot.H, spot.K, spot.L)
        spot_lookup[key] = spot

    if not spot_lookup:
        print(f"[write_spots_to_refl] WARNING: SpotsList for exp_id="
              f"{exp_id} is empty, skipping.")

    # Find all refl-table rows belonging to this experiment
    exp_mask = exp_ids == exp_id
    exp_indices = np.where(exp_mask)[0]

    for row_idx in exp_indices:
        H, K, L = millers[row_idx]
        key = (int(H), int(K), int(L))

        if key in spot_lookup:
            spot = spot_lookup[key]
            if intensity_mode == 'obs':
                values[row_idx] = spot.intensity
            elif intensity_mode == 'cal':
                values[row_idx] = (spot.Fc * spots.global_scale)**2
            elif intensity_mode == 'fit':
                values[row_idx] = (spot.Fo_corrected *
                                   spots.global_scale)**2
            else:
                msg = f"Unknown intensity_mode: {intensity_mode}\n"
                msg += "Use either 'obs', 'cal', or 'fit'"
                raise ValueError(msg)

            variances[row_idx] = spot.sigma
            stats["matched"] += 1
        else:
            stats["unmatched"] += 1
            if unmatched_policy == "zero":
                values[row_idx] = 0.0
                variances[row_idx] = 1.0
            elif unmatched_policy == "warn":
                print(
                    f"[write_spots_to_refl] No match for "
                    f"({H:+d},{K:+d},{L:+d}) in exp_id={exp_id}"
                )
            # "keep": do nothing, original value remains

    # Push modified arrays back into the flex table
    refl[val_key] = flex.double(values)
    refl[var_key] = flex.double(variances)

    refl.as_file(output_file)

    print(f"[write_spots_to_refl] Written to: {output_file}")
    print(f"  Mode          : {mode}")
    print(f"  Total rows    : {stats['total']}")
    print(f"  Matched       : {stats['matched']}")
    print(f"  Unmatched     : {stats['unmatched']}")

    return stats
