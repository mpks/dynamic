"""
patterson.py

Compute 2D Patterson maps (per image) from SpotsList objects.
Maps are computed in fractional coordinates (h, k) for each image/frame.

For each image we compute:

    P_obs(u, v) = sum_{hk on image} Fo_scaled^2 * cos(2*pi*(h*u + k*v))
    P_cal(u, v) = sum_{hk on image} Fc^2        * cos(2*pi*(h*u + k*v))

The map is sampled on a regular (N x N) grid over u, v in [0, 1).
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class PattersonMap:
    """Container for a single 2D Patterson map."""
    map_obs:    np.ndarray      # shape (N, N) — from Fo_scaled
    map_cal:    np.ndarray      # shape (N, N) — from Fc
    image_id:   int             # frame index (spot.z)
    n_spots:    int             # number of spots contributing
    grid_size:  int             # N
    dataset_id: str             # output_prefix of the parent SpotsList

    @property
    def deformation(self) -> np.ndarray:
        """
        Pixel-wise ratio map_obs / map_cal.
        This is the quantity the CNN is trained to predict.
        Clipped to avoid division by zero.
        """
        denom = np.where(np.abs(self.map_cal) > 1e-6,
                         self.map_cal,
                         1e-6 * np.sign(self.map_cal + 1e-12))
        return self.map_obs / denom


def compute_patterson_2d(spots,
                         grid_size: int = 64,
                         use_fo: str = 'Fo_scaled',
                         normalise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a 2D Patterson map from a list of Spot objects on a single image.

    P(u, v) = sum_{hkl} I_hkl * cos(2*pi*(h*u + k*v))

    Only the (h, k) indices are used — this is a projection Patterson
    onto the ab plane, which is appropriate for per-image computation
    where l information is incomplete.

    Parameters
    ----------
    spots       : list of Spot objects (all from the same image/frame)
    grid_size   : number of grid points along each axis
    use_fo      : which intensity attribute to use for observed map
                  ('Fo_scaled', 'intensity', 'Fo')
    normalise   : if True, normalise each map to [-1, 1]

    Returns
    -------
    map_obs : np.ndarray, shape (grid_size, grid_size)
    map_cal : np.ndarray, shape (grid_size, grid_size)
    """
    N = grid_size

    # Grid coordinates u, v in [0, 1)
    u = np.linspace(0, 1, N, endpoint=False)
    v = np.linspace(0, 1, N, endpoint=False)
    uu, vv = np.meshgrid(u, v)   # shape (N, N)

    map_obs = np.zeros((N, N), dtype=np.float32)
    map_cal = np.zeros((N, N), dtype=np.float32)

    n_used = 0
    for spot in spots:

        # Get observed intensity
        Fo = getattr(spot, use_fo, None)
        if Fo is None or np.isnan(Fo):
            continue

        # Get calculated structure factor
        Fc = spot.Fc
        if Fc is None or np.isnan(Fc):
            continue

        h = spot.H
        k = spot.K

        # Patterson coefficient = |F|^2
        I_obs = Fo ** 2
        I_cal = Fc ** 2

        # cos term — shape (N, N)
        phase = 2.0 * np.pi * (h * uu + k * vv)
        cos_term = np.cos(phase)

        map_obs += I_obs * cos_term
        map_cal += I_cal * cos_term
        n_used  += 1

    if normalise and n_used > 0:
        # Normalise to [-1, 1] independently
        def norm(m):
            vmax = np.abs(m).max()
            if vmax > 1e-9:
                return m / vmax
            return m
        map_obs = norm(map_obs)
        map_cal = norm(map_cal)

    return map_obs, map_cal


def compute_patterson_maps_for_dataset(spots_list,
                                       grid_size: int = 64,
                                       min_spots_per_image: int = 5,
                                       normalise: bool = True,
                                       verbose: bool = True
                                       ) -> List[PattersonMap]:
    """
    Compute per-image Patterson maps for an entire SpotsList.

    Parameters
    ----------
    spots_list          : SpotsList object
    grid_size           : Patterson map grid size (N x N)
    min_spots_per_image : skip images with fewer spots than this
    normalise           : normalise each map to [-1, 1]
    verbose             : print progress

    Returns
    -------
    List of PattersonMap objects, one per image that passes the
    min_spots_per_image threshold.
    """
    groups = spots_list.group_by_image()
    patterson_maps = []

    for image_id, image_spots in groups.items():

        n_spots = len(image_spots)
        if n_spots < min_spots_per_image:
            if verbose:
                print(f"  Skipping image {image_id} — only {n_spots} spots")
            continue

        map_obs, map_cal = compute_patterson_2d(
            image_spots.spots,
            grid_size=grid_size,
            normalise=normalise
        )

        pm = PattersonMap(
            map_obs=map_obs,
            map_cal=map_cal,
            image_id=image_id,
            n_spots=n_spots,
            grid_size=grid_size,
            dataset_id=spots_list.output_prefix
        )
        patterson_maps.append(pm)

        if verbose:
            print(f"  Image {image_id:04d}: {n_spots} spots, "
                  f"obs range [{map_obs.min():.3f}, {map_obs.max():.3f}], "
                  f"cal range [{map_cal.min():.3f}, {map_cal.max():.3f}]")

    if verbose:
        print(f"Computed {len(patterson_maps)} Patterson maps "
              f"for dataset {spots_list.output_prefix}")

    return patterson_maps


def compute_patterson_maps_for_datasets(spots_lists: list,
                                        grid_size: int = 64,
                                        min_spots_per_image: int = 5,
                                        normalise: bool = True,
                                        verbose: bool = True
                                        ) -> List[PattersonMap]:
    """
    Compute per-image Patterson maps across multiple SpotsList objects.
    Convenience wrapper around compute_patterson_maps_for_dataset.
    """
    all_maps = []
    for idx, spots_list in enumerate(spots_lists):
        if verbose:
            print(f"Dataset {idx+1}/{len(spots_lists)}: "
                  f"{spots_list.output_prefix}")
        maps = compute_patterson_maps_for_dataset(
            spots_list,
            grid_size=grid_size,
            min_spots_per_image=min_spots_per_image,
            normalise=normalise,
            verbose=verbose
        )
        all_maps.extend(maps)
    return all_maps


def compute_dataset_conditioning(spots_list) -> np.ndarray:
    """
    Compute a conditioning vector for a SpotsList that encodes
    global dataset properties related to dynamical effect strength.

    These are used to condition the CNN so it knows how strong
    the dynamical effects are in this dataset.

    Returns
    -------
    np.ndarray of shape (8,) with the following entries:
        [0] mean_I_over_sigma   — signal to noise ratio
        [1] CV                  — coefficient of variation of intensities
        [2] dynamic_range       — 95th / 5th percentile ratio
        [3] log_mean_intensity  — log of mean intensity
        [4] fraction_weak       — fraction of spots with Fo < median/2
        [5] fraction_strong     — fraction of spots with Fo > 2*median
        [6] mean_resolution     — mean resolution of spots
        [7] n_spots_normalised  — number of spots / 1000
    """
    spots = [s for s in spots_list.spots
             if s.intensity > 0 and s.Fo_scaled is not None]

    if len(spots) == 0:
        return np.zeros(8, dtype=np.float32)

    intensities = np.array([s.intensity  for s in spots])
    Fo_scaled   = np.array([s.Fo_scaled  for s in spots])
    sigmas      = np.array([s.sigma      for s in spots])
    resolutions = np.array([s.resolution for s in spots
                             if s.resolution is not None])

    mean_I    = intensities.mean()
    std_I     = intensities.std()
    median_Fo = np.median(Fo_scaled)

    mean_I_over_sigma = np.mean(intensities / (sigmas + 1e-6))
    CV                = std_I / (mean_I + 1e-6)
    p95               = np.percentile(intensities, 95)
    p05               = np.percentile(intensities,  5)
    dynamic_range     = p95 / (p05 + 1e-6)
    log_mean          = np.log(mean_I + 1e-6)
    fraction_weak     = np.mean(Fo_scaled < median_Fo / 2.0)
    fraction_strong   = np.mean(Fo_scaled > median_Fo * 2.0)
    mean_res          = resolutions.mean() if len(resolutions) > 0 else 0.0
    n_spots_norm      = len(spots) / 1000.0

    conditioning = np.array([
        mean_I_over_sigma,
        CV,
        dynamic_range,
        log_mean,
        fraction_weak,
        fraction_strong,
        mean_res,
        n_spots_norm
    ], dtype=np.float32)

    return conditioning


def plot_patterson_pair(pm: PattersonMap,
                        filename: Optional[str] = None):
    """
    Plot observed, calculated, and deformation Patterson maps side by side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    def show(ax, data, title, cmap='RdBu_r'):
        vmax = np.abs(data).max()
        im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax,
                       origin='lower', extent=[0, 1, 0, 1])
        ax.set_title(title)
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    show(axes[0], pm.map_obs, f'Patterson obs\nimage {pm.image_id}')
    show(axes[1], pm.map_cal, f'Patterson cal\nimage {pm.image_id}')
    show(axes[2], pm.deformation, 'Deformation\n(obs/cal)', cmap='seismic')

    plt.suptitle(f"Dataset: {pm.dataset_id}, "
                 f"n_spots: {pm.n_spots}", y=1.02)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"Saved to {filename}")
    else:
        plt.show()
