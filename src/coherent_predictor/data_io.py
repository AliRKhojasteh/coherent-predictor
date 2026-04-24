"""Small helpers for loading the DNS trajectory files used in the paper.

The published datasets ship as MATLAB ``.mat`` files with a variable named
``P_RK4`` of shape ``(N_particles, N_snapshots, dim)``. This module exposes
a thin wrapper around ``scipy.io.loadmat`` plus a noise injector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import scipy.io as sio


def load_trajectories(
    path: Union[str, Path],
    variable: str = "P_RK4",
    dims: int = 2,
) -> np.ndarray:
    """Read a ``.mat`` or ``.npz`` file and return ``(N, T, dims)`` float64.

    Parameters
    ----------
    path
        Path to the data file. ``.mat`` (MATLAB v5) and ``.npz`` (numpy
        compressed) are both recognised. The extension is used to dispatch.
    variable
        Name of the array inside the file. Defaults to ``"P_RK4"``.
    dims
        Truncate the spatial dimension to the first ``dims`` columns. The
        2D HIT ``.mat`` file stores three columns with the third constant.
    """
    p = Path(path)
    if p.suffix.lower() == ".npz":
        archive = np.load(str(p))
        if variable not in archive.files:
            raise KeyError(
                f"Variable '{variable}' not found in {p}. "
                f"Available: {list(archive.files)}"
            )
        arr = archive[variable]
    else:
        mat = sio.loadmat(str(p))
        if variable not in mat:
            raise KeyError(
                f"Variable '{variable}' not found in {p}. "
                f"Available: {[k for k in mat if not k.startswith('__')]}"
            )
        arr = mat[variable]

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    return arr[:, :, :dims].astype(np.float64)


def add_positional_noise(
    positions: np.ndarray,
    noise_fraction: float = 0.10,
    seed: int = 123,
) -> tuple[np.ndarray, float]:
    """Add zero mean Gaussian noise proportional to the characteristic displacement.

    The characteristic displacement is the mean Euclidean step size between
    consecutive snapshots, which is close to the smallest resolvable length
    scale of the DNS. Returns both the noisy array and the noise sigma used,
    so downstream code can report SNRs consistently.
    """
    char_disp = float(
        np.mean(np.sqrt(np.sum(np.diff(positions, axis=1) ** 2, axis=2)))
    )
    sigma = noise_fraction * char_disp
    rng = np.random.default_rng(seed)
    return positions + rng.normal(0.0, sigma, size=positions.shape), sigma


def median_nn_distance(positions_snapshot: np.ndarray) -> float:
    """Median nearest neighbour distance at a single snapshot."""
    from sklearn.neighbors import KDTree

    tree = KDTree(positions_snapshot)
    dd, _ = tree.query(positions_snapshot, k=2)
    return float(np.median(dd[:, 1]))
