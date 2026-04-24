"""Backward finite time Lyapunov exponent and coherent neighbour weighting.

The rate of separation used throughout the paper is

    Lambda_j(t_e, T) = |log(d_j(t_e - T) / d_j(t_e))| / T

where ``d_j`` is the Euclidean distance between the target particle and a
neighbour ``j`` at a given snapshot. A neighbour is declared coherent when
``Lambda_j`` sits at or below a percentile threshold of the local
distribution.

Weights for the coherent cost function combine a Lambda term and a distance
term,

    w_j = (Lambda_j / <Lambda>)**(-1) + alpha_w * (d_j / max d)**(-1),

with a final L1 normalisation so sum_j w_j = 1. Larger ``alpha_w`` pulls the
weights towards the nearest neighbours; ``alpha_w = 3`` has worked well
across all datasets tested in the paper.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-15


def backward_ftle(
    positions: np.ndarray,
    target_idx: int,
    neighbour_ids: np.ndarray,
    t_e: int,
    T_int: int,
) -> np.ndarray:
    """Compute Lambda for each neighbour of a target particle.

    Parameters
    ----------
    positions
        Array shaped ``(N, T, d)``.
    target_idx
        Index of the target particle.
    neighbour_ids
        Integer array of candidate neighbour indices, not including the target.
    t_e
        Snapshot at which the target is evaluated.
    T_int
        Backward integration time in snapshot units. The past reference is
        ``max(t_e - T_int, 0)``.
    """
    t_past = max(t_e - T_int, 0)
    d_now = np.linalg.norm(
        positions[neighbour_ids, t_e] - positions[target_idx, t_e], axis=1
    )
    d_past = np.linalg.norm(
        positions[neighbour_ids, t_past] - positions[target_idx, t_past], axis=1
    )
    denom = max(t_e - t_past, 1)
    return np.abs(np.log((d_past + EPS) / (d_now + EPS))) / denom


def coherent_mask(lam: np.ndarray, percentile: float = 50.0) -> np.ndarray:
    """Boolean mask of neighbours whose Lambda is at or below ``percentile``.

    The threshold is taken from the local distribution of ``lam`` so the
    classifier adapts to the local turbulence regime.
    """
    if len(lam) == 0:
        return np.zeros(0, dtype=bool)
    thresh = np.percentile(lam, percentile)
    mask = lam <= thresh
    if mask.sum() < 2 and len(lam) >= 2:
        # Keep at least two neighbours so the weighted mean is defined.
        mask[:] = True
    return mask


def compute_weights(
    lam: np.ndarray, d_now: np.ndarray, alpha_w: float = 3.0
) -> np.ndarray:
    """L1 normalised coherent weights.

    See module docstring for the formula. ``lam`` and ``d_now`` must be the
    same length. Zeros are floored at ``EPS`` to avoid divide by zero.
    """
    lam_safe = np.maximum(lam, EPS)
    lam_mean = lam_safe.mean() + EPS
    r = d_now.max() + EPS

    w = (lam_safe / lam_mean) ** (-1) + alpha_w * (d_now / r) ** (-1)
    w = np.where(np.isfinite(w), w, 1.0)
    total = w.sum()
    if total <= 0:
        # Degenerate: fall back to uniform.
        return np.full_like(w, 1.0 / len(w))
    return w / total
