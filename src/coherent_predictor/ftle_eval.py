"""Diagnostic tools for the integration time ``T`` of the rate of separation.

These helpers reproduce the analysis in Section 3 of the paper where the
stretching factor

    sigma_j(t_e, T) = |log(d_j(t_e - T) / d_j(t_e))|

is accumulated as a function of ``T`` for a fixed snapshot ``t_e``. Three
things are then computed:

    median_sigma(T)                  central tendency of the distribution
    coherent_fraction(T, sigma_star) share of neighbours below threshold
    alpha_optimisation(T, alpha)     coherent velocity error as a function
                                     of (integration time, weight exponent)

The core predictor uses the same backward FTLE logic (see ``ftle.py``); this
module wraps it into the grid scans used to pick ``T_ftle = 8`` and
``alpha_w = 3``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.neighbors import KDTree

from .ftle import EPS


def sigma_field(
    positions: np.ndarray,
    t_e: int,
    T_values: Iterable[int],
    target_ids: np.ndarray,
    r_search: float,
) -> dict[int, np.ndarray]:
    """Compute per-neighbour ``sigma`` at snapshot ``t_e`` for each T.

    Parameters
    ----------
    positions
        Array ``(N, T_steps, d)``.
    t_e
        Evaluation snapshot.
    T_values
        Iterable of integer integration times to scan.
    target_ids
        Target particle indices to average over.
    r_search
        Radius passed to the KD tree query at ``t_e``.

    Returns
    -------
    dict mapping ``T -> np.ndarray`` of pooled sigma values (across all
    targets and their neighbours).
    """
    tree = KDTree(positions[:, t_e])
    out: dict[int, list] = {int(T): [] for T in T_values}

    for pid in target_ids:
        nids = tree.query_radius(
            positions[pid, t_e].reshape(1, -1), r=r_search
        )[0]
        nids = np.array([n for n in nids if n != pid])
        if len(nids) < 2:
            continue
        d_now = np.linalg.norm(
            positions[nids, t_e] - positions[pid, t_e], axis=1
        )
        for T in out:
            t_past = max(t_e - int(T), 0)
            d_past = np.linalg.norm(
                positions[nids, t_past] - positions[pid, t_past], axis=1
            )
            sig = np.abs(np.log((d_past + EPS) / (d_now + EPS)))
            out[int(T)].extend(sig.tolist())

    return {T: np.asarray(v) for T, v in out.items()}


def coherent_fraction(sigma_by_T: dict[int, np.ndarray], sigma_star: float) -> dict:
    """Fraction of neighbours with ``sigma <= sigma_star`` at each T."""
    return {T: float((s <= sigma_star).mean()) for T, s in sigma_by_T.items()}


def alpha_sweep(
    positions: np.ndarray,
    velocity_truth: np.ndarray,
    t_e: int,
    target_ids: np.ndarray,
    T_values: Iterable[int],
    alpha_values: Iterable[float],
    r_search: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scan ``(T, alpha)`` and return the mean relative error of ``v_coh``.

    At each ``(T, alpha)`` the coherent velocity is

        v_coh_j = sum_k w_k v_truth[k, t_e]   with
        w_k = (Lambda_k / <Lambda>)^(-1) + alpha * (d_k / max d)^(-1),

    normalised. The relative error is ``|v_coh - v_true| / |v_true|`` averaged
    over all targets.

    Returns
    -------
    T_array, alpha_array, err_grid
        ``err_grid`` has shape ``(len(T_values), len(alpha_values))``.
    """
    T_arr = np.asarray(list(T_values), dtype=int)
    a_arr = np.asarray(list(alpha_values), dtype=float)
    err_grid = np.zeros((len(T_arr), len(a_arr)))

    tree = KDTree(positions[:, t_e])
    for it, T in enumerate(T_arr):
        for ia, alpha in enumerate(a_arr):
            errs = []
            for pid in target_ids:
                nids = tree.query_radius(
                    positions[pid, t_e].reshape(1, -1), r=r_search
                )[0]
                nids = np.array([n for n in nids if n != pid])
                if len(nids) < 2:
                    continue

                d_now = np.linalg.norm(
                    positions[nids, t_e] - positions[pid, t_e], axis=1
                )
                t_past = max(t_e - int(T), 0)
                d_past = np.linalg.norm(
                    positions[nids, t_past] - positions[pid, t_past], axis=1
                )
                T_int = max(t_e - t_past, 1)
                lam = np.abs(np.log((d_past + EPS) / (d_now + EPS))) / T_int

                ls = np.maximum(lam, EPS)
                lm = ls.mean() + EPS
                r = d_now.max() + EPS
                w = (ls / lm) ** (-1) + alpha * (d_now / r) ** (-1)
                w = np.where(np.isfinite(w), w, 1.0)
                w /= w.sum()

                v_coh = (w[:, None] * velocity_truth[nids, t_e]).sum(0)
                v_true = velocity_truth[pid, t_e]
                denom = np.linalg.norm(v_true) + EPS
                errs.append(np.linalg.norm(v_coh - v_true) / denom)

            err_grid[it, ia] = float(np.mean(errs)) if errs else np.nan

    return T_arr, a_arr, err_grid
