"""Polynomial and coherent-motion predictors for Lagrangian trajectories.

The three solvers compared in the paper all predict position, velocity and
acceleration at ``tau_{n+1}`` from a short history of noisy positions
``{X(tau_{n-k+1}), ..., X(tau_n)}``.

* ``solve_poly_full`` is the pure polynomial baseline used by Novara and
  Scarano (2013) and many later variants.

* ``solve_coherent`` adds coherent motion constraints at ``tau_n`` (primary
  neighbours) and optionally at ``tau_{n+1}`` (secondary neighbours). The
  cost function is Eq. 2.25 in the paper,

      J = (1/k) sum ||X(tau_i) - y_i||^2
        + a1 ||dX(tau_n) - v_coh||^2
        + a2 ||ddX(tau_n) - a_coh||^2
        + b0 ||X(tau_{n+1}) - y_sec||^2
        + b1 ||dX(tau_{n+1}) - v_sec||^2
        + b2 ||ddX(tau_{n+1}) - a_sec||^2.

  Setting ``y_sec, v_sec, a_sec`` to ``None`` reduces it to the primary only
  (P) variant. Supplying them activates the primary plus secondary (P+S)
  variant.

* ``predict_one_particle`` is an end to end driver that combines neighbour
  search, FTLE classification, weighting and the solver above. It is what
  the notebook calls inside its evaluation loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.neighbors import KDTree

from .ftle import EPS, backward_ftle, coherent_mask, compute_weights

# ---------------------------------------------------------------------------
# Low level solvers (pure algebra, no neighbour search)
# ---------------------------------------------------------------------------


def solve_poly_full(
    tau: np.ndarray, pos: np.ndarray, order: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure polynomial extrapolation to ``tau[-1] + 1``.

    Parameters
    ----------
    tau
        1D array of length ``k``, history times.
    pos
        Array shaped ``(k, d)``, history positions. ``d`` is 2 or 3.
    order
        Polynomial order, clipped to ``k - 1``.

    Returns
    -------
    pp, pv, pa
        Predicted position, velocity and acceleration, each shaped ``(d,)``.
    """
    k = len(tau)
    order = min(order, k - 1)
    tp = tau[-1] + 1.0
    ndim = pos.shape[1]

    pp = np.zeros(ndim)
    pv = np.zeros(ndim)
    pa = np.zeros(ndim)
    for d in range(ndim):
        # np.polyfit returns highest order first; flip so c[j] is the t^j coeff.
        c = np.polyfit(tau, pos[:, d], order)[::-1]
        for j in range(order + 1):
            pp[d] += c[j] * tp**j
        for j in range(1, order + 1):
            pv[d] += j * c[j] * tp ** (j - 1)
        for j in range(2, order + 1):
            pa[d] += j * (j - 1) * c[j] * tp ** (j - 2)
    return pp, pv, pa


def solve_coherent(
    tau: np.ndarray,
    pos: np.ndarray,
    v_coh: np.ndarray,
    a_coh: np.ndarray,
    a1: float,
    a2: float,
    order: int = 3,
    y_sec: Optional[np.ndarray] = None,
    v_sec: Optional[np.ndarray] = None,
    a_sec: Optional[np.ndarray] = None,
    b0: float = 0.1,
    b1: float = 0.5,
    b2: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Polynomial fit with coherent motion constraints (Eq. 2.25).

    If ``y_sec``, ``v_sec`` and ``a_sec`` are all ``None`` the solver is the
    primary only variant (P). Supplying them switches to primary plus
    secondary (P+S).
    """
    k = len(tau)
    order = min(order, k - 1)
    ndim = pos.shape[1]
    tp = tau[-1] + 1.0
    tn = tau[-1]

    H = np.vander(tau, N=order + 1, increasing=True)

    Av = np.zeros(order + 1)
    for j in range(1, order + 1):
        Av[j] = j * tn ** (j - 1)

    Aa = np.zeros(order + 1)
    for j in range(2, order + 1):
        Aa[j] = j * (j - 1) * tn ** (j - 2)

    Xp = np.array([tp**j for j in range(order + 1)])
    Vp = np.zeros(order + 1)
    for j in range(1, order + 1):
        Vp[j] = j * tp ** (j - 1)
    Aap = np.zeros(order + 1)
    for j in range(2, order + 1):
        Aap[j] = j * (j - 1) * tp ** (j - 2)

    M = (1.0 / k) * H.T @ H + a1 * np.outer(Av, Av) + a2 * np.outer(Aa, Aa)
    has_sec = y_sec is not None and v_sec is not None and a_sec is not None
    if has_sec:
        M += (
            b0 * np.outer(Xp, Xp)
            + b1 * np.outer(Vp, Vp)
            + b2 * np.outer(Aap, Aap)
        )

    pp = np.zeros(ndim)
    pv = np.zeros(ndim)
    pa = np.zeros(ndim)
    for d in range(ndim):
        rhs = (
            (1.0 / k) * H.T @ pos[:, d]
            + a1 * Av * v_coh[d]
            + a2 * Aa * a_coh[d]
        )
        if has_sec:
            rhs += b0 * Xp * y_sec[d] + b1 * Vp * v_sec[d] + b2 * Aap * a_sec[d]

        try:
            c = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(M, rhs, rcond=None)[0]

        pp[d] = Xp @ c
        pv[d] = Vp @ c
        pa[d] = Aap @ c
    return pp, pv, pa


# ---------------------------------------------------------------------------
# High level driver
# ---------------------------------------------------------------------------


@dataclass
class PredictorConfig:
    """All tunable parameters collected in one place."""

    hist: int = 7            # history length in snapshots
    order: int = 3           # polynomial order
    r_scale: float = 4.0     # r_search = U_local * dt * r_scale
    ftle_pctile: float = 50  # coherence threshold percentile
    alpha_w: float = 3.0     # weighting coefficient between Lambda and d
    T_ftle: int = 8          # backward integration window in snapshots
    a1: float = 0.5          # primary velocity weight
    a2: float = 5.0          # primary acceleration weight
    b0: float = 0.1          # secondary position weight
    b1: float = 0.5          # secondary velocity weight
    b2: float = 5.0          # secondary acceleration weight
    min_primary: int = 2     # minimum primary neighbours to keep P+S alive
    min_secondary: int = 2   # minimum secondary neighbours to keep P+S alive
    # P+S flavour:
    #   "pooled"    -> merge primary + past-phase-delayed coherent neighbours
    #                  into a single weighted (v_coh, a_coh) and reuse the
    #                  3-term cost. Reproduces Fig. 7b of the paper.
    #   "augmented" -> use secondary constraints at tau_{n+1} (non-coherent
    #                  pool). Corresponds to the 6-term Eq. 2.25 and is what
    #                  the SIREN PINN in Appendix C uses.
    secondary_mode: str = "pooled"
    # Phase delays (in snapshots) for pooled secondary neighbours. The first
    # delay that yields enough coherent secondaries wins.
    phase_delays: tuple = (1, 2)


def _adaptive_radius(
    velocity_history: np.ndarray,
    dt: float,
    r_scale: float,
    r_min: float,
) -> float:
    """Local search radius proportional to the history max velocity."""
    U_loc = np.linalg.norm(velocity_history, axis=-1).max()
    return float(max(U_loc * dt * r_scale, r_min))


def _primary_block(
    pid: int,
    te: int,
    positions: np.ndarray,
    velocity_noisy: np.ndarray,
    velocity_smooth: np.ndarray,
    accel_smooth: np.ndarray,
    tree_te: KDTree,
    cfg: PredictorConfig,
    dt: float,
    r_min: float,
) -> Optional[dict]:
    """Find primary coherent neighbours at ``te`` and build their v/a targets.

    Returns ``None`` if not enough neighbours are found.
    """
    hist = cfg.hist
    tidx = np.arange(max(te - hist + 1, 0), te + 1)
    if len(tidx) < 3:
        return None
    r_s = _adaptive_radius(
        velocity_noisy[pid, tidx], dt, cfg.r_scale, r_min
    )

    nids = tree_te.query_radius(
        positions[pid, te].reshape(1, -1), r=r_s
    )[0]
    nids = np.array([n for n in nids if n != pid])
    if len(nids) < 4:
        # Fall back to nearest neighbours so we always have a pool to threshold.
        _, ind = tree_te.query(positions[pid, te].reshape(1, -1), k=8)
        nids = ind[0, 1:]
    if len(nids) < 2:
        return None

    lam = backward_ftle(positions, pid, nids, te, cfg.T_ftle)
    mask = coherent_mask(lam, cfg.ftle_pctile)
    if mask.sum() < cfg.min_primary:
        return None

    prim_ids = nids[mask]
    d_now = np.linalg.norm(
        positions[prim_ids, te] - positions[pid, te], axis=1
    )
    weights = compute_weights(lam[mask], d_now, cfg.alpha_w)

    v_coh = (weights[:, None] * velocity_smooth[prim_ids, te]).sum(0) * dt
    a_coh = (weights[:, None] * accel_smooth[prim_ids, te]).sum(0) * dt**2

    return {
        "neighbour_ids_all": nids,
        "coh_mask": mask,
        "lam": lam,
        "prim_ids": prim_ids,
        "d_now": d_now,
        "weights": weights,
        "v_coh": v_coh,
        "a_coh": a_coh,
    }


def _augmented_secondary(
    pid: int,
    te: int,
    positions: np.ndarray,
    velocity_smooth: np.ndarray,
    accel_smooth: np.ndarray,
    prim_block: dict,
    cfg: PredictorConfig,
    dt: float,
) -> Optional[dict]:
    """6-term Eq. 2.25 secondary: non-coherent neighbours at ``te + 1``.

    This is what the SIREN PINN appendix uses. It adds three new constraint
    terms to the cost function instead of pooling with the primary set.
    """
    nids = prim_block["neighbour_ids_all"]
    lam = prim_block["lam"]
    non_coh = ~prim_block["coh_mask"]
    if non_coh.sum() < cfg.min_secondary:
        return None

    sec_ids = nids[non_coh]
    lam_sec = lam[non_coh]
    d_sec = np.linalg.norm(
        positions[sec_ids, te] - positions[pid, te], axis=1
    )
    w_sec = compute_weights(lam_sec, d_sec, cfg.alpha_w)

    tp_idx = min(te + 1, positions.shape[1] - 1)
    y_sec = positions[pid, te] + (
        w_sec[:, None] * (positions[sec_ids, tp_idx] - positions[sec_ids, te])
    ).sum(0)
    v_sec = (w_sec[:, None] * velocity_smooth[sec_ids, tp_idx]).sum(0) * dt
    a_sec = (w_sec[:, None] * accel_smooth[sec_ids, tp_idx]).sum(0) * dt**2

    return {
        "sec_ids": sec_ids,
        "y_sec": y_sec,
        "v_sec": v_sec,
        "a_sec": a_sec,
    }


def _pooled_secondary(
    pid: int,
    te: int,
    positions: np.ndarray,
    velocity_noisy: np.ndarray,
    velocity_smooth: np.ndarray,
    accel_smooth: np.ndarray,
    prim_block: dict,
    cfg: PredictorConfig,
    dt: float,
    r_min: float,
) -> Optional[dict]:
    """Pooled P+S: merge primary and past-phase-delayed coherent neighbours.

    At each phase delay ``td`` in ``cfg.phase_delays`` the solver:
      1. Finds neighbours within a local search radius at ``te - td``
      2. Drops the ones that are already primaries at ``te``
      3. Computes Lambda using the past-to-now distance ratio
      4. Keeps the ones at or below the FTLE percentile threshold

    The first delay that produces at least ``cfg.min_secondary`` coherent
    survivors wins. Primary + secondary are then concatenated and weighted
    with a single ``compute_weights`` call. The solver runs with the
    pooled ``(v_coh, a_coh)`` and the 3-term cost function.

    Reproduces Fig. 7b of the paper.
    """
    hist = cfg.hist
    prim_ids = prim_block["prim_ids"]
    prim_set = set(prim_ids.tolist())
    lam_prim = prim_block["lam"][prim_block["coh_mask"]]
    d_prim = prim_block["d_now"]

    for td in cfg.phase_delays:
        tp = te - int(td)
        if tp < cfg.T_ftle:
            continue
        tree_tp = KDTree(positions[:, tp])
        vhp = np.linalg.norm(
            velocity_noisy[pid, max(tp - hist + 1, 0):tp + 1], axis=1
        )
        U_loc_p = max(vhp.max() if len(vhp) else 1e-12, 1e-12)
        r_sec = max(U_loc_p * dt * cfg.r_scale, r_min)

        cand = tree_tp.query_radius(
            positions[pid, tp].reshape(1, -1), r=r_sec
        )[0]
        cand = np.array([n for n in cand if n not in prim_set and n != pid])
        if len(cand) < 4:
            _, ind = tree_tp.query(positions[pid, tp].reshape(1, -1), k=31)
            cand = np.array(
                [n for n in ind[0, 1:] if n not in prim_set and n != pid]
            )
        if len(cand) < 2:
            continue

        # FTLE for the secondary candidates (past -> te distance ratio).
        d_tp = np.linalg.norm(positions[cand, tp] - positions[pid, tp], axis=1)
        d_te = np.linalg.norm(positions[cand, te] - positions[pid, te], axis=1)
        ls = np.abs(np.log((d_tp + 1e-15) / (d_te + 1e-15))) / max(int(td), 1)
        sm = ls <= np.percentile(ls, cfg.ftle_pctile)
        if sm.sum() < cfg.min_secondary:
            continue

        sec_ids = cand[sm]
        lam_sec = ls[sm]
        d_sec = np.linalg.norm(
            positions[sec_ids, te] - positions[pid, te], axis=1
        )

        # Pool with the primaries and build a single weighted target.
        lam_pool = np.concatenate([lam_prim, lam_sec])
        d_pool = np.concatenate([d_prim, d_sec])
        v_prim = velocity_smooth[prim_ids, te] * dt
        a_prim = accel_smooth[prim_ids, te] * dt**2
        v_sec_arr = velocity_smooth[sec_ids, te] * dt
        a_sec_arr = accel_smooth[sec_ids, te] * dt**2
        v_pool = np.vstack([v_prim, v_sec_arr])
        a_pool = np.vstack([a_prim, a_sec_arr])

        w_pool = compute_weights(lam_pool, d_pool, cfg.alpha_w)
        v_coh_pooled = (w_pool[:, None] * v_pool).sum(0)
        a_coh_pooled = (w_pool[:, None] * a_pool).sum(0)

        return {
            "sec_ids": sec_ids,
            "v_coh_pooled": v_coh_pooled,
            "a_coh_pooled": a_coh_pooled,
            "phase_delay": int(td),
        }

    return None


def predict_one_particle(
    pid: int,
    te: int,
    positions: np.ndarray,
    velocity_noisy: np.ndarray,
    velocity_smooth: np.ndarray,
    accel_smooth: np.ndarray,
    tree_te: KDTree,
    median_nn: float,
    dt: float,
    cfg: PredictorConfig,
) -> Optional[dict]:
    """Run Poly, P and P+S for a single particle at a single snapshot.

    Returns a dictionary with keys ``{poly_p, poly_v, poly_a, P_p, P_v, P_a,
    PS_p, PS_v, PS_a, n_primary, n_secondary}``. Returns ``None`` if the
    history window is too short or no primary neighbours are found (the
    particle is then excluded from the evaluation).
    """
    hist = cfg.hist
    if te - hist + 1 < 0 or te + 1 >= positions.shape[1]:
        return None

    tidx = np.arange(te - hist + 1, te + 1)
    tau = np.arange(hist, dtype=float)
    ph = positions[pid, tidx]

    r_min = 1.5 * median_nn

    prim = _primary_block(
        pid,
        te,
        positions,
        velocity_noisy,
        velocity_smooth,
        accel_smooth,
        tree_te,
        cfg,
        dt,
        r_min,
    )
    if prim is None:
        return None

    poly_p, poly_v, poly_a = solve_poly_full(tau, ph, order=cfg.order)

    P_p, P_v, P_a = solve_coherent(
        tau, ph,
        prim["v_coh"], prim["a_coh"],
        a1=cfg.a1, a2=cfg.a2, order=cfg.order,
    )

    if cfg.secondary_mode == "pooled":
        sec = _pooled_secondary(
            pid, te, positions, velocity_noisy, velocity_smooth,
            accel_smooth, prim, cfg, dt, r_min,
        )
        if sec is not None:
            PS_p, PS_v, PS_a = solve_coherent(
                tau, ph,
                sec["v_coh_pooled"], sec["a_coh_pooled"],
                a1=cfg.a1, a2=cfg.a2, order=cfg.order,
            )
            n_sec = int(len(sec["sec_ids"]))
        else:
            PS_p, PS_v, PS_a = P_p, P_v, P_a
            n_sec = 0
    elif cfg.secondary_mode == "augmented":
        sec = _augmented_secondary(
            pid, te, positions, velocity_smooth, accel_smooth, prim, cfg, dt,
        )
        if sec is not None:
            PS_p, PS_v, PS_a = solve_coherent(
                tau, ph,
                prim["v_coh"], prim["a_coh"],
                a1=cfg.a1, a2=cfg.a2, order=cfg.order,
                y_sec=sec["y_sec"], v_sec=sec["v_sec"], a_sec=sec["a_sec"],
                b0=cfg.b0, b1=cfg.b1, b2=cfg.b2,
            )
            n_sec = int(len(sec["sec_ids"]))
        else:
            PS_p, PS_v, PS_a = P_p, P_v, P_a
            n_sec = 0
    else:
        raise ValueError(
            f"secondary_mode must be 'pooled' or 'augmented', got "
            f"{cfg.secondary_mode!r}"
        )

    return {
        "poly_p": poly_p, "poly_v": poly_v, "poly_a": poly_a,
        "P_p": P_p, "P_v": P_v, "P_a": P_a,
        "PS_p": PS_p, "PS_v": PS_v, "PS_a": PS_a,
        "n_primary": int(len(prim["prim_ids"])),
        "n_secondary": n_sec,
    }
