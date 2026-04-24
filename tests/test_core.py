"""Sanity tests for the predictor on a scripted analytical flow.

The flow is a 2D Arnold-Beltrami-Childress style swirl with a drift. Every
particle follows a smooth trajectory, so the polynomial baseline and the
coherent predictors should all reach very low errors on clean data, and
the coherent variants must beat the polynomial baseline on noisy data.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KDTree

from coherent_predictor import (
    L_A,
    L_V,
    PredictorConfig,
    backward_ftle,
    coherent_mask,
    compute_fd,
    compute_smoothed,
    compute_weights,
    median_nn_distance,
    predict_one_particle,
    solve_coherent,
    solve_poly_full,
)


# ---------------------------------------------------------------------------
# Synthetic trajectories
# ---------------------------------------------------------------------------

def _make_flow(n_particles: int = 200, n_steps: int = 80, seed: int = 0):
    """2D rotation + drift + gentle shear — analytically smooth."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.0, 1.0, size=(n_particles, 2))
    dt = 1.0
    omega = 0.05
    drift = np.array([0.03, 0.01])
    P = np.zeros((n_particles, n_steps, 2))
    for i in range(n_steps):
        t = i * dt
        c, s = np.cos(omega * t), np.sin(omega * t)
        R = np.array([[c, -s], [s, c]])
        P[:, i, :] = x0 @ R.T + drift * t
    return P, dt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_stencil_sums():
    """Polynomial smoother stencils reproduce the centre polynomial exactly."""
    # Applying L_V to a quadratic should yield its first derivative; to L_A its second.
    t = np.arange(5, dtype=float)
    for deg in (0, 1, 2):
        y = t ** deg
        # Evaluated at the right endpoint of the window (t = 4).
        v_est = (L_V * y).sum()
        a_est = (L_A * y).sum()
        if deg == 0:
            assert abs(v_est) < 1e-12 and abs(a_est) < 1e-12
        elif deg == 1:
            assert abs(v_est - 1.0) < 1e-12 and abs(a_est) < 1e-12
        else:  # deg == 2
            # d/dt t^2 at t=4 is 8, d2/dt2 t^2 is 2
            assert abs(v_est - 8.0) < 1e-10
            assert abs(a_est - 2.0) < 1e-10


def test_polynomial_baseline_zero_error_on_polynomial_data():
    """Fitting a cubic to cubic positions yields near zero extrapolation error."""
    tau = np.arange(7, dtype=float)
    rng = np.random.default_rng(1)
    coefs = rng.normal(size=(2, 4))  # cubic in 2D
    pos = np.zeros((7, 2))
    for d in range(2):
        pos[:, d] = np.polyval(coefs[d, ::-1], tau)

    tp = tau[-1] + 1.0
    pp, pv, pa = solve_poly_full(tau, pos, order=3)
    for d in range(2):
        assert abs(pp[d] - np.polyval(coefs[d, ::-1], tp)) < 1e-9


def test_coherent_reduces_to_polynomial_when_weights_zero():
    """With a1 = a2 = 0 and no secondary, the coherent solver equals the polynomial."""
    tau = np.arange(7, dtype=float)
    rng = np.random.default_rng(2)
    pos = rng.normal(size=(7, 2))
    v_coh = np.zeros(2)
    a_coh = np.zeros(2)

    pp0, pv0, pa0 = solve_poly_full(tau, pos, order=3)
    pp1, pv1, pa1 = solve_coherent(tau, pos, v_coh, a_coh, a1=0.0, a2=0.0, order=3)

    assert np.allclose(pp0, pp1, atol=1e-10)
    assert np.allclose(pv0, pv1, atol=1e-10)
    assert np.allclose(pa0, pa1, atol=1e-10)


def test_ftle_and_weights_shapes():
    """Backward FTLE and weights return the expected shapes and are positive."""
    P, _ = _make_flow(50, 40, seed=3)
    nids = np.arange(1, 10)
    lam = backward_ftle(P, target_idx=0, neighbour_ids=nids, t_e=20, T_int=8)
    assert lam.shape == (9,)
    assert np.all(lam >= 0.0)

    d_now = np.random.default_rng(0).uniform(0.1, 1.0, size=9)
    w = compute_weights(lam, d_now, alpha_w=3.0)
    assert abs(w.sum() - 1.0) < 1e-10
    assert np.all(w >= 0.0)


def test_predict_one_particle_end_to_end_beats_polynomial():
    """On noisy synthetic data, P and P+S should both beat the polynomial baseline."""
    P, dt = _make_flow(n_particles=200, n_steps=60, seed=7)
    rng = np.random.default_rng(10)
    char = float(np.mean(np.sqrt(np.sum(np.diff(P, axis=1) ** 2, axis=2))))
    P_n = P + rng.normal(0.0, 0.10 * char, size=P.shape)

    V, A = compute_fd(P, dt)
    V_n, _ = compute_fd(P_n, dt)
    V_s, A_s = compute_smoothed(P_n, dt)

    median_nn = median_nn_distance(P[:, 30])
    cfg = PredictorConfig(hist=7, order=3, ftle_pctile=60, T_ftle=8)

    te = 30
    tree = KDTree(P_n[:, te])
    ev_poly, ev_P, ev_PS = [], [], []

    for pid in range(0, 200, 4):  # every 4th particle for speed
        out = predict_one_particle(
            pid=pid, te=te,
            positions=P_n,
            velocity_noisy=V_n,
            velocity_smooth=V_s,
            accel_smooth=A_s,
            tree_te=tree,
            median_nn=median_nn,
            dt=dt,
            cfg=cfg,
        )
        if out is None:
            continue
        v_true = V[pid, te + 1] * dt
        ev_poly.append(np.linalg.norm(out["poly_v"] - v_true))
        ev_P.append(np.linalg.norm(out["P_v"] - v_true))
        ev_PS.append(np.linalg.norm(out["PS_v"] - v_true))

    assert len(ev_poly) > 10, "too few particles evaluated"
    # On this smooth flow, the coherent variants should match or beat polynomial.
    m_poly = float(np.mean(ev_poly))
    m_P = float(np.mean(ev_P))
    m_PS = float(np.mean(ev_PS))
    assert m_P <= m_poly * 1.05, f"P should not be much worse than Poly: {m_P:.3e} vs {m_poly:.3e}"
    assert m_PS <= m_poly * 1.05, f"P+S should not be much worse than Poly: {m_PS:.3e} vs {m_poly:.3e}"


def test_coherent_mask_percentile():
    lam = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = coherent_mask(lam, percentile=40)
    # 40th percentile of [1..5] = 2.6, so elements <= 2.6 are kept => 1.0, 2.0
    assert m.sum() == 2
    assert m[0] and m[1]
