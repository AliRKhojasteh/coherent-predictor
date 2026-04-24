"""SIREN + Fourier feature physics informed predictor (Appendix C).

A tiny MLP with sinusoidal activations plays the role of the trajectory
interpolant. Fourier features sit before the hidden layer to give the network
a smooth periodic basis over the history window. The total loss has the
same three data terms as the polynomial P+S solver but evaluates them at
every history snapshot (collocation), not just at the right endpoint.

This module reimplements the v7i-d configuration from the paper's appendix:

    FOURIER_K = 3        2*K = 6 Fourier features
    N_HIDDEN  = 12       hidden layer width
    OMEGA_0   = 0.50     SIREN frequency scale
    LBFGS_ITERS = 100    early stopping to avoid noise memorisation

``autograd`` supplies analytical derivatives of the SIREN output with respect
to the time coordinate, so ``v(t) = d f / d t`` and ``a(t) = d^2 f / d t^2``
are exact. ``scipy.optimize.minimize`` with L-BFGS-B drives the parameter
update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

try:
    import autograd.numpy as anp
    from autograd import grad
    _HAVE_AUTOGRAD = True
except ImportError:  # pragma: no cover - autograd is a hard requirement at runtime
    _HAVE_AUTOGRAD = False


@dataclass
class SirenConfig:
    """All knobs for the SIREN PINN."""

    fourier_k: int = 3        # number of Fourier frequency pairs -> 2*K features
    n_hidden: int = 12        # hidden units
    omega_0: float = 0.50     # SIREN frequency scale
    lbfgs_maxiter: int = 100  # iterations for L-BFGS-B
    ftol: float = 1e-15
    gtol: float = 1e-10
    # Cost function weights (same as the polynomial P+S solver)
    a1: float = 0.5
    a2: float = 15.0
    b0: float = 0.1
    b1: float = 0.5
    b2: float = 5.0

    @property
    def ff_dim(self) -> int:
        return 2 * self.fourier_k

    def n_params(self, ndim: int) -> int:
        nh = self.n_hidden
        return nh * self.ff_dim + nh + nh * ndim + ndim


# ---------------------------------------------------------------------------
# Parameter packing, Fourier features, SIREN forward pass
# ---------------------------------------------------------------------------


def _pack(W1, b1, W2, b2):
    return anp.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])


def _unpack(theta, cfg: SirenConfig, ndim: int):
    nh = cfg.n_hidden
    ff = cfg.ff_dim
    i = 0
    W1 = theta[i : i + nh * ff].reshape(nh, ff); i += nh * ff
    b1 = theta[i : i + nh]; i += nh
    W2 = theta[i : i + nh * ndim].reshape(nh, ndim); i += nh * ndim
    b2 = theta[i : i + ndim]
    return W1, b1, W2, b2


def _fourier_features_scalar(tau_scalar, T_span: float, fourier_k: int):
    feats = []
    for k in range(1, fourier_k + 1):
        ang = 2.0 * anp.pi * k * tau_scalar / T_span
        feats.append(anp.sin(ang))
        feats.append(anp.cos(ang))
    return anp.array(feats)


def _fourier_features_np(tau_arr: np.ndarray, T_span: float, fourier_k: int):
    out = np.zeros((len(tau_arr), 2 * fourier_k))
    for k in range(1, fourier_k + 1):
        ang = 2.0 * np.pi * k * tau_arr / T_span
        out[:, 2 * (k - 1)] = np.sin(ang)
        out[:, 2 * (k - 1) + 1] = np.cos(ang)
    return out


def _mlp_pos_d(theta, tau_scalar, cfg: SirenConfig, ndim: int, d: int, T_span: float):
    W1, b1, W2, b2 = _unpack(theta, cfg, ndim)
    ff = _fourier_features_scalar(tau_scalar, T_span, cfg.fourier_k)
    pre = anp.dot(W1, ff) + b1
    h = anp.sin(cfg.omega_0 * pre)
    return anp.dot(h, W2[:, d]) + b2[d]


def _make_derivs(cfg: SirenConfig, ndim: int, T_span: float):
    vel_fns = []
    acc_fns = []
    for d in range(ndim):
        f_d = lambda theta, tau, _d=d: _mlp_pos_d(theta, tau, cfg, ndim, _d, T_span)
        df_d = grad(f_d, argnum=1)
        d2f_d = grad(df_d, argnum=1)
        vel_fns.append(df_d)
        acc_fns.append(d2f_d)
    return vel_fns, acc_fns


# ---------------------------------------------------------------------------
# Warm start
# ---------------------------------------------------------------------------


def _warm_start(
    poly_coefs: np.ndarray,
    tau: np.ndarray,
    cfg: SirenConfig,
    ndim: int,
    T_span: float,
    seed: int = 42,
) -> np.ndarray:
    """Pick theta so the SIREN reproduces the polynomial baseline at history taus.

    Small W1 / b1 keep the hidden pre activation in the near linear regime of
    ``sin``. Then W2 and b2 are set by least squares on the polynomial
    evaluated at the history taus.
    """
    rng = np.random.default_rng(seed)
    nh = cfg.n_hidden
    scale = 0.05 / cfg.omega_0

    W1 = rng.normal(0.0, scale, size=(nh, cfg.ff_dim))
    b1 = rng.normal(0.0, scale, size=nh)

    # Evaluate polynomial at history taus.
    k = len(tau)
    poly_vals = np.zeros((k, ndim))
    for d in range(ndim):
        for j, cj in enumerate(poly_coefs[d]):
            poly_vals[:, d] += cj * tau**j

    FF = _fourier_features_np(tau, T_span, cfg.fourier_k)
    pre = FF @ W1.T + b1[None, :]
    H = np.sin(cfg.omega_0 * pre)

    H_aug = np.column_stack([H, np.ones(k)])
    W2 = np.zeros((nh, ndim))
    b2 = np.zeros(ndim)
    for d in range(ndim):
        sol = np.linalg.lstsq(H_aug, poly_vals[:, d], rcond=None)[0]
        W2[:, d] = sol[:nh]
        b2[d] = sol[nh]

    return _pack(W1, b1, W2, b2)


# ---------------------------------------------------------------------------
# Cost function and public entry point
# ---------------------------------------------------------------------------


def _make_cost(
    tau: np.ndarray,
    pos: np.ndarray,
    v_coh_all: np.ndarray,
    a_coh_all: np.ndarray,
    y_sec: Optional[np.ndarray],
    v_sec: Optional[np.ndarray],
    a_sec: Optional[np.ndarray],
    cfg: SirenConfig,
    ndim: int,
    T_span: float,
):
    k = len(tau)
    tp = tau[-1] + 1.0
    has_sec = (y_sec is not None) and (v_sec is not None) and (a_sec is not None)
    vel_fns, acc_fns = _make_derivs(cfg, ndim, T_span)

    def cost(theta):
        J = 0.0
        # Data fidelity at every history point.
        for i in range(k):
            for d in range(ndim):
                xi = _mlp_pos_d(theta, tau[i], cfg, ndim, d, T_span)
                J = J + (1.0 / k) * (xi - pos[i, d]) ** 2
        # Collocation velocity and acceleration at every history point.
        for c_idx in range(k):
            for d in range(ndim):
                vi = vel_fns[d](theta, tau[c_idx])
                ai = acc_fns[d](theta, tau[c_idx])
                J = J + cfg.a1 * (vi - v_coh_all[c_idx, d]) ** 2
                J = J + cfg.a2 * (ai - a_coh_all[c_idx, d]) ** 2
        # Secondary constraints at tau_{n+1}.
        if has_sec:
            for d in range(ndim):
                xp = _mlp_pos_d(theta, tp, cfg, ndim, d, T_span)
                vp = vel_fns[d](theta, tp)
                ap = acc_fns[d](theta, tp)
                J = J + cfg.b0 * (xp - y_sec[d]) ** 2
                J = J + cfg.b1 * (vp - v_sec[d]) ** 2
                J = J + cfg.b2 * (ap - a_sec[d]) ** 2
        return J

    cost_grad = grad(cost)

    def cost_and_grad(theta_np):
        c = cost(theta_np)
        g = cost_grad(theta_np)
        return float(c), np.asarray(g, dtype=np.float64)

    return cost_and_grad, vel_fns, acc_fns


def predict_siren(
    tau: np.ndarray,
    pos: np.ndarray,
    v_coh_all: np.ndarray,
    a_coh_all: np.ndarray,
    poly_coefs: np.ndarray,
    y_sec: Optional[np.ndarray] = None,
    v_sec: Optional[np.ndarray] = None,
    a_sec: Optional[np.ndarray] = None,
    cfg: Optional[SirenConfig] = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Train a SIREN PINN and return predictions at ``tau_{n+1}``.

    Parameters
    ----------
    tau
        History times, length ``k``.
    pos
        History positions, shape ``(k, d)``.
    v_coh_all, a_coh_all
        Collocation targets at every history point, shape ``(k, d)`` each.
    poly_coefs
        Polynomial baseline coefficients, shape ``(d, order + 1)`` where
        ``poly_coefs[d, j]`` is the coefficient of ``tau**j``. Used to warm
        start the network.
    y_sec, v_sec, a_sec
        Optional secondary constraints at ``tau_{n+1}``, each shape ``(d,)``.
        Provide all three or none.
    cfg
        Optional ``SirenConfig``. Defaults reproduce v7i-d from the paper.
    seed
        RNG seed for the warm start.

    Returns
    -------
    pp, pv, pa, final_cost
        Predicted position, velocity and acceleration at ``tau_{n+1}``, plus
        the final value of the cost function.
    """
    if not _HAVE_AUTOGRAD:
        raise ImportError(
            "autograd is required for predict_siren. "
            "Install with `pip install autograd`."
        )

    if cfg is None:
        cfg = SirenConfig()

    ndim = pos.shape[1]
    tp = tau[-1] + 1.0
    # v7b fix: stretch T_span so history + prediction stays inside one period.
    T_span = 4.0 * (tau[-1] + 1.0)

    cost_and_grad, vel_fns, acc_fns = _make_cost(
        tau, pos, v_coh_all, a_coh_all, y_sec, v_sec, a_sec, cfg, ndim, T_span
    )

    theta0 = _warm_start(poly_coefs, tau, cfg, ndim, T_span, seed=seed)
    result = minimize(
        cost_and_grad,
        theta0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": cfg.lbfgs_maxiter, "ftol": cfg.ftol, "gtol": cfg.gtol},
    )
    theta_opt = result.x

    pp = np.zeros(ndim)
    pv = np.zeros(ndim)
    pa = np.zeros(ndim)
    for d in range(ndim):
        pp[d] = float(_mlp_pos_d(theta_opt, tp, cfg, ndim, d, T_span))
        pv[d] = float(vel_fns[d](theta_opt, tp))
        pa[d] = float(acc_fns[d](theta_opt, tp))

    return pp, pv, pa, float(result.fun)
