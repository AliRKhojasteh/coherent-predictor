"""Finite difference and polynomial smoothing operators for Lagrangian derivatives.

The noisy FD operators are second order central differences with one sided
closures at the boundaries. The polynomial smoother fits a quadratic by least
squares over a 5 point window and returns velocity and acceleration
coefficients analytically derived from that fit (stencils ``L_V`` and ``L_A``).

These stencils come from differentiating the least squares quadratic
    p(t) = c0 + c1 t + c2 t^2
at the centre point of a symmetric 5 point window. The same smoother is used
for every neighbour in the coherent cost function so that primary and
secondary constraints are treated identically.
"""

from __future__ import annotations

import numpy as np

# Coefficients derived from a 5 point least squares quadratic fit.
# Applied as V_s[te] = sum_k L_V[k] * P_n[te-4+k] / dt, evaluated at the
# right endpoint of the window so that it stays causal.
L_V = np.array([260.0, -270.0, -400.0, -130.0, 540.0]) / 700.0
L_A = np.array([2.0, -1.0, -2.0, -1.0, 2.0]) / 7.0


def compute_fd(positions: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Central finite difference velocity and acceleration.

    Parameters
    ----------
    positions
        Array of shape ``(N, T, d)``. Particle index first, time index second,
        spatial dimension last.
    dt
        Time step between successive snapshots, in the same units used later
        when solving the predictor.

    Returns
    -------
    velocity, acceleration
        Two arrays, both shaped like ``positions``.
    """
    if positions.ndim != 3:
        raise ValueError("positions must have shape (N, T, d)")

    V = np.zeros_like(positions)
    V[:, 1:-1] = (positions[:, 2:] - positions[:, :-2]) / (2.0 * dt)
    V[:, 0] = (positions[:, 1] - positions[:, 0]) / dt
    V[:, -1] = (positions[:, -1] - positions[:, -2]) / dt

    A = np.zeros_like(V)
    A[:, 1:-1] = (V[:, 2:] - V[:, :-2]) / (2.0 * dt)
    A[:, 0] = (V[:, 1] - V[:, 0]) / dt
    A[:, -1] = (V[:, -1] - V[:, -2]) / dt
    return V, A


def compute_smoothed(
    positions: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Polynomial smoothed velocity and acceleration over a 5 point window.

    Window sits to the left of the evaluation index, so ``V_s[te]`` uses
    positions from ``te-4`` up to ``te``. Values for the first four samples
    (where the window is incomplete) are filled in from the plain FD
    estimator.
    """
    if positions.ndim != 3:
        raise ValueError("positions must have shape (N, T, d)")

    T_steps = positions.shape[1]
    V_s = np.zeros_like(positions)
    A_s = np.zeros_like(positions)

    for te in range(4, T_steps):
        for k in range(5):
            V_s[:, te, :] += L_V[k] * positions[:, te - 4 + k, :]
            A_s[:, te, :] += L_A[k] * positions[:, te - 4 + k, :]

    V_s[:, 4:] /= dt
    A_s[:, 4:] /= dt**2

    V_fd, A_fd = compute_fd(positions, dt)
    V_s[:, :4] = V_fd[:, :4]
    A_s[:, :4] = A_fd[:, :4]
    return V_s, A_s


def smooth_history_targets(
    target_by_history: np.ndarray, order: int = 2
) -> np.ndarray:
    """Refit a short time series of collocation targets with a low order polynomial.

    Used to denoise the acceleration collocation targets a_coh(t) before they
    are fed to the PINN. The SNR of acceleration is low (~0.17 at 10% noise),
    so fitting a quadratic over the history window removes most of the
    zero mean jitter while preserving the underlying trend.

    Parameters
    ----------
    target_by_history
        Array of shape ``(k, d)`` where ``k`` is the history length and ``d``
        the spatial dimension. ``target_by_history[i]`` is the collocation
        target at history index ``i``.
    order
        Polynomial order. Defaults to 2 (quadratic). Clipped to ``k-1``.
    """
    k, ndim = target_by_history.shape
    if k < 3:
        return target_by_history.copy()

    order = min(order, k - 1)
    t_local = np.arange(k, dtype=float)

    out = np.zeros_like(target_by_history)
    for d in range(ndim):
        coefs = np.polyfit(t_local, target_by_history[:, d], order)
        out[:, d] = np.polyval(coefs, t_local)
    return out
