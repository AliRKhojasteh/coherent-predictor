# Method summary

A one page reminder of the notation and default parameters. The full
derivation lives in the paper.

## Cost function (Eq. 2.25)

For a target particle with history positions ``y_i`` at times ``tau_i``
(``i = 1 ... k``, ``tau_k = tau_n``) and a prediction time
``tau_{n+1}``, the coherent cost function is

    J = (1/k) sum_i ||X(tau_i) - y_i||^2           (data fidelity)
      + a1  ||dX(tau_n)    - v_coh||^2             (primary velocity)
      + a2  ||ddX(tau_n)   - a_coh||^2             (primary acceleration)
      + b0  ||X(tau_{n+1}) - y_sec||^2             (secondary position)
      + b1  ||dX(tau_{n+1}) - v_sec||^2            (secondary velocity)
      + b2  ||ddX(tau_{n+1}) - a_sec||^2           (secondary acceleration)

``X(tau)`` is a polynomial of order ``ell`` (default ``ell = 3``). The
polynomial coefficients are found in closed form by solving a small linear
system. The network based predictor in Appendix C replaces ``X(tau)`` with a
SIREN MLP and evaluates the data and collocation terms at every history
snapshot.

## FTLE based coherence

For each candidate neighbour ``j`` the backward rate of separation

    Lambda_j = |log(d_j(tau_n - T) / d_j(tau_n))| / T

is compared against a percentile threshold of the local distribution.
Neighbours at or below the threshold are **primary** (coherent); the rest
make up the **secondary** pool.

## Coherent weighting

    w_j = (Lambda_j / <Lambda>)^(-1) + alpha_w * (d_j / max d)^(-1)

followed by L1 normalisation. The first term rewards kinematic coherence,
the second rewards spatial proximity.

## Default parameters (reproduce the paper)

| Symbol       | Code                | Value |
|--------------|---------------------|------:|
| k            | ``cfg.hist``        | 7     |
| ell          | ``cfg.order``       | 3     |
| r scale      | ``cfg.r_scale``     | 4.0   |
| FTLE pctile  | ``cfg.ftle_pctile`` | 50    |
| alpha_w      | ``cfg.alpha_w``     | 3.0   |
| T (FTLE)     | ``cfg.T_ftle``      | 8     |
| a1           | ``cfg.a1``          | 0.5   |
| a2           | ``cfg.a2``          | 5.0   |
| b0           | ``cfg.b0``          | 0.1   |
| b1           | ``cfg.b1``          | 0.5   |
| b2           | ``cfg.b2``          | 5.0   |

The SIREN PINN (Appendix C, v7i-d) uses ``omega_0 = 0.5`` with 100
L-BFGS-B iterations, a width 12 hidden layer and 6 Fourier features.
