"""Smoke test against the real DNS file. Not in the pytest suite.

Run with:
    PYTHONPATH=src python3 tests/_smoke_on_real_data.py

Expected output (approximate, on 50 particles):
    vel P  reduction: ~50%
    vel PS reduction: ~70-74%
    acc P  reduction: ~75%
    acc PS reduction: ~79-80%
"""
import sys

import numpy as np
from sklearn.neighbors import KDTree

from coherent_predictor import (
    PredictorConfig,
    add_positional_noise,
    compute_fd,
    compute_smoothed,
    load_trajectories,
    median_nn_distance,
    predict_one_particle,
)


def main(data_path: str, n_particles: int = 50):
    dt = 10.0
    P = load_trajectories(data_path, dims=2)
    print(f"P shape: {P.shape}")

    P_n, _ = add_positional_noise(P, noise_fraction=0.10, seed=123)
    V, A = compute_fd(P, dt)
    V_n, _ = compute_fd(P_n, dt)
    V_s, A_s = compute_smoothed(P_n, dt)
    median_nn = median_nn_distance(P[:, 50])
    print(f"median_nn: {median_nn:.4e}")

    cfg = PredictorConfig()
    rng = np.random.default_rng(42)
    eval_particles = rng.choice(P.shape[0], size=n_particles, replace=False)

    te = 50
    tree = KDTree(P_n[:, te])
    errs = {k: [] for k in ("poly_v", "P_v", "PS_v", "poly_a", "P_a", "PS_a")}

    for pid in eval_particles:
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
        a_true = A[pid, te + 1] * dt ** 2
        errs["poly_v"].append(np.linalg.norm(out["poly_v"] - v_true))
        errs["P_v"].append(np.linalg.norm(out["P_v"] - v_true))
        errs["PS_v"].append(np.linalg.norm(out["PS_v"] - v_true))
        errs["poly_a"].append(np.linalg.norm(out["poly_a"] - a_true))
        errs["P_a"].append(np.linalg.norm(out["P_a"] - a_true))
        errs["PS_a"].append(np.linalg.norm(out["PS_a"] - a_true))

    def red(a, b):
        return (1 - np.mean(a) / np.mean(b)) * 100

    print(f"particles evaluated: {len(errs['poly_v'])}")
    print(f"vel  P  reduction: {red(errs['P_v'],  errs['poly_v']):6.1f} %")
    print(f"vel P+S reduction: {red(errs['PS_v'], errs['poly_v']):6.1f} %")
    print(f"acc  P  reduction: {red(errs['P_a'],  errs['poly_a']):6.1f} %")
    print(f"acc P+S reduction: {red(errs['PS_a'], errs['poly_a']):6.1f} %")

    # Minimum thresholds so this script doubles as a regression check.
    assert red(errs["P_v"], errs["poly_v"]) > 30.0, "P velocity too weak"
    assert red(errs["PS_v"], errs["poly_v"]) > 40.0, "P+S velocity too weak"
    print("smoke test passed")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "TBarrier/TBarrier/2D/demos/AdvectiveBarriers/FTLE2D/Main/"
        "1_10000_dt10_comparison.mat"
    )
    main(path)
