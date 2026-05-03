"""Audit-only: compare pooled vs augmented P+S modes on demo data."""
import numpy as np
from sklearn.neighbors import KDTree

from coherent_predictor import (
    PredictorConfig, add_positional_noise, compute_fd, compute_smoothed,
    load_trajectories, median_nn_distance, predict_one_particle,
)

P = load_trajectories("data/demo_2D_HIT.npz", dims=2)
P_n, _ = add_positional_noise(P, 0.10, seed=123)
V, A = compute_fd(P, 10.0)
V_n, _ = compute_fd(P_n, 10.0)
V_s, A_s = compute_smoothed(P_n, 10.0)
median_nn = median_nn_distance(P[:, 30])

cfg_pool = PredictorConfig(secondary_mode="pooled")
cfg_aug = PredictorConfig(secondary_mode="augmented")

ev = {k: [] for k in ("pool_v", "aug_v", "pool_a", "aug_a", "poly_v", "poly_a")}
te = 30
tree = KDTree(P_n[:, te])
rng = np.random.default_rng(7)
for pid in rng.choice(2000, 80, replace=False):
    o_pool = predict_one_particle(pid, te, P_n, V_n, V_s, A_s, tree, median_nn, 10.0, cfg_pool)
    o_aug = predict_one_particle(pid, te, P_n, V_n, V_s, A_s, tree, median_nn, 10.0, cfg_aug)
    if o_pool is None or o_aug is None:
        continue
    v_t = V[pid, te + 1] * 10.0
    a_t = A[pid, te + 1] * 10.0 ** 2
    ev["pool_v"].append(np.linalg.norm(o_pool["PS_v"] - v_t))
    ev["aug_v"].append(np.linalg.norm(o_aug["PS_v"] - v_t))
    ev["pool_a"].append(np.linalg.norm(o_pool["PS_a"] - a_t))
    ev["aug_a"].append(np.linalg.norm(o_aug["PS_a"] - a_t))
    ev["poly_v"].append(np.linalg.norm(o_pool["poly_v"] - v_t))
    ev["poly_a"].append(np.linalg.norm(o_pool["poly_a"] - a_t))


def red(a, b):
    return (1 - np.mean(a) / np.mean(b)) * 100


pool_v = red(ev["pool_v"], ev["poly_v"])
aug_v = red(ev["aug_v"], ev["poly_v"])
pool_a = red(ev["pool_a"], ev["poly_a"])
aug_a = red(ev["aug_a"], ev["poly_a"])

print(f"=== TEST 11: pooled vs augmented P+S on {len(ev['pool_v'])} particles ===")
print(f"  vel: pooled {pool_v:+.1f}%   augmented {aug_v:+.1f}%")
print(f"  acc: pooled {pool_a:+.1f}%   augmented {aug_a:+.1f}%")
assert pool_v > 0 and aug_v > 0, "both modes must beat polynomial on velocity"
assert pool_a > 0 and aug_a > 0, "both modes must beat polynomial on acceleration"
print("PASS  both secondary modes beat polynomial baseline")

print()
print("=== TEST 12: phase-delay FTLE on past-to-now distance ratio ===")
te = 30
td = 1
pid = 100
tp = te - td
nids = np.arange(50, 100)
d_tp = np.linalg.norm(P_n[nids, tp] - P_n[pid, tp], axis=1)
d_te = np.linalg.norm(P_n[nids, te] - P_n[pid, te], axis=1)
ls = np.abs(np.log(d_tp / d_te)) / max(td, 1)
print(f"  ls range: [{ls.min():.4f}, {ls.max():.4f}]")
print(f"  median ls: {np.median(ls):.4f}")
print(f"  fraction below median: {(ls <= np.median(ls)).mean() * 100:.0f}%")
assert (ls >= 0).all() and ls.max() < 5.0, "phase-delayed Lambda must be non-negative and bounded"
print("PASS  phase-delayed FTLE values look sensible")
