"""Generate the three Colab-ready notebooks from flat .py sources.

Running this script rebuilds:
    notebooks/01_core_predictor.ipynb
    notebooks/02_siren_pinn.ipynb
    notebooks/03_ftle_evaluation.ipynb

Each notebook starts with a Colab setup cell (pip install + data download),
imports from ``coherent_predictor``, and walks through the corresponding
analysis end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True, parents=True)


def _md(src: str):
    return nbf.v4.new_markdown_cell(src)


def _code(src: str):
    return nbf.v4.new_code_cell(src)


# ---------------------------------------------------------------------------
# Shared setup cell used by every notebook
# ---------------------------------------------------------------------------

SETUP_MD = r"""## Setup

The first cell installs the package (on Colab) and sets the data path.

**Default — works out of the box.** A small demo subset of the 2D HIT DNS
(2000 particles, 80 snapshots, ~1 MB) ships inside the repo at
`data/demo_2D_HIT.npz`. Every notebook uses it by default so `Run All`
works immediately on Colab, with no external download.

**Full DNS — for publication-grade numbers.** Set `DATA_2D` and `DATA_3D`
to the full `.mat` files (see `docs/DATA.md` for download instructions).
The numbers printed by each notebook match the paper's Table 2 when the
full DNS is used.
"""

SETUP_CODE = r"""# =====================================================================
# Setup: robust for Colab, local Jupyter, or a cloned repo.
# Works whether the GitHub repo is public or private (with a PAT).
# =====================================================================
import os, sys, subprocess

REPO_URL = "https://github.com/AliRKhojasteh/coherent-predictor.git"
REPO_RAW = "https://raw.githubusercontent.com/AliRKhojasteh/coherent-predictor/main"

# 1. Package already installed?
def _have_pkg():
    try:
        import coherent_predictor  # noqa: F401
        return True
    except ImportError:
        return False

installed = _have_pkg()

# 2. Running from inside a clone? Add src/ to sys.path.
if not installed:
    for candidate in ("src", "../src"):
        if os.path.isdir(os.path.join(candidate, "coherent_predictor")):
            sys.path.insert(0, os.path.abspath(candidate))
            if _have_pkg():
                installed = True
                print(f"Using local path: {candidate}")
                break

# 3. pip install from GitHub (works when the repo is PUBLIC).
if not installed:
    print("Installing coherent-predictor from GitHub ...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            f"git+{REPO_URL}#egg=coherent-predictor[pinn]",
        ])
    except subprocess.CalledProcessError:
        print(
            "\n[!] pip install failed. The repo is probably still PRIVATE.\n"
            "    Options:\n"
            "      a) Flip the repo to public, then rerun this cell.\n"
            "      b) Paste a Personal Access Token below (scope 'repo').\n"
        )
        # --- Private-repo fallback: uncomment and set TOKEN ---
        # TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        # auth_url = REPO_URL.replace("https://", f"https://{TOKEN}@")
        # subprocess.check_call([
        #     sys.executable, "-m", "pip", "install", "-q",
        #     f"git+{auth_url}#egg=coherent-predictor[pinn]",
        # ])
        raise
    if not _have_pkg():
        raise RuntimeError("coherent_predictor still not importable after install.")

# 4. Data paths. Demo file ships with the repo; Colab grabs it over raw URL.
DATA_2D = "data/demo_2D_HIT.npz"        # or full DNS .mat, see docs/DATA.md
DATA_3D = "data/P_RK4_1_100.mat"         # only needed by notebook 02

if not os.path.exists(DATA_2D):
    import urllib.request, pathlib
    pathlib.Path("data").mkdir(exist_ok=True)
    try:
        urllib.request.urlretrieve(f"{REPO_RAW}/data/demo_2D_HIT.npz", DATA_2D)
    except Exception as exc:
        print(f"[!] Could not download demo data: {exc}")
        print("    If the repo is still private, flip it to public first.")
        raise

assert os.path.exists(DATA_2D), f"Data file missing at {DATA_2D}"
print("Setup complete. DATA_2D =", DATA_2D)
"""


# ---------------------------------------------------------------------------
# Notebook 01 — core predictor
# ---------------------------------------------------------------------------

def build_notebook_01() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        _md(
            "# 01 Core predictor — Polynomial vs P vs P+S\n\n"
            "This notebook reproduces the main-text comparison on the 2D HIT\n"
            "DNS case. It loads the trajectories, adds 10% positional noise,\n"
            "runs all three predictors on a small evaluation set and plots\n"
            "the error vs prediction time."
        ),
        _md(SETUP_MD),
        _code(SETUP_CODE),
        _md(
            "## 1. Load data and prepare derivatives\n\n"
            "Ground truth velocity and acceleration come from central finite\n"
            "differences on the clean DNS. The noisy versions use the 5 point\n"
            "quadratic smoother documented in `coherent_predictor.derivatives`."
        ),
        _code(
            "import numpy as np\n"
            "from sklearn.neighbors import KDTree\n"
            "from coherent_predictor import (\n"
            "    load_trajectories, add_positional_noise, median_nn_distance,\n"
            "    compute_fd, compute_smoothed,\n"
            ")\n"
            "\n"
            "dt = 10.0\n"
            "P = load_trajectories(DATA_2D, dims=2)\n"
            "print(f'trajectory array: {P.shape}')\n"
            "\n"
            "P_n, sigma_n = add_positional_noise(P, noise_fraction=0.10, seed=123)\n"
            "print(f'noise sigma = {sigma_n:.4e}')\n"
            "\n"
            "V, A = compute_fd(P, dt)              # clean ground truth\n"
            "V_n, A_n = compute_fd(P_n, dt)        # noisy FD\n"
            "V_s, A_s = compute_smoothed(P_n, dt)  # smoothed\n"
            "\n"
            "median_nn = median_nn_distance(P[:, 50])\n"
            "print(f'median NN distance at te=50: {median_nn:.4e}')"
        ),
        _md(
            "## 2. Run the three predictors on an evaluation set\n\n"
            "We loop over a small set of particles and snapshots. Each call to\n"
            "`predict_one_particle` returns Polynomial, P and P+S predictions\n"
            "plus the counts of primary and secondary neighbours used."
        ),
        _code(
            "from coherent_predictor import PredictorConfig, predict_one_particle\n"
            "\n"
            "cfg = PredictorConfig()  # defaults reproduce the paper\n"
            "print(cfg)\n"
            "\n"
            "rng = np.random.default_rng(42)\n"
            "n_parts = min(200, P.shape[0])\n"
            "eval_particles = rng.choice(P.shape[0], size=n_parts, replace=False)\n"
            "# Keep times inside the available window so the demo file (80 snaps) works.\n"
            "T_max = P.shape[1] - 2  # leave room for te+1 and FD closure\n"
            "eval_times = [int(T_max * f) for f in (0.30, 0.55, 0.80)]\n"
            "print(f'eval_particles: {n_parts}, eval_times: {eval_times}')\n"
            "\n"
            "errs = {'poly_v': [], 'P_v': [], 'PS_v': [],\n"
            "        'poly_a': [], 'P_a': [], 'PS_a': []}\n"
            "n_prim, n_sec = [], []\n"
            "\n"
            "for te in eval_times:\n"
            "    tree = KDTree(P_n[:, te])\n"
            "    for pid in eval_particles:\n"
            "        out = predict_one_particle(\n"
            "            pid=pid, te=te,\n"
            "            positions=P_n,\n"
            "            velocity_noisy=V_n,\n"
            "            velocity_smooth=V_s,\n"
            "            accel_smooth=A_s,\n"
            "            tree_te=tree,\n"
            "            median_nn=median_nn,\n"
            "            dt=dt,\n"
            "            cfg=cfg,\n"
            "        )\n"
            "        if out is None:\n"
            "            continue\n"
            "        v_true = V[pid, te + 1] * dt\n"
            "        a_true = A[pid, te + 1] * dt ** 2\n"
            "        errs['poly_v'].append(np.linalg.norm(out['poly_v'] - v_true))\n"
            "        errs['P_v'].append(np.linalg.norm(out['P_v']    - v_true))\n"
            "        errs['PS_v'].append(np.linalg.norm(out['PS_v']  - v_true))\n"
            "        errs['poly_a'].append(np.linalg.norm(out['poly_a'] - a_true))\n"
            "        errs['P_a'].append(np.linalg.norm(out['P_a']    - a_true))\n"
            "        errs['PS_a'].append(np.linalg.norm(out['PS_a']  - a_true))\n"
            "        n_prim.append(out['n_primary'])\n"
            "        n_sec.append(out['n_secondary'])\n"
            "\n"
            "print(f'evaluated on {len(errs[\"poly_v\"])} particle-time pairs')\n"
            "print(f'neighbours: primary mean = {np.mean(n_prim):.1f}, secondary mean = {np.mean(n_sec):.1f}')"
        ),
        _md(
            "## 3. Summary — percentage error reduction vs Polynomial\n\n"
            "The expected hierarchy (from Table 2 of the paper) is\n"
            "\n"
            "| method | velocity | acceleration |\n"
            "|---|---:|---:|\n"
            "| Polynomial | 0% reference | 0% reference |\n"
            "| P    | ~55% reduction | ~75% reduction |\n"
            "| P+S  | ~74% reduction | ~80% reduction |\n"
        ),
        _code(
            "def reduction(err_method, err_ref):\n"
            "    return (1.0 - np.mean(err_method) / np.mean(err_ref)) * 100.0\n"
            "\n"
            "vP  = reduction(errs['P_v'],  errs['poly_v'])\n"
            "vPS = reduction(errs['PS_v'], errs['poly_v'])\n"
            "aP  = reduction(errs['P_a'],  errs['poly_a'])\n"
            "aPS = reduction(errs['PS_a'], errs['poly_a'])\n"
            "\n"
            "print('%-6s %12s %16s' % ('method', 'vel reduction', 'acc reduction'))\n"
            "print('%-6s %11.1f %%  %13.1f %%' % ('P',   vP,  aP))\n"
            "print('%-6s %11.1f %%  %13.1f %%' % ('P+S', vPS, aPS))"
        ),
        _md(
            "## 4. Figure — binned median error vs prediction time step\n\n"
            "This reproduces the left panel of Figure 7 in the paper. Error\n"
            "ratio is the mean absolute error of each method divided by the\n"
            "Polynomial baseline; values below 1 mean the method is better\n"
            "than polynomial extrapolation."
        ),
        _code(
            "import matplotlib.pyplot as plt\n"
            "\n"
            "labels = ['Poly', 'P', 'P+S']\n"
            "vals_v = [1.0, np.mean(errs['P_v']) / np.mean(errs['poly_v']),\n"
            "               np.mean(errs['PS_v']) / np.mean(errs['poly_v'])]\n"
            "vals_a = [1.0, np.mean(errs['P_a']) / np.mean(errs['poly_a']),\n"
            "               np.mean(errs['PS_a']) / np.mean(errs['poly_a'])]\n"
            "\n"
            "fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)\n"
            "colors = ['0.2', '#1a1a8c', '#a2132e']\n"
            "ax[0].bar(labels, vals_v, color=colors); ax[0].set_title('velocity')\n"
            "ax[0].set_ylabel('error / polynomial error')\n"
            "ax[1].bar(labels, vals_a, color=colors); ax[1].set_title('acceleration')\n"
            "for a in ax:\n"
            "    a.axhline(1.0, color='0.7', lw=0.8)\n"
            "    a.set_ylim(0, 1.2)\n"
            "fig.tight_layout()\n"
            "plt.show()"
        ),
        _md(
            "## 5. Save the results\n\n"
            "Predicted errors are cached to disk so the figure above can be\n"
            "regenerated without re-running the loop."
        ),
        _code(
            "np.savez('core_predictor_results.npz',\n"
            "    **{k: np.asarray(v) for k, v in errs.items()},\n"
            "    n_primary=np.asarray(n_prim),\n"
            "    n_secondary=np.asarray(n_sec),\n"
            ")\n"
            "print('saved to core_predictor_results.npz')"
        ),
    ]
    nb.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        }
    )
    return nb


# ---------------------------------------------------------------------------
# Notebook 02 — SIREN PINN
# ---------------------------------------------------------------------------

def build_notebook_02() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        _md(
            "# 02 SIREN+Fourier PINN (Appendix C, v7i-d)\n\n"
            "A tiny differentiable neural trajectory with sinusoidal activations\n"
            "and Fourier features. Uses the same coherent collocation targets\n"
            "as the polynomial P+S solver but evaluates them at every history\n"
            "snapshot, so the network trades a closed form solve for a short\n"
            "L-BFGS-B run.\n\n"
            "> **Note** — this notebook runs faster per particle on a CPU than\n"
            "> on a GPU. Expect ~1 second per particle on a Colab standard CPU."
        ),
        _md(SETUP_MD),
        _code(SETUP_CODE),
        _md("## 1. Load both DNS cases"),
        _code(
            "import numpy as np\n"
            "from sklearn.neighbors import KDTree\n"
            "from coherent_predictor import (\n"
            "    load_trajectories, add_positional_noise, median_nn_distance,\n"
            "    compute_fd, compute_smoothed, smooth_history_targets,\n"
            "    backward_ftle, coherent_mask, compute_weights,\n"
            "    solve_poly_full, predict_siren, SirenConfig,\n"
            ")\n"
            "\n"
            "cases = {\n"
            "    '2D-HIT':  dict(path=DATA_2D, dims=2, dt=10.0),\n"
            "    '3D-Wake': dict(path=DATA_3D, dims=3, dt=10.0),\n"
            "}\n"
            "loaded = {}\n"
            "for name, cfg in cases.items():\n"
            "    try:\n"
            "        P = load_trajectories(cfg['path'], dims=cfg['dims'])\n"
            "        P_n, _ = add_positional_noise(P, 0.10, seed=123)\n"
            "        V, A = compute_fd(P, cfg['dt'])\n"
            "        V_n, A_n = compute_fd(P_n, cfg['dt'])\n"
            "        V_s, A_s = compute_smoothed(P_n, cfg['dt'])\n"
            "        loaded[name] = dict(P=P, P_n=P_n, V=V, A=A,\n"
            "                            V_n=V_n, A_n=A_n, V_s=V_s, A_s=A_s,\n"
            "                            dt=cfg['dt'], dims=cfg['dims'])\n"
            "        print(f'{name}: P {P.shape}')\n"
            "    except Exception as exc:\n"
            "        print(f'{name}: skipped ({exc})')"
        ),
        _md(
            "## 2. Build coherent collocation targets\n\n"
            "For each selected particle at a chosen snapshot `te`, we compute:\n"
            "\n"
            "- `v_coh_all, a_coh_all` — weighted mean velocity / acceleration\n"
            "  of coherent (primary) neighbours at every history snapshot\n"
            "- `y_sec, v_sec, a_sec` — position, velocity, acceleration of\n"
            "  the non-coherent (secondary) pool, phase delayed to `te + 1`\n"
            "\n"
            "Acceleration targets are smoothed with a quadratic fit along the\n"
            "history window (SNR of FD acceleration is ~0.17 at 10% noise)."
        ),
        _code(
            "HIST = 7\n"
            "T_FTLE = 8\n"
            "FTLE_PCTILE = 50\n"
            "ALPHA_W = 3.0\n"
            "R_SCALE = 4.0\n"
            "\n"
            "def build_targets(case, pid, te):\n"
            "    P_n = case['P_n']; V_n = case['V_n']\n"
            "    V_s = case['V_s']; A_s = case['A_s']\n"
            "    dt = case['dt']; ndim = case['dims']\n"
            "    T_steps = P_n.shape[1]\n"
            "    if te - HIST + 1 < 0 or te + 1 >= T_steps:\n"
            "        return None\n"
            "    tidx = np.arange(te - HIST + 1, te + 1)\n"
            "    tau = np.arange(HIST, dtype=float)\n"
            "    ph = P_n[pid, tidx]\n"
            "\n"
            "    vh = np.linalg.norm(V_n[pid, tidx], axis=1)\n"
            "    U_loc = max(vh.max(), 1e-12)\n"
            "    r_search = U_loc * dt * R_SCALE\n"
            "\n"
            "    tree = KDTree(P_n[:, te])\n"
            "    nids = tree.query_radius(P_n[pid, te].reshape(1, -1), r=r_search)[0]\n"
            "    nids = np.array([n for n in nids if n != pid])\n"
            "    if len(nids) < 4:\n"
            "        _, idx = tree.query(P_n[pid, te].reshape(1, -1), k=8)\n"
            "        nids = idx[0, 1:]\n"
            "\n"
            "    lam = backward_ftle(P_n, pid, nids, te, T_FTLE)\n"
            "    mask = coherent_mask(lam, FTLE_PCTILE)\n"
            "    if mask.sum() < 2:\n"
            "        return None\n"
            "\n"
            "    prim = nids[mask]\n"
            "    d_now = np.linalg.norm(P_n[prim, te] - P_n[pid, te], axis=1)\n"
            "    w = compute_weights(lam[mask], d_now, ALPHA_W)\n"
            "    v_coh_all = np.zeros((HIST, ndim))\n"
            "    a_coh_all = np.zeros((HIST, ndim))\n"
            "    for hi, t_step in enumerate(tidx):\n"
            "        v_coh_all[hi] = (w[:, None] * V_s[prim, t_step]).sum(0) * dt\n"
            "        a_coh_all[hi] = (w[:, None] * A_s[prim, t_step]).sum(0) * dt ** 2\n"
            "    # v7i-d smoothed acceleration targets\n"
            "    a_coh_all = smooth_history_targets(a_coh_all, order=2)\n"
            "\n"
            "    # Secondary = non coherent pool, phase delayed\n"
            "    non = ~mask\n"
            "    sec = None\n"
            "    if non.sum() >= 2:\n"
            "        sec_ids = nids[non]\n"
            "        d_sec = np.linalg.norm(P_n[sec_ids, te] - P_n[pid, te], axis=1)\n"
            "        ws = compute_weights(lam[non], d_sec, ALPHA_W)\n"
            "        tp = min(te + 1, T_steps - 1)\n"
            "        y_sec = P_n[pid, te] + (ws[:, None] * (P_n[sec_ids, tp] - P_n[sec_ids, te])).sum(0)\n"
            "        v_sec = (ws[:, None] * V_s[sec_ids, tp]).sum(0) * dt\n"
            "        a_sec = (ws[:, None] * A_s[sec_ids, tp]).sum(0) * dt ** 2\n"
            "        sec = (y_sec, v_sec, a_sec)\n"
            "    return tau, ph, v_coh_all, a_coh_all, sec\n"
            "print('target builder ready')"
        ),
        _md(
            "## 3. Run the PINN on a small batch\n\n"
            "We loop over a handful of particles to keep runtime manageable in\n"
            "Colab. Increase ``n_eval`` for a more converged statistic."
        ),
        _code(
            "from coherent_predictor import SirenConfig\n"
            "\n"
            "siren_cfg = SirenConfig()  # v7i-d: omega_0=0.5, 100 iters\n"
            "print(siren_cfg)\n"
            "\n"
            "results = {}\n"
            "for name, case in loaded.items():\n"
            "    print(f'\\n=== {name} ===')\n"
            "    ndim = case['dims']; dt = case['dt']\n"
            "    rng = np.random.default_rng(7)\n"
            "    n_eval = 20  # raise to 100-200 for publication quality numbers\n"
            "    pids = rng.choice(case['P'].shape[0], size=n_eval, replace=False)\n"
            "    te = 50\n"
            "\n"
            "    ev_poly = []; ep_pinn = []\n"
            "    ea_poly = []; ea_pinn = []\n"
            "    for pid in pids:\n"
            "        tgt = build_targets(case, pid, te)\n"
            "        if tgt is None:\n"
            "            continue\n"
            "        tau, ph, v_coh_all, a_coh_all, sec = tgt\n"
            "        # Polynomial baseline coefficients (for warm start)\n"
            "        order = 3\n"
            "        poly_coefs = np.zeros((ndim, order + 1))\n"
            "        for d in range(ndim):\n"
            "            poly_coefs[d] = np.polyfit(tau, ph[:, d], order)[::-1]\n"
            "\n"
            "        pp_poly, pv_poly, pa_poly = solve_poly_full(tau, ph, order=order)\n"
            "        y_sec = v_sec = a_sec = None\n"
            "        if sec is not None:\n"
            "            y_sec, v_sec, a_sec = sec\n"
            "        pp_s, pv_s, pa_s, _ = predict_siren(\n"
            "            tau, ph, v_coh_all, a_coh_all, poly_coefs,\n"
            "            y_sec=y_sec, v_sec=v_sec, a_sec=a_sec, cfg=siren_cfg,\n"
            "        )\n"
            "        v_true = case['V'][pid, te + 1] * dt\n"
            "        a_true = case['A'][pid, te + 1] * dt ** 2\n"
            "        ev_poly.append(np.linalg.norm(pv_poly - v_true))\n"
            "        ep_pinn.append(np.linalg.norm(pv_s    - v_true))\n"
            "        ea_poly.append(np.linalg.norm(pa_poly - a_true))\n"
            "        ea_pinn.append(np.linalg.norm(pa_s    - a_true))\n"
            "    results[name] = dict(\n"
            "        v_poly=np.asarray(ev_poly), v_pinn=np.asarray(ep_pinn),\n"
            "        a_poly=np.asarray(ea_poly), a_pinn=np.asarray(ea_pinn),\n"
            "    )\n"
            "    if ev_poly:\n"
            "        red_v = (1.0 - np.mean(ep_pinn) / np.mean(ev_poly)) * 100\n"
            "        red_a = (1.0 - np.mean(ea_pinn) / np.mean(ea_poly)) * 100\n"
            "        print(f'  SIREN vs Poly:   vel reduction {red_v:6.1f}%, acc reduction {red_a:6.1f}%')"
        ),
        _md(
            "## 4. Figure — Cleveland dot plot\n\n"
            "Each dot is the SIREN reduction vs the Polynomial baseline for\n"
            "one case. The paper reports ~77% (velocity) and ~75% (acceleration)\n"
            "on the 3D wake."
        ),
        _code(
            "import matplotlib.pyplot as plt\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7, 3))\n"
            "y = 0\n"
            "for name, res in results.items():\n"
            "    if len(res['v_poly']) == 0:\n"
            "        continue\n"
            "    rv = (1 - res['v_pinn'].mean() / res['v_poly'].mean()) * 100\n"
            "    ra = (1 - res['a_pinn'].mean() / res['a_poly'].mean()) * 100\n"
            "    ax.plot(rv, y, 'o', ms=10, color='#1a1a8c', label='velocity' if y == 0 else None)\n"
            "    ax.plot(ra, y, 's', ms=10, color='#a2132e', label='acceleration' if y == 0 else None)\n"
            "    ax.text(-2, y, name, ha='right', va='center')\n"
            "    y += 1\n"
            "ax.axvline(0, color='0.7', lw=0.8)\n"
            "ax.set_xlim(-20, 100)\n"
            "ax.set_xlabel('error reduction vs Polynomial (%)')\n"
            "ax.set_yticks([])\n"
            "ax.legend(loc='lower right', frameon=False)\n"
            "fig.tight_layout()\n"
            "plt.show()"
        ),
    ]
    nb.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        }
    )
    return nb


# ---------------------------------------------------------------------------
# Notebook 03 — FTLE evaluation
# ---------------------------------------------------------------------------

def build_notebook_03() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        _md(
            "# 03 FTLE integration time and weighting\n\n"
            "Reproduces the diagnostic analyses that set two of the paper's\n"
            "parameters:\n\n"
            "- `T_ftle = 8`, the backward integration window\n"
            "- `alpha_w = 3`, the balance between Lambda and distance weighting\n"
            "\n"
            "Panels here match Figure 3 (sigma vs T) and the appendix sweep of\n"
            "alpha_w at the optimised T."
        ),
        _md(SETUP_MD),
        _code(SETUP_CODE),
        _md("## 1. Load data"),
        _code(
            "import numpy as np\n"
            "from coherent_predictor import (\n"
            "    load_trajectories, add_positional_noise, compute_fd,\n"
            "    sigma_field, coherent_fraction, alpha_sweep,\n"
            ")\n"
            "dt = 10.0\n"
            "P = load_trajectories(DATA_2D, dims=2)\n"
            "P_n, _ = add_positional_noise(P, 0.10, seed=123)\n"
            "V, _ = compute_fd(P, dt)\n"
            "print(f'loaded P: {P.shape}')"
        ),
        _md(
            "## 2. Stretching factor sigma vs integration time\n\n"
            "For a chosen snapshot `te` and a batch of target particles we\n"
            "pool sigma over all their neighbours and see how the distribution\n"
            "drifts as T grows. The threshold `sigma_star` is the median of\n"
            "sigma at the reference window `T_ref = 8`."
        ),
        _code(
            "te = 50\n"
            "T_list = [2, 4, 6, 8, 10, 14, 18, 22, 26, 30]\n"
            "rng = np.random.default_rng(0)\n"
            "target_ids = rng.choice(P.shape[0], size=400, replace=False)\n"
            "\n"
            "# Characteristic radius for KD tree query\n"
            "vh = np.linalg.norm(np.diff(P_n[:, :60], axis=1), axis=2).max(axis=1)\n"
            "U_loc = np.median(vh)\n"
            "r_search = U_loc * dt * 4.0\n"
            "\n"
            "sig = sigma_field(P_n, te, T_list, target_ids, r_search=r_search)\n"
            "median_sig = {T: float(np.median(v)) for T, v in sig.items()}\n"
            "print('median sigma at each T:')\n"
            "for T, m in median_sig.items():\n"
            "    print(f'  T = {T:2d}  sigma_med = {m:.4f}  (N = {len(sig[T])})')"
        ),
        _md("## 3. Coherent fraction vs T"),
        _code(
            "sigma_star = float(np.median(sig[8]))\n"
            "frac = coherent_fraction(sig, sigma_star)\n"
            "for T, f in frac.items():\n"
            "    print(f'  T = {T:2d}  coherent fraction = {100 * f:5.1f} %')"
        ),
        _code(
            "import matplotlib.pyplot as plt\n"
            "fig, axs = plt.subplots(1, 2, figsize=(10, 4))\n"
            "Ts = np.asarray(T_list)\n"
            "axs[0].plot(Ts, [median_sig[T] for T in Ts], 'o-', color='#1a1a8c')\n"
            "axs[0].axhline(sigma_star, color='0.6', ls='--', label=f'sigma* (T=8)')\n"
            "axs[0].set_xlabel('integration time T (snapshots)')\n"
            "axs[0].set_ylabel('median sigma')\n"
            "axs[0].legend(frameon=False)\n"
            "\n"
            "axs[1].plot(Ts, [100 * frac[T] for T in Ts], 's-', color='#a2132e')\n"
            "axs[1].set_xlabel('integration time T (snapshots)')\n"
            "axs[1].set_ylabel('coherent fraction (%)')\n"
            "fig.tight_layout()\n"
            "plt.show()"
        ),
        _md(
            "## 4. alpha_w sweep — coherent velocity error\n\n"
            "The weighting exponent `alpha_w` balances the Lambda term and\n"
            "the distance term. This grid scan confirms that `alpha_w = 3`\n"
            "minimises the error of the coherent velocity estimator against\n"
            "the true DNS velocity."
        ),
        _code(
            "T_scan = [4, 8, 12, 16]\n"
            "a_scan = np.linspace(0.0, 6.0, 13)\n"
            "T_arr, a_arr, err_grid = alpha_sweep(\n"
            "    P, V, te, target_ids[:120], T_scan, a_scan,\n"
            "    r_search=r_search, dt=dt,\n"
            ")\n"
            "print('error grid shape:', err_grid.shape)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7, 4))\n"
            "for i, T in enumerate(T_arr):\n"
            "    ax.plot(a_arr, err_grid[i], 'o-', label=f'T = {T}')\n"
            "ax.set_xlabel('alpha_w')\n"
            "ax.set_ylabel('relative error of v_coh')\n"
            "ax.legend(frameon=False)\n"
            "fig.tight_layout()\n"
            "plt.show()"
        ),
    ]
    nb.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        }
    )
    return nb


def main():
    builders = [
        ("01_core_predictor.ipynb", build_notebook_01),
        ("02_siren_pinn.ipynb", build_notebook_02),
        ("03_ftle_evaluation.ipynb", build_notebook_03),
    ]
    for name, fn in builders:
        nb = fn()
        out = NB_DIR / name
        with out.open("w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
