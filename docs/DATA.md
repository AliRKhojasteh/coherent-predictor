# Data

## Short answer — can I just pull the repo and run?

Yes. A small demo subset of the 2D HIT DNS (2000 particles, 80 snapshots,
~1 MB) ships inside the repo at `data/demo_2D_HIT.npz`. Every notebook
defaults to this file, so `Run All` works straight away on Colab.

If you want the full Table 2 numbers from the paper, swap the demo file
for the full DNS as described below.

## Why the full DNS is not in git

The full files are large:

| File | Size | What |
|---|---:|---|
| `1_10000_dt10_comparison.mat` | 87 MB | 2D HIT DNS (30000 particles, 100 snaps) |
| `P_RK4_1_100.mat` | 67 MB | 3D cylinder wake (Re = 3900) |

GitHub warns at 50 MB and rejects at 100 MB per file. Git itself is not
built for binary blobs. Three options are standard for academic datasets:

1. **GitHub Release artifact** — up to 2 GB per file, tracks the tag, best
   for binary distribution tied to a release.
2. **Zenodo / figshare DOI** — cite-able, versioned, permanent. This is the
   community convention when the data accompanies a paper.
3. **Institutional mirror** — SJTU, LML or author's own URL.

Our plan: after the JFM acceptance, upload both `.mat` files to **Zenodo**
(DOI linked from the paper), and mirror them on a **GitHub Release** so the
notebooks can fetch them with a single `wget`.

## What the notebooks look for

Every setup cell defines two paths:

```python
DATA_2D = "data/demo_2D_HIT.npz"   # ships with the repo
DATA_3D = "data/P_RK4_1_100.mat"    # for notebook 02 only
```

Replace either with a full-size `.mat` file and rerun the notebook. The
`load_trajectories` helper accepts both formats transparently.

## Downloading the full DNS

Once the Zenodo deposit is live, the URLs will be

```
https://zenodo.org/records/<id>/files/1_10000_dt10_comparison.mat
https://zenodo.org/records/<id>/files/P_RK4_1_100.mat
```

and a Colab cell like

```python
!wget -q -O data/1_10000_dt10_comparison.mat https://zenodo.org/records/<id>/files/1_10000_dt10_comparison.mat
DATA_2D = "data/1_10000_dt10_comparison.mat"
```

will replace the demo subset.

## How the demo file was built

From the full `.mat`:

```python
from coherent_predictor import load_trajectories
import numpy as np

P = load_trajectories("1_10000_dt10_comparison.mat", dims=2)  # (30000, 100, 2)
rng = np.random.default_rng(2026)
idx = rng.choice(P.shape[0], size=2000, replace=False)
P_demo = P[idx, :80, :].astype(np.float32)
np.savez_compressed("data/demo_2D_HIT.npz",
                    P_RK4=P_demo, dt=10.0, dt_phys=0.1,
                    note="Subset, 2000 random particles, first 80 snaps")
```

2000 particles is enough to give a clear P < P+S hierarchy on both the
velocity and acceleration channels, at a cost of slightly weaker percentage
numbers compared with the paper's 30000 particle pool.

## Reference: expected numbers

| dataset | method | velocity reduction | acceleration reduction |
|---|---|---:|---:|
| demo (2000 particles) | P   | ~54% | ~73% |
| demo (2000 particles) | P+S | ~64% | ~82% |
| full DNS (30000 particles) | P   | ~57% | ~75% |
| full DNS (30000 particles) | P+S | ~74% | ~80% |

Demo figures use fewer neighbours and a coarser particle pool, so the P+S
lift over P is visible but less pronounced.
