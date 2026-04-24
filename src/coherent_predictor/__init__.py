"""Coherent motions to predict Lagrangian trajectories.

Companion code for Rahimi Khojasteh et al., J. Fluid Mech., 2026.

Three predictors are exposed:
    solve_poly_full   polynomial extrapolation baseline
    solve_coherent    polynomial + v/a constraints from coherent neighbours (P or P+S)
    predict_siren     tiny SIREN+Fourier PINN with collocation loss (appendix C)

plus helpers for FTLE based neighbour classification, polynomial smoothing
of noisy derivatives and finite difference operators.
"""

from .data_io import add_positional_noise, load_trajectories, median_nn_distance
from .derivatives import L_A, L_V, compute_fd, compute_smoothed, smooth_history_targets
from .ftle import backward_ftle, coherent_mask, compute_weights
from .ftle_eval import alpha_sweep, coherent_fraction, sigma_field
from .predictor import (
    PredictorConfig,
    predict_one_particle,
    solve_coherent,
    solve_poly_full,
)

try:
    from .siren_pinn import SirenConfig, predict_siren
    _HAVE_SIREN = True
except ImportError:  # autograd missing
    SirenConfig = None  # type: ignore
    predict_siren = None  # type: ignore
    _HAVE_SIREN = False

__all__ = [
    "add_positional_noise",
    "load_trajectories",
    "median_nn_distance",
    "compute_fd",
    "compute_smoothed",
    "smooth_history_targets",
    "L_V",
    "L_A",
    "backward_ftle",
    "coherent_mask",
    "compute_weights",
    "sigma_field",
    "coherent_fraction",
    "alpha_sweep",
    "PredictorConfig",
    "solve_poly_full",
    "solve_coherent",
    "predict_one_particle",
    "predict_siren",
    "SirenConfig",
]

__version__ = "1.0.0"
