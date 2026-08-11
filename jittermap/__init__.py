"""jittermap: stellar surface mapping from astrometric jitter and photometry.

The forward model maps a stellar surface, expanded in complex spherical
harmonics s_{l,m}, to time series of the astrometric photocenter shift
(x, y) and the disk-integrated photometric flux as the star rotates at
inclination beta:

    y_c(t) = Re[ D_c(t, beta) s ],
    D_c    = B(t) C(beta) a_c,l    per degree l, channel c in {x, y, p}

where a_c,l are the visible-hemisphere moment kernels and B, C are Wigner
rotation blocks for the spin and inclination (see jittermap.forward).
Inference recovers s (and beta) from any subset of the channels by
GMRF-regularized MAP inversion (see jittermap.inference).

Quickstart
----------
>>> import numpy as np, jittermap as jm
>>> times = np.linspace(0, 2*np.pi, 200, endpoint=False)
>>> s = jm.multispot_surface([(30, 150, 10.0), (50, 260, 8.0)], l_max=10)
>>> fm = jm.ForwardModel(times, l_max=10)
>>> y = fm.observe(s, inclination=0.6, channels='xyp')
>>> res = jm.reconstruct(y, times, l_max=10, channels='xyp')
"""

__version__ = "0.1.0"

from jittermap.harmonics.indexing import (SHIndexer, build_real_surface_transform,
                                          project_real_surface, mse)
from jittermap.harmonics.spots import generate_spot, spot_area_fraction
from jittermap.harmonics.surfaces import (random_surface, multispot_surface,
                                          rfrac_to_deg, deg_to_rfrac)
from jittermap.forward.wigner import wigner_M, rotation_funcs, precompute_rotations
from jittermap.forward.kernels import compute_A_lm, compute_A_lm_photo
from jittermap.forward.design import (ForwardModel, design_matrix_reference,
                                      precompute_vandermonde,
                                      design_matrix_vandermonde)
from jittermap.forward.fourier import (compress, fourier_design_matrix,
                                       fourier_to_real, real_to_fourier)
from jittermap.inference.inversion import (gmrf_precision_diag,
                                           gmrf_covariance_diag,
                                           sample_from_prior, solve_ridge,
                                           solve_ridge_real_constraint,
                                           default_lambda)
from jittermap.inference.inclination import estimate_inclination, profile_objective
from jittermap.inference.reconstruct import reconstruct, ReconstructionResult

__all__ = [
    "SHIndexer", "build_real_surface_transform", "project_real_surface", "mse",
    "generate_spot", "spot_area_fraction",
    "random_surface", "multispot_surface", "rfrac_to_deg", "deg_to_rfrac",
    "wigner_M", "rotation_funcs", "precompute_rotations",
    "compute_A_lm", "compute_A_lm_photo",
    "ForwardModel", "design_matrix_reference", "precompute_vandermonde",
    "design_matrix_vandermonde",
    "compress", "fourier_design_matrix", "fourier_to_real", "real_to_fourier",
    "gmrf_precision_diag", "gmrf_covariance_diag", "sample_from_prior",
    "solve_ridge", "solve_ridge_real_constraint", "default_lambda",
    "estimate_inclination", "profile_objective",
    "reconstruct", "ReconstructionResult",
]
