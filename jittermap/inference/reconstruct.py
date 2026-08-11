"""High-level surface reconstruction: joint inclination + surface MAP fit.

This wraps the full pipeline used for the reconstruction galleries in the
accompanying paper: given time series in any subset of the three channels
(x/y photocenter shifts and photometry), estimate the inclination by
profile grid search and reconstruct the surface by GMRF-regularized MAP.
"""

from dataclasses import dataclass, field

import numpy as np

from jittermap.forward.design import ForwardModel, design_matrix_vandermonde
from jittermap.inference.inversion import (gmrf_precision_diag, solve_ridge,
                                           default_lambda)
from jittermap.inference.inclination import estimate_inclination


@dataclass
class ReconstructionResult:
    """Result of a surface reconstruction.

    Attributes
    ----------
    s_hat : complex ndarray
        Estimated SH coefficients.
    inclination : float
        Inclination used for the fit (estimated unless it was fixed).
    channels : str
        Channels used ('x', 'y', 'p' subsets).
    lam : float
        Regularization strength used.
    extras : dict
        Diagnostics (optimizer output, residual norm, ...).
    """
    s_hat: np.ndarray
    inclination: float
    channels: str
    lam: float
    extras: dict = field(default_factory=dict)


def reconstruct(y, times, l_max, channels="xyp", inclination=None,
                omega=1.0, lam=None, snr_total=None, alpha_reg=1.0,
                scale_reg=1.0, include_l0=False, inc_bounds=(0.1, np.pi / 2 - 0.1),
                model=None):
    """Reconstruct a stellar surface from observed time series.

    Parameters
    ----------
    y : ndarray or dict
        Observations: a dict {channel: (N,) array} or a stacked array
        (len(channels)*N,) in channel order.
    times : ndarray, shape (N,)
        Observation times (rotation phase = omega * t).
    l_max : int
        Fit degree.
    channels : str
        Subset of 'xyp'.
    inclination : float or None
        If None (default), estimated by profile grid search.
    lam : float or None
        Regularization strength; if None, uses the gallery schedule
        default_lambda(snr_total, l_max).
    snr_total : float or None
        Total signal-to-noise (for the default lambda schedule only).
    model : ForwardModel or None
        Reuse a precomputed ForwardModel (recommended in sweeps).

    Returns
    -------
    ReconstructionResult
    """
    if model is None:
        model = ForwardModel(times, l_max, omega=omega)
    if isinstance(y, dict):
        y_vec = np.concatenate([np.asarray(y[c]) for c in channels])
    else:
        y_vec = np.asarray(y)

    if lam is None:
        lam = default_lambda(snr_total, l_max)

    extras = {}
    if inclination is None:
        inclination, opt = estimate_inclination(
            y_vec, channels, l_max, model.kernels, model._precomp,
            lam=lam, alpha_reg=alpha_reg, bounds=inc_bounds)
        extras["optimizer"] = opt

    D = np.vstack([
        design_matrix_vandermonde(l_max, inclination, model.kernels[c],
                                  *model._precomp)
        for c in channels
    ])
    q = gmrf_precision_diag(l_max, alpha=alpha_reg, scale=scale_reg,
                            include_l0=include_l0)
    s_hat = solve_ridge(D, y_vec, q, lam)
    resid = y_vec - D.dot(s_hat)
    extras["residual_norm"] = float(np.linalg.norm(resid))

    return ReconstructionResult(s_hat=s_hat, inclination=float(inclination),
                                channels=channels, lam=lam, extras=extras)
