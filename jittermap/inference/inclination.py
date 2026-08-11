"""Inclination estimation by penalized-residual grid search.

The inclination beta enters the design matrix nonlinearly; it is
estimated by minimizing the (regularized) profile objective
    J(beta) = || y - D(beta) s_hat(beta) ||^2 + lam * s_hat^H Q s_hat
over beta with a bounded scalar minimizer, where s_hat(beta) is the
ridge/MAP solution at that inclination.
"""

import numpy as np
from scipy.optimize import minimize_scalar

from jittermap.inference.inversion import gmrf_precision_diag, solve_ridge
from jittermap.forward.design import design_matrix_vandermonde


def profile_objective(inc, y_vec, channels, l_max, kernels, precomp,
                      lam=2e-4, alpha_reg=1.0, include_l0=False):
    """Profile objective J(beta) for a stacked observation vector.

    Parameters
    ----------
    inc : float
        Trial inclination (radians).
    y_vec : ndarray
        Stacked observations for the requested channels (in order).
    channels : str
        Subset of 'xyp'; y_vec must stack the same channels in the same order.
    l_max : int
        Fit degree.
    kernels : dict
        {'x': A_lm_x, 'y': A_lm_y, 'p': A_lm_photo} (only used channels needed).
    precomp : tuple
        (B0_list, C_funcs, phases_list) from precompute_vandermonde.
    """
    D = np.vstack([
        design_matrix_vandermonde(l_max, inc, kernels[c], *precomp)
        for c in channels
    ])
    if lam is None or lam <= 0:
        s_hat, *_ = np.linalg.lstsq(D, y_vec, rcond=None)
        resid = y_vec - D.dot(s_hat)
        return float(np.vdot(resid, resid).real)
    q = gmrf_precision_diag(l_max, alpha=alpha_reg, scale=1.0,
                            include_l0=include_l0)
    s_hat = solve_ridge(D, y_vec, q, lam)
    resid = y_vec - D.dot(s_hat)
    pen = lam * np.sum(q * (np.abs(s_hat) ** 2))
    return float(np.vdot(resid, resid).real + pen)


def estimate_inclination(y_vec, channels, l_max, kernels, precomp,
                         lam=2e-4, alpha_reg=1.0, include_l0=False,
                         bounds=(0.1, np.pi / 2 - 0.1), xatol=1e-3):
    """Estimate the stellar inclination from stacked observations.

    Returns
    -------
    inc_hat : float
        Estimated inclination in radians.
    result : OptimizeResult
        Full scipy result for diagnostics.
    """
    result = minimize_scalar(
        lambda inc: profile_objective(inc, y_vec, channels, l_max, kernels,
                                      precomp, lam=lam, alpha_reg=alpha_reg,
                                      include_l0=include_l0),
        bounds=bounds, method="bounded", options={"xatol": xatol})
    return float(result.x), result
