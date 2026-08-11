"""GMRF-regularized MAP inversion of astrometric / photometric time series.

Convention: the GMRF prior on SH coefficients s is s ~ CN(mu, Q^{-1})
with Q = diag(q), q_l = (l / scale)^alpha. Larger l -> higher precision
-> more shrinkage, encoding the expectation that high-frequency surface
features have smaller amplitude.
"""

import numpy as np

from jittermap.harmonics.indexing import SHIndexer


def gmrf_precision_diag(l_max, alpha=1.0, scale=2.0, include_l0=False,
                        eps=1e-12):
    """Diagonal precision vector for the GMRF prior in SHIndexer ordering.

    q_l = (l / scale)^alpha for l >= 1; the monopole gets near-zero
    precision unless include_l0 is set.
    """
    sh = SHIndexer(l_max=l_max)
    q = np.zeros(sh.total_coeffs, dtype=float)
    for l in range(l_max + 1):
        if l == 0 and not include_l0:
            weight = eps
        else:
            weight = (max(l, eps) / max(scale, eps)) ** alpha
        q[sh.get_l_indices(l)] = weight
    return q


def gmrf_covariance_diag(l_max, alpha=1.0, scale=2.0, include_l0=False,
                         eps=1e-12):
    """Diagonal covariance vector: the marginal variance 1/q_l per coefficient."""
    q = gmrf_precision_diag(l_max, alpha, scale, include_l0, eps)
    return 1.0 / q


def sample_from_prior(q, mu=None, rng=None):
    """Sample s ~ CN(mu, diag(1/q)); variance split equally between the
    real and imaginary parts."""
    if rng is None:
        rng = np.random.default_rng()
    p = len(q)
    std = 1.0 / np.sqrt(2.0 * q)
    s = (rng.standard_normal(p) + 1j * rng.standard_normal(p)) * std
    if mu is not None:
        s = s + mu
    return s


def solve_ridge(D, y, q, lam):
    """Ridge / MAP estimate: minimize ||D s - y||^2 + lam * s^H diag(q) s.

    Uses augmented least squares for numerical stability. With lam <= 0
    (or None) this reduces to plain least squares.
    """
    if lam is None or lam <= 0:
        s_hat, *_ = np.linalg.lstsq(D, y, rcond=None)
        return s_hat
    sqrt_q = np.sqrt(q)
    A = np.vstack((D, np.sqrt(lam) * np.diag(sqrt_q)))
    b = np.concatenate((y, np.zeros(D.shape[1], dtype=complex)))
    s_hat, *_ = np.linalg.lstsq(A, b, rcond=None)
    return s_hat


def solve_ridge_real_constraint(D, y, q, lam, R, mu=None):
    """Ridge / MAP under the real-surface constraint via the
    reparameterization s = R r (R from build_real_surface_transform):
    minimizes ||D R r - y||^2 + lam * (R r - mu)^H diag(q) (R r - mu).
    """
    if mu is None:
        mu = np.zeros(R.shape[0], dtype=complex)
    if lam is None or lam <= 0:
        r_hat, *_ = np.linalg.lstsq(D @ R, y, rcond=None)
        return R @ r_hat
    sqrt_q = np.sqrt(q)
    A = np.vstack((D @ R, np.sqrt(lam) * (sqrt_q[:, None] * R)))
    b = np.concatenate((y, np.sqrt(lam) * (sqrt_q * mu)))
    r_hat, *_ = np.linalg.lstsq(A, b, rcond=None)
    return R @ r_hat


def default_lambda(snr_total, l_fit):
    """Regularization schedule tuned on the reconstruction gallery:
    more smoothing at low SNR, and extra smoothing for high-degree fits
    at moderate SNR (suppresses streak artifacts)."""
    if snr_total is None:
        return 2e-4
    if snr_total <= 20:
        return 2e-2
    if l_fit >= 20 and snr_total <= 200:
        return 2e-3
    return 2e-4
