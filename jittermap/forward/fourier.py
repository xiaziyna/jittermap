"""Fourier-domain compression of uniformly sampled time series.

For uniform sampling of a full rotation period (times t_n = 2*pi*n/(N*omega),
n = 0..N-1), the forward model is a frequency comb: each channel's signal
contains only the harmonics m = -L..L of the rotation rate::

    y(t) = sum_{l} sum_{m} e^{-i m omega t} v^{(l)}_m s_{l,m}
    v^{(l)} = M_l C(beta) a_l

Projecting onto the comb therefore compresses each channel from N samples
to 2L+1 Fourier coefficients with no information loss, and the design
matrix becomes block-diagonal-by-frequency::

    F[m, (l, m')] = v^{(l)}_m * delta_{m m'}

The compressed coefficients satisfy the conjugate symmetry
f_{-m} = conj(f_m) for real signals, so they can be repacked into 2L+1
real numbers (fourier_to_real / real_to_fourier).

Requires N >= 2L+1 (no aliasing of the comb).
"""

import numpy as np

from jittermap.harmonics.indexing import SHIndexer
from jittermap.forward.wigner import wigner_M, rotation_funcs


def compress(y, l_max):
    """Project a uniformly sampled time series onto the frequency comb.

    Parameters
    ----------
    y : ndarray, shape (N,)
        Signal sampled at t_n = 2*pi*n/(N*omega), n = 0..N-1
        (uniform, full period, endpoint excluded).
    l_max : int
        Maximum harmonic; returns coefficients for m = -l_max..l_max.

    Returns
    -------
    f : complex ndarray, shape (2*l_max+1,)
        f[m + l_max] = (1/N) sum_n y_n e^{+i m omega t_n}, the coefficient
        of the basis function e^{-i m omega t} in y.
    """
    y = np.asarray(y)
    N = y.shape[0]
    if N < 2 * l_max + 1:
        raise ValueError(f"need N >= 2*l_max+1 samples (N={N}, l_max={l_max})")
    spec = np.fft.ifft(y)  # ifft[k] = (1/N) sum_n y_n e^{+2i pi k n / N}
    m_vals = np.arange(-l_max, l_max + 1)
    return spec[m_vals % N]


def fourier_design_matrix(l_max, inclination, A_lm, omega=1.0):
    """Fourier-domain design matrix F of shape (2L+1, (L+1)^2) such that
    compress(D(times, beta) @ s) == F @ s for uniform full-period sampling.

    Row m holds v^{(l)}_m at column (l, m) for every degree l >= abs(m).
    """
    L = l_max
    sh = SHIndexer(l_max=L)
    F = np.zeros((2 * L + 1, sh.total_coeffs), dtype=complex)
    for lp in range(L + 1):
        _, C_func = rotation_funcs(lp, omega)
        C = C_func(inclination)
        C_a = C.dot(A_lm[lp, L - lp: L + lp + 1])
        v = wigner_M(lp) @ C_a  # (2lp+1,)
        F[L - lp: L + lp + 1, lp * lp: (lp + 1) ** 2] = np.diag(v)
    return F


def fourier_to_real(f):
    """Repack a conjugate-symmetric coefficient vector f (2L+1,) into
    2L+1 real numbers: [f_0, Re f_1, Im f_1, Re f_2, Im f_2, ...]."""
    f = np.asarray(f)
    L = (f.shape[0] - 1) // 2
    out = np.empty(2 * L + 1)
    out[0] = f[L].real
    out[1::2] = f[L + 1:].real
    out[2::2] = f[L + 1:].imag
    return out


def real_to_fourier(x):
    """Inverse of fourier_to_real: rebuild the conjugate-symmetric
    complex coefficient vector."""
    x = np.asarray(x)
    L = (x.shape[0] - 1) // 2
    f = np.zeros(2 * L + 1, dtype=complex)
    f[L] = x[0]
    f[L + 1:] = x[1::2] + 1j * x[2::2]
    f[:L] = np.conj(f[L + 1:][::-1])
    return f


def time_noise_to_fourier(sigma, n, l_max, rng=None):
    """Draw i.i.d. Gaussian time-domain noise of standard deviation sigma
    over n samples and return its compressed Fourier coefficients.

    Each compressed coefficient has variance sigma^2 / n (complex, with
    the conjugate-symmetry correlation of a real time series).
    """
    if rng is None:
        rng = np.random.default_rng()
    noise_t = rng.normal(0, sigma, n)
    return compress(noise_t, l_max)
