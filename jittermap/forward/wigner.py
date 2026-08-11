"""Wigner rotation operators for a spinning star viewed at inclination beta.

Convention (as in the accompanying paper): the design matrix factors as
    D(t, beta) = B(t) C(beta) a_l
per degree l, with the two rotation blocks defined through Wigner D
matrices in the sympy.physics.quantum.spin convention,
    B(t)     = D^l(omega t + pi/2, pi/2, pi/2)
    C(beta)  = D^l(-beta + pi/2,   pi/2, pi/2)
where rows are indexed by m = -l..l and columns by m' = -l..l.

Since D^l_{m m'}(alpha, beta, gamma) = e^{-i m alpha} d^l_{m m'}(beta)
e^{-i m' gamma} and the middle angle is fixed at pi/2, both blocks are
phase-diagonal modulations of a single fixed matrix per degree,
    M_l = D^l(pi/2, pi/2, pi/2),   (M_l)_{m m'} = (-i)^{m+m'} d^l_{m m'}(pi/2),
via
    B(t)    = diag(e^{-i m omega t}) M_l
    C(beta) = diag(e^{+i m beta})   M_l.

The on-disk cache therefore stores only the numeric M_l (complex128),
computed from the exact Wigner sum formula at beta = pi/2 in high-precision
arithmetic — no pickled symbolic objects, stable across library versions.
"""

import os
from fractions import Fraction
from functools import lru_cache
from math import factorial

import numpy as np

_PKG_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "wigner")


def user_cache_dir():
    """Writable cache directory for degrees beyond the shipped tables."""
    base = os.environ.get("JITTERMAP_CACHE_DIR")
    if base is None:
        base = os.path.join(os.path.expanduser("~"), ".cache", "jittermap")
    path = os.path.join(base, "wigner")
    return path


def _wigner_d_pi2_exact(l):
    """Exact Wigner small-d matrix d^l(pi/2) as high-precision floats.

    Uses the Wigner sum formula at beta = pi/2, where
    cos(beta/2) = sin(beta/2) = 1/sqrt(2), so
        d^l_{m m'}(pi/2) = 2^{-l} sqrt((l+m)!(l-m)!(l+m')!(l-m')!) * S
        S = sum_k (-1)^{k - m' + m} / [ (l+m'-k)! k! (l-k-m)! (k-m'+m)! ]
    with the sum over all k giving non-negative factorial arguments.
    The rational sum S is evaluated exactly with Fraction, and the square
    root at 50-digit precision via sympy, so the returned float64 matrix
    is correctly rounded.

    The (m, m') convention is validated in tests against
    sympy.physics.quantum.spin.Rotation.d.
    """
    import sympy as sp

    dim = 2 * l + 1
    d = np.zeros((dim, dim), dtype=float)
    for m in range(-l, l + 1):
        for mp in range(-l, l + 1):
            k_min = max(0, mp - m)
            k_max = min(l + mp, l - m)
            S = Fraction(0)
            for k in range(k_min, k_max + 1):
                denom = (factorial(l + mp - k) * factorial(k)
                         * factorial(l - k - m) * factorial(k - mp + m))
                S += Fraction((-1) ** (k - mp + m), denom)
            if S == 0:
                continue
            F = (factorial(l + m) * factorial(l - m)
                 * factorial(l + mp) * factorial(l - mp))
            val = (sp.Rational(S.numerator, S.denominator)
                   * sp.sqrt(sp.Integer(F)) / sp.Integer(2) ** l)
            d[m + l, mp + l] = float(val.evalf(50))
    return d


def compute_M_exact(l):
    """Compute M_l = D^l(pi/2, pi/2, pi/2) exactly (no cache).

    (M_l)_{m m'} = (-i)^{m+m'} d^l_{m m'}(pi/2).
    """
    d = _wigner_d_pi2_exact(l)
    m = np.arange(-l, l + 1)
    phase = (-1j) ** (m[:, None] + m[None, :])
    return phase * d


@lru_cache(maxsize=None)
def wigner_M(l):
    """The fixed rotation matrix M_l = D^l(pi/2, pi/2, pi/2), complex128.

    Loaded from the packaged cache when available; computed exactly (and
    stored in the user cache) otherwise.
    """
    fname = f"wigner_M_l{l}.npz"
    for directory in (_PKG_DATA_DIR, user_cache_dir()):
        path = os.path.join(directory, fname)
        if os.path.exists(path):
            return np.load(path)["M"]
    M = compute_M_exact(l)
    cache = user_cache_dir()
    os.makedirs(cache, exist_ok=True)
    np.savez_compressed(os.path.join(cache, fname), M=M)
    return M


def rotation_funcs(l, omega=1.0):
    """Return callables (B_func, C_func) for degree l.

    B_func(t) and C_func(beta) return (2l+1, 2l+1) complex matrices,
    matching the classic symbolic-Wigner interface but built from the
    numeric M_l by pure phase modulation.
    """
    M = wigner_M(l)
    m = np.arange(-l, l + 1)

    def B_func(t):
        return np.exp(-1j * m * omega * t)[:, None] * M

    def C_func(beta):
        return np.exp(1j * m * beta)[:, None] * M

    return B_func, C_func


def precompute_rotations(l_max, omega=1.0):
    """Precompute (B_func, C_func) for all degrees up to l_max."""
    return {l: rotation_funcs(l, omega) for l in range(l_max + 1)}
