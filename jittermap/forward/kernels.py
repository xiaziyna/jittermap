"""Astrometric and photometric moment kernels.

The astrometric kernels k^x_{l,m}, k^y_{l,m} are the first moments of the
visible-hemisphere projection of each complex spherical harmonic (the
photocenter response), and the photometric kernel k^p_{l,m} is the
corresponding zeroth moment (the disk-integrated flux response).

Tables are assembled per maximum degree lp as arrays A_lm of shape
(lp+1, 2*lp+1), row ln holding the kernels for degree ln at columns
lp+m (m = -ln..ln). Selection rule: kernels vanish for even l > 2.

Three independent computation paths are provided, used to cross-validate
one another in the test suite: SciPy quadrature (compute_A_lm /
compute_A_lm_photo, the default), SymPy symbolic radial integrals with
closed-form azimuthal integrals, and brute-force numeric summation over
a projected-disk grid.

Results are cached in memory and on disk (packaged tables first, then a
writable user cache).
"""

import os
from functools import lru_cache

import numpy as np
from scipy import integrate
from scipy.special import lpmv, factorial
from jittermap.harmonics._compat import sph_harm
import sympy as sp

_PKG_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "kernels")


def user_cache_dir():
    base = os.environ.get("JITTERMAP_CACHE_DIR")
    if base is None:
        base = os.path.join(os.path.expanduser("~"), ".cache", "jittermap")
    return os.path.join(base, "kernels")


def _cache_load(fname, keys):
    for directory in (_PKG_DATA_DIR, user_cache_dir()):
        path = os.path.join(directory, fname)
        if os.path.exists(path):
            data = np.load(path)
            return tuple(data[k] for k in keys)
    return None


def _cache_store(fname, **arrays):
    cache = user_cache_dir()
    os.makedirs(cache, exist_ok=True)
    np.savez_compressed(os.path.join(cache, fname), **arrays)


# ---------------------------------------------------------------------------
# Closed-form azimuthal integrals
# ---------------------------------------------------------------------------

def I_phi_y_closed(m, prec=80):
    """Closed-form azimuthal integral for k_y:
    I_phi_y(m) = int_{-pi/2}^{pi/2} cos(phi) cos(m phi) dphi
               = -2 cos(pi m / 2) / (m^2 - 1)   for m != +-1
                 pi/2                            for m = +-1
    """
    m_sym = sp.Integer(m)
    if m_sym == 1 or m_sym == -1:
        return float((sp.pi / 2).evalf(prec))
    expr = -2 * sp.cos(sp.pi * m_sym / 2) / (m_sym ** 2 - 1)
    return float(expr.evalf(prec))


def I_phi_x_closed(m, prec=80):
    """Closed-form azimuthal integral for k_x:
    I_phi_x(m) = int_{-pi/2}^{pi/2} sin(phi) cos(phi) sin(m phi) dphi
               = -2 sin(pi m / 2) / (m^2 - 4)   for m != +-2
                 +-pi/4                          for m = +-2
    """
    m_sym = sp.Integer(m)
    if m_sym == 2:
        return float((sp.pi / 4).evalf(prec))
    if m_sym == -2:
        return float((-sp.pi / 4).evalf(prec))
    expr = -2 * sp.sin(sp.pi * m_sym / 2) / (m_sym ** 2 - 4)
    return float(expr.evalf(prec))


# ---------------------------------------------------------------------------
# Elementwise kernels: SciPy quadrature
# ---------------------------------------------------------------------------

def _norm_lm(l, m):
    return np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - m) / factorial(l + m))


def compute_k_y(l, m):
    """k^y_{l,m} via SciPy quadrature."""
    radial, _ = integrate.quad(lambda x: lpmv(m, l, x) * x * (1 - x ** 2) ** 0.5,
                               -1, 1, epsabs=1e-13)
    azimuthal, _ = integrate.quad(lambda x: np.cos(x) * np.cos(m * x),
                                  -np.pi / 2, np.pi / 2, epsabs=1e-13)
    return radial * azimuthal * _norm_lm(l, m)


def compute_k_x(l, m):
    """k^x_{l,m} via SciPy quadrature (without the i-phase applied by
    compute_A_lm)."""
    radial, _ = integrate.quad(lambda x: lpmv(m, l, x) * (1 - x ** 2),
                               -1, 1, epsabs=1e-13)
    azimuthal, _ = integrate.quad(lambda x: np.sin(x) * np.cos(x) * np.sin(m * x),
                                  -np.pi / 2, np.pi / 2, epsabs=1e-13)
    return radial * azimuthal * _norm_lm(l, m)


# ---------------------------------------------------------------------------
# Elementwise kernels: SymPy cross-check path
# ---------------------------------------------------------------------------

def _norm_lm_sympy(l, m):
    return sp.sqrt((2 * l + 1) / (4 * sp.pi)
                   * sp.factorial(l - m) / sp.factorial(l + m))


def compute_k_y_sympy(l, m, prec=50):
    """SymPy version of compute_k_y: symbolic radial integral and
    closed-form azimuthal integral."""
    x = sp.symbols("x", real=True)
    P_lm = sp.assoc_legendre(l, m, x)
    I_radial = sp.integrate(P_lm * x * sp.sqrt(1 - x ** 2), (x, -1, 1))
    val = (I_radial * I_phi_y_closed(m, prec=prec) * _norm_lm_sympy(l, m)).evalf(prec)
    return complex(val)


def compute_k_x_sympy(l, m, prec=50):
    """SymPy version of compute_k_x."""
    x = sp.symbols("x", real=True)
    P_lm = sp.assoc_legendre(l, m, x)
    I_radial = sp.integrate(P_lm * (1 - x ** 2), (x, -1, 1))
    val = (I_radial * I_phi_x_closed(m, prec=prec) * _norm_lm_sympy(l, m)).evalf(prec)
    return complex(val)


def compute_k_photo(l, m, prec=50):
    """Photometric kernel k^p_{l,m}:
        F_l^m proportional to N_l^m * int_{-1}^1 sqrt(1-x^2) P_l^m(x) dx
                             * int_{-pi/2}^{pi/2} cos(phi) cos(m phi) dphi.
    Zero for odd l > 2 (selection rule).
    """
    if l > 2 and (l % 2 == 1):
        return 0.0 + 0.0j
    x = sp.symbols("x", real=True)
    P_lm = sp.assoc_legendre(l, m, x)
    I_radial = sp.integrate(P_lm * sp.sqrt(1 - x ** 2), (x, -1, 1))
    val = (_norm_lm_sympy(l, m) * I_radial * I_phi_y_closed(m, prec=prec)).evalf(prec)
    return complex(val)


# ---------------------------------------------------------------------------
# Assembled kernel tables (cached)
# ---------------------------------------------------------------------------

def _selection_rows(lp):
    """Degrees with non-vanishing astrometric kernels: l <= 2 or odd l."""
    return [ln for ln in range(lp + 1) if ln <= 2 or ln % 2 == 1]


def _compute_A_lm_nocache(lp):
    lp = int(lp)
    A_lm_y = np.zeros((lp + 1, 2 * lp + 1), dtype=complex)
    A_lm_x = np.zeros((lp + 1, 2 * lp + 1), dtype=complex)
    for ln in _selection_rows(lp):
        for mn in range(-ln, ln + 1):
            kx = compute_k_x(ln, mn)
            ky = compute_k_y(ln, mn)
            A_lm_x[ln, mn + lp] = 0 if np.isnan(kx) else kx
            A_lm_y[ln, mn + lp] = 0 if np.isnan(ky) else ky
    A_lm_x *= 1j  # x-channel phase in the complex e^{i m phi} convention
    return A_lm_x, A_lm_y


@lru_cache(maxsize=None)
def _compute_A_lm_cached(lp):
    lp = int(lp)
    fname = f"A_lm_numeric_lp{lp}.npz"
    cached = _cache_load(fname, ("A_lm_x", "A_lm_y"))
    if cached is not None:
        return cached
    A_lm_x, A_lm_y = _compute_A_lm_nocache(lp)
    _cache_store(fname, A_lm_x=A_lm_x, A_lm_y=A_lm_y)
    return A_lm_x, A_lm_y


def compute_A_lm(lp):
    """Astrometric kernel tables (A_lm_x, A_lm_y) for max degree lp,
    each of shape (lp+1, 2*lp+1). Cached in memory and on disk."""
    A_lm_x, A_lm_y = _compute_A_lm_cached(int(lp))
    return A_lm_x.copy(), A_lm_y.copy()


def _compute_A_lm_sympy_nocache(lp):
    A_lm_y = np.zeros((lp + 1, 2 * lp + 1), dtype=complex)
    A_lm_x = np.zeros((lp + 1, 2 * lp + 1), dtype=complex)
    for ln in _selection_rows(lp):
        for mn in range(-ln, ln + 1):
            A_lm_x[ln, mn + lp] = compute_k_x_sympy(ln, mn)
            A_lm_y[ln, mn + lp] = compute_k_y_sympy(ln, mn)
    A_lm_x *= 1j
    return A_lm_x, A_lm_y


@lru_cache(maxsize=None)
def _compute_A_lm_sympy_cached(lp):
    lp = int(lp)
    fname = f"A_lm_sympy_lp{lp}.npz"
    cached = _cache_load(fname, ("A_lm_x", "A_lm_y"))
    if cached is not None:
        return cached
    A_lm_x, A_lm_y = _compute_A_lm_sympy_nocache(lp)
    _cache_store(fname, A_lm_x=A_lm_x, A_lm_y=A_lm_y)
    return A_lm_x, A_lm_y


def compute_A_lm_sympy(lp):
    """SymPy cross-check version of compute_A_lm."""
    A_lm_x, A_lm_y = _compute_A_lm_sympy_cached(int(lp))
    return A_lm_x.copy(), A_lm_y.copy()


def _compute_A_lm_photo_nocache(lp):
    A_lm_photo = np.zeros((lp + 1, 2 * lp + 1), dtype=complex)
    for ln in range(lp + 1):
        for mn in range(-ln, ln + 1):
            A_lm_photo[ln, mn + lp] = compute_k_photo(ln, mn)
    return A_lm_photo


@lru_cache(maxsize=None)
def _compute_A_lm_photo_cached(lp):
    lp = int(lp)
    fname = f"A_lm_photo_lp{lp}.npz"
    cached = _cache_load(fname, ("A_lm_photo",))
    if cached is not None:
        return cached[0]
    A_lm_photo = _compute_A_lm_photo_nocache(lp)
    _cache_store(fname, A_lm_photo=A_lm_photo)
    return A_lm_photo


def compute_A_lm_photo(lp):
    """Photometric kernel table A_lm_photo for max degree lp,
    shape (lp+1, 2*lp+1). Cached in memory and on disk."""
    return _compute_A_lm_photo_cached(int(lp)).copy()


# ---------------------------------------------------------------------------
# Brute-force grid cross-check
# ---------------------------------------------------------------------------

def numerical_A_lm_photo(lp, n_grid=1000):
    """Photometric kernel by direct summation over the visible disk;
    slow, used as an independent cross-check of compute_A_lm_photo."""
    y = np.linspace(-1, 1, n_grid)
    z = np.linspace(-1, 1, n_grid)
    Y, Z = np.meshgrid(y, z)
    R = np.hypot(Y, Z)
    mask = R > 1
    X = np.sqrt(np.clip(1 - Y ** 2 - Z ** 2, 0, None))
    X[mask] = np.nan
    Y[mask] = np.nan
    Z[mask] = np.nan
    THETA = np.arccos(np.clip(Z, -1, 1))
    PHI = np.arctan2(Y, X)
    A_lm_photo = np.zeros((lp + 1, 2 * lp + 1), dtype=complex)
    for ln in range(lp + 1):
        for mn in range(-ln, ln + 1):
            A_lm_photo[ln, mn + lp] = np.nansum(sph_harm(mn, ln, PHI, THETA))
    A_lm_photo /= (n_grid / 2) ** 2
    return A_lm_photo
