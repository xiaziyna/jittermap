"""
Generate starspot surfaces directly in complex spherical harmonics.

A circular starspot (spherical cap) centered at the north pole has an
analytic SH expansion with only m=0 terms.  Rotating the cap to an
arbitrary (lat, lon) via the Wigner-D relation gives all (l, m) coefficients.
No external dependencies beyond numpy/scipy.
"""

import numpy as np
from scipy.special import eval_legendre
from jittermap.harmonics._compat import sph_harm


def _lanczos_sigma(l, l_max):
    """Lanczos sigma factor: sinc(l / (l_max + 1)) to suppress Gibbs ringing."""
    if l == 0:
        return 1.0
    x = np.pi * l / (l_max + 1)
    return np.sin(x) / x


def _cap_alm0(l_max, radius_rad, contrast=1.0, sigma_taper=True):
    """
    Axisymmetric (m=0) complex SH coefficients for a spherical cap of angular
    radius ``radius_rad`` centered at the north pole.

    The cap has value ``-contrast`` (dark spot) relative to a zero background.
    Add a monopole (l=0) offset separately if you want a uniform background of 1.

    Uses the analytic integral of Legendre polynomials over a cap:
        integral_{cos(theta0)}^{1} P_l(x) dx
            = (1 - cos(theta0))                             for l = 0
            = [P_{l-1}(cos(theta0)) - P_{l+1}(cos(theta0))] / (2l+1)   for l >= 1

    If ``sigma_taper`` is True, applies a Lanczos sigma factor to each degree
    to suppress Gibbs ringing from the sharp cap boundary.
    """
    cos_r = np.cos(radius_rad)
    a_l0 = np.zeros(l_max + 1)

    # l = 0
    a_l0[0] = -contrast * np.sqrt(np.pi) * (1.0 - cos_r)

    # l >= 1
    for l in range(1, l_max + 1):
        integral = (eval_legendre(l - 1, cos_r) - eval_legendre(l + 1, cos_r)) / (2 * l + 1)
        a_l0[l] = -contrast * np.sqrt(np.pi * (2 * l + 1)) * integral
        if sigma_taper:
            a_l0[l] *= _lanczos_sigma(l, l_max)

    return a_l0


def _wigner_d_m0(l, m, beta):
    """
    Wigner small-d matrix element d^l_{m,0}(beta).

    Uses the identity:
        d^l_{m,0}(beta) = sqrt(4*pi / (2l+1)) * Y_l^m(beta, 0)

    where Y_l^m(beta, 0) is real (phi=0). Note: no (-1)^m factor — with
    the Condon-Shortley phase already included in Y_l^m, an extra (-1)^m
    is equivalent to a 180-degree longitude shift of the rotated cap.
    """
    # legacy scipy convention: sph_harm(m, n, theta_azimuth, phi_polar)
    ylm = np.real(sph_harm(m, l, 0.0, beta))
    return np.sqrt(4.0 * np.pi / (2 * l + 1)) * ylm


def _rotate_axisymmetric(a_l0, lat_rad, lon_rad, l_max):
    """
    Rotate axisymmetric (m=0 only) coefficients to place the cap centre
    at geographic coordinates (lat_rad, lon_rad).

    The rotation uses Euler angles (alpha=lon, beta=colatitude, gamma=0):
        a'_{l,m} = e^{-i m lon} * d^l_{m,0}(colat) * a_{l,0}

    This automatically satisfies conjugate symmetry for a real surface.
    """
    colat = np.pi / 2.0 - lat_rad  # colatitude
    total = (l_max + 1) ** 2
    coeffs = np.zeros(total, dtype=complex)

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            idx = l * l + l + m  # standard (l,m) -> linear index
            d_val = _wigner_d_m0(l, m, colat)
            coeffs[idx] = np.exp(-1j * m * lon_rad) * d_val * a_l0[l]

    return coeffs


def generate_spot(lat_deg, lon_deg, radius_deg, l_max, contrast=1.0,
                  include_background=True, sigma_taper=True):
    """
    Generate complex SH coefficients for a single circular starspot.

    Parameters
    ----------
    lat_deg : float
        Spot latitude in degrees (-90 = south pole, +90 = north pole).
    lon_deg : float
        Spot longitude in degrees.
    radius_deg : float
        Angular radius of the spot in degrees.
    l_max : int
        Maximum spherical harmonic degree.
    contrast : float
        Spot contrast (1.0 = fully dark relative to background).
    include_background : bool
        If True, add a uniform background of 1 (monopole term).
    sigma_taper : bool
        If True, apply Lanczos sigma factor to suppress Gibbs ringing.

    Returns
    -------
    coeffs : ndarray, complex, shape ((l_max+1)^2,)
        Complex spherical harmonic coefficients.
    """
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    radius_rad = np.deg2rad(radius_deg)

    # Axisymmetric cap coefficients (north pole)
    a_l0 = _cap_alm0(l_max, radius_rad, contrast=contrast, sigma_taper=sigma_taper)

    # Rotate to (lat, lon)
    coeffs = _rotate_axisymmetric(a_l0, lat_rad, lon_rad, l_max)

    # Optionally add uniform background (Y_0^0 = 1/sqrt(4*pi), so
    # a_{0,0} = sqrt(4*pi) * 1 = 2*sqrt(pi) for a surface of value 1)
    if include_background:
        coeffs[0] += 2.0 * np.sqrt(np.pi)

    return coeffs


def spot_area_fraction(radius_deg):
    """Fraction of total sphere area covered by a cap of given angular radius."""
    return (1.0 - np.cos(np.deg2rad(radius_deg))) / 2.0
