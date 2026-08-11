"""Composite surface generation: multi-spot maps and GMRF random textures.

All surfaces are returned as complex SH coefficient vectors in SHIndexer
ordering, satisfying the real-surface conjugate symmetry.
"""

import numpy as np

from jittermap.harmonics.indexing import SHIndexer
from jittermap.harmonics.spots import generate_spot


def random_surface(l_max, alpha=1.0, scale=2.0, magnitude=1.0, seed=None):
    """Sample real-valued surface SH coefficients from the GMRF prior.

    The marginal variance per degree is magnitude * (scale / l)^alpha for
    l >= 1 (and `magnitude` at l = 0). For m = 0 the full variance goes to
    the real part; for m != 0 it is split equally between real and
    imaginary parts, and conjugate symmetry is enforced so the surface
    is real.

    Parameters
    ----------
    l_max : int
        Maximum spherical harmonic degree.
    alpha : float
        Power-law exponent of the prior variance spectrum.
    scale : float
        Normalizing scale; variance at degree l is magnitude*(scale/l)^alpha.
    magnitude : float
        Overall variance scaling.
    seed : int, np.random.Generator, or None
        Seed or generator for reproducibility.

    Returns
    -------
    s : complex ndarray, shape ((l_max+1)^2,)
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    sh = SHIndexer(l_max=l_max)
    s = np.zeros(sh.total_coeffs, dtype=complex)
    for l in range(l_max + 1):
        var_l = magnitude if l == 0 else magnitude * (scale / l) ** alpha
        for m in range(l + 1):
            if m == 0:
                s[sh.get_index(l, 0)] = rng.normal(0, np.sqrt(var_l))
            else:
                std_part = np.sqrt(var_l / 2.0)
                coeff = rng.normal(0, std_part) + 1j * rng.normal(0, std_part)
                s[sh.get_index(l, m)] = coeff
                s[sh.get_index(l, -m)] = (-1) ** m * np.conj(coeff)
    return s


def multispot_surface(spots, l_max, contrast=1.0, sigma_taper=False,
                      texture_amplitude=0.0, texture_seed=None,
                      drop_monopole=True):
    """Generate a surface with one or more circular starspots, optionally
    adding a GMRF texture to emulate variation from smaller active regions.

    Parameters
    ----------
    spots : list of (lat_deg, lon_deg, radius_deg) tuples
        Spot positions and angular radii in degrees.
    l_max : int
        Maximum spherical harmonic degree.
    contrast : float
        Spot contrast (1.0 = fully dark relative to background).
    sigma_taper : bool
        If True, apply a Lanczos sigma factor to suppress Gibbs ringing
        from the sharp cap boundary.
    texture_amplitude : float
        Amplitude of the additive GMRF texture (0 disables it).
    texture_seed : int or None
        Seed for the GMRF texture.
    drop_monopole : bool
        If True, zero the l=0 coefficient (unobservable overall offset).

    Returns
    -------
    coeffs : complex ndarray, shape ((l_max+1)^2,)
    """
    sh = SHIndexer(l_max=l_max)
    coeffs = np.zeros(sh.total_coeffs, dtype=complex)
    for lat, lon, rad in spots:
        coeffs += generate_spot(lat, lon, rad, l_max, contrast=contrast,
                                include_background=False,
                                sigma_taper=sigma_taper)
    if texture_amplitude:
        coeffs = coeffs + texture_amplitude * random_surface(
            l_max, alpha=1.0, scale=2.0, seed=texture_seed)
    if drop_monopole:
        coeffs[0] = 0
    return coeffs


def rfrac_to_deg(r_frac):
    """Convert a spot radius expressed as a fraction of the stellar radius
    R_spot / R_star to the angular cap radius in degrees."""
    return float(np.rad2deg(2.0 * np.arcsin(r_frac)))


def deg_to_rfrac(radius_deg):
    """Inverse of rfrac_to_deg."""
    return float(np.sin(np.deg2rad(radius_deg) / 2.0))
