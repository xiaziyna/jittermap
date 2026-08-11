"""Spot generator and surface composition tests."""

import numpy as np
import pytest
from jittermap.harmonics._compat import sph_harm

from jittermap.harmonics.spots import generate_spot, spot_area_fraction
from jittermap.harmonics.indexing import SHIndexer, project_real_surface
from jittermap.harmonics.surfaces import (multispot_surface, random_surface,
                                          rfrac_to_deg, deg_to_rfrac)


def evaluate_surface(coeffs, l_max, theta, phi):
    val = np.zeros_like(theta, dtype=complex)
    sh = SHIndexer(l_max)
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            val += coeffs[sh.get_index(l, m)] * sph_harm(m, l, phi, theta)
    return val.real


def test_area_fraction():
    assert abs(spot_area_fraction(90.0) - 0.5) < 1e-12
    assert abs(spot_area_fraction(180.0) - 1.0) < 1e-12


def test_conjugate_symmetry():
    l_max = 8
    c = generate_spot(35.0, 120.0, 10.0, l_max, sigma_taper=False)
    assert np.allclose(c, project_real_surface(c, l_max), atol=1e-13)


def test_spot_depth_at_center():
    """The reconstructed surface value at the spot center approaches
    background - contrast for a well-resolved spot."""
    l_max = 25
    lat, lon, radius = 20.0, 40.0, 25.0
    c = generate_spot(lat, lon, radius, l_max, contrast=0.8,
                      include_background=True, sigma_taper=True)
    theta_c = np.deg2rad(90.0 - lat)
    phi_c = np.deg2rad(lon)
    val = evaluate_surface(c, l_max, np.array([theta_c]), np.array([phi_c]))[0]
    background = 1.0
    assert abs(val - (background - 0.8)) < 0.05


def test_spot_outside_untouched():
    """Far from the spot the surface stays at the background level."""
    l_max = 25
    c = generate_spot(40.0, 100.0, 8.0, l_max, include_background=True,
                      sigma_taper=True)
    theta = np.deg2rad(90.0 + 40.0)  # antipodal latitude
    phi = np.deg2rad(280.0)
    val = evaluate_surface(c, l_max, np.array([theta]), np.array([phi]))[0]
    assert abs(val - 1.0) < 0.05


def test_rotation_invariance_of_degree_power():
    """Moving the spot must not change the per-degree power (rotations
    act unitarily within each degree)."""
    l_max = 10
    sh = SHIndexer(l_max)
    c1 = generate_spot(0.0, 0.0, 12.0, l_max, sigma_taper=False,
                       include_background=False)
    c2 = generate_spot(55.0, 213.0, 12.0, l_max, sigma_taper=False,
                       include_background=False)
    for l in range(l_max + 1):
        p1 = np.sum(np.abs(c1[sh.get_l_indices(l)]) ** 2)
        p2 = np.sum(np.abs(c2[sh.get_l_indices(l)]) ** 2)
        assert abs(p1 - p2) < 1e-12 * max(1.0, p1)


def test_multispot_additivity():
    l_max = 8
    single_a = generate_spot(30, 100, 10, l_max, include_background=False,
                             sigma_taper=False)
    single_b = generate_spot(-10, 250, 6, l_max, include_background=False,
                             sigma_taper=False)
    both = multispot_surface([(30, 100, 10), (-10, 250, 6)], l_max,
                             drop_monopole=False)
    assert np.allclose(both, single_a + single_b, atol=1e-13)


def test_random_surface_symmetry_and_seed():
    s1 = random_surface(8, seed=42)
    s2 = random_surface(8, seed=42)
    assert np.allclose(s1, s2)
    assert np.allclose(s1, project_real_surface(s1, 8), atol=1e-13)


def test_rfrac_roundtrip():
    for r in [0.05, 0.1, 0.25]:
        assert abs(deg_to_rfrac(rfrac_to_deg(r)) - r) < 1e-12
