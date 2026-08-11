"""Rendering tests: the fast grid-rotation renderer must agree with the
Wigner coefficient-rotation reference at every spin phase and inclination
(regression test for a spurious-precession bug in the fast path)."""

import numpy as np
import pytest

import jittermap as jm
from jittermap.plotting.render import (build_projection_grid, render_surface,
                                       render_surface_fast)

L = 8


@pytest.fixture(scope="module")
def scene():
    s = jm.multispot_surface([(35, 300, 14.0), (-20, 100, 9.0)], L,
                             texture_amplitude=0.003, texture_seed=5)
    grid = build_projection_grid(n_grid=80)
    return s, grid


@pytest.mark.parametrize("t", [0.0, 0.9, 2.2, 4.5])
@pytest.mark.parametrize("inc", [0.0, 0.6, np.pi / 2])
def test_fast_renderer_matches_wigner(scene, t, inc):
    s, (X, Y, Z, THETA, PHI, mask) = scene
    m_ref = render_surface(s, L, inc, THETA, PHI, mask, t=t)
    m_fast = render_surface_fast(s, L, inc, THETA, PHI, mask, X, Y, Z, t=t)
    assert np.nanmax(np.abs(m_ref - m_fast)) < 1e-10


def test_fast_renderer_omega(scene):
    """omega scaling: fast and reference must agree for omega != 1."""
    s, (X, Y, Z, THETA, PHI, mask) = scene
    from jittermap.forward.wigner import precompute_rotations
    omega, t = 2.5, 0.7
    rot = precompute_rotations(L, omega)
    m_ref = render_surface(s, L, 0.6, THETA, PHI, mask, t=t, omega=omega,
                           rotations=rot)
    m_fast = render_surface_fast(s, L, 0.6, THETA, PHI, mask, X, Y, Z,
                                 t=t, omega=omega)
    assert np.nanmax(np.abs(m_ref - m_fast)) < 1e-10


def test_spot_appears_where_placed(scene):
    """A spot placed at (lat, lon=0) faces the observer at t=0 for an
    equator-on star: the rendered minimum must sit at that latitude on
    the central meridian."""
    s = jm.generate_spot(lat_deg=30, lon_deg=0, radius_deg=12, l_max=L)
    X, Y, Z, THETA, PHI, mask = build_projection_grid(n_grid=201)
    m = render_surface_fast(s, L, 0.0, THETA, PHI, mask, X, Y, Z, t=0.0)
    m_filled = np.where(np.isnan(m), np.inf, m)
    i, j = np.unravel_index(np.argmin(m_filled), m.shape)
    y_img = np.linspace(-1, 1, 201)[j]   # horizontal image coord
    z_img = np.linspace(-1, 1, 201)[i]   # vertical image coord
    assert abs(z_img - np.sin(np.deg2rad(30))) < 0.05
    assert abs(y_img) < 0.05
