"""Limb-darkening tests: the laws, the per-power kernel tables against
SymPy with the law kept unexpanded, and an end-to-end check that the
forward model reproduces the flux and first moments of a limb-darkened
disk integrated exactly on the sky plane."""

import numpy as np
import pytest
import sympy as sp

from jittermap.forward.design import ForwardModel
from jittermap.forward.kernels import (compute_A_lm, compute_A_lm_photo,
                                       compute_k_photo, compute_k_photo_quad,
                                       numerical_A_lm_photo)
from jittermap.harmonics.surfaces import multispot_surface
from jittermap.plotting.render import render_surface_fast
from jittermap.star_physics.limb_darkening import LimbDarkening

L = 6
N = 6
TIMES = np.linspace(0, 2 * np.pi, N, endpoint=False)
INC = 0.6
A, B = 0.4, 0.26
QUAD = LimbDarkening.quadratic(A, B)
NONLINEAR = LimbDarkening.nonlinear(0.5, -0.1, 0.3, -0.05)


def test_law_values():
    mu = 0.3
    assert QUAD(1.0) == pytest.approx(1.0)
    assert QUAD(0.0) == pytest.approx(1 - A - B)
    assert QUAD(mu) == pytest.approx(1 - A * (1 - mu) - B * (1 - mu) ** 2)
    assert LimbDarkening.linear(0.6)(mu) == pytest.approx(1 - 0.6 * (1 - mu))
    c = (0.5, -0.1, 0.3, -0.05)
    nl = LimbDarkening.nonlinear(*c)
    assert nl(1.0) == pytest.approx(1.0)
    assert nl(mu) == pytest.approx(
        1 - sum(cj * (1 - mu ** (j / 2)) for j, cj in enumerate(c, start=1)))


def test_disk_flux():
    assert LimbDarkening.quadratic(0.0, 0.0).disk_flux() == pytest.approx(np.pi)
    assert QUAD.disk_flux() == pytest.approx(np.pi * (1 - A / 3 - B / 6))


def test_uniform_law_matches_uniform_tables():
    """A law with zero darkening must give the shipped kernel tables."""
    tables = LimbDarkening.quadratic(0.0, 0.0).kernel_tables(L)
    A_x, A_y = compute_A_lm(L)
    assert np.allclose(tables["x"], A_x, atol=1e-12)
    assert np.allclose(tables["y"], A_y, atol=1e-12)
    assert np.allclose(tables["p"], compute_A_lm_photo(L), atol=1e-12)


@pytest.mark.parametrize("l,m", [(0, 0), (1, 1), (2, 0), (2, 2), (4, 3)])
def test_photo_quad_vs_sympy_at_power_zero(l, m):
    """The quadrature photometric path used for the darkening powers
    agrees with the exact SymPy path on the uniform disk."""
    assert abs(compute_k_photo_quad(l, m, 0) - compute_k_photo(l, m)) < 1e-10


def _sympy_darkened_kernel(l, m, channel, a, b):
    """Hemisphere integral of w(mu) * mu * (moment weight) * Y_lm with the
    quadratic law w kept as a function of mu = sin(theta) cos(phi); no
    separation into powers is assumed. Integrated in x = cos(theta)."""
    x, ph = sp.symbols("x phi", real=True)
    s = sp.sqrt(1 - x ** 2)
    mu = s * sp.cos(ph)
    w = 1 - a * (1 - mu) - b * (1 - mu) ** 2
    weight = {"p": 1, "y": x, "x": s * sp.sin(ph)}[channel]
    norm = sp.sqrt((2 * l + 1) / (4 * sp.pi) * sp.factorial(l - m) / sp.factorial(l + m))
    Y = norm * sp.assoc_legendre(l, m, x) * sp.exp(sp.I * m * ph)
    I_phi = sp.integrate(sp.expand(w * mu * weight * Y), (ph, -sp.pi / 2, sp.pi / 2))
    return complex(sp.N(sp.integrate(sp.expand(I_phi), (x, -1, 1)), 20))


@pytest.mark.parametrize("l,m,channel", [(0, 0, "p"), (1, 1, "x"), (1, 0, "y"),
                                         (2, 2, "p"), (3, 2, "y")])
def test_darkened_kernel_vs_sympy(l, m, channel):
    """The coefficient-weighted sum of per-power tables equals the direct
    symbolic integral of the darkened integrand."""
    a, b = sp.Rational(2, 5), sp.Rational(13, 50)
    table = LimbDarkening.quadratic(float(a), float(b)).kernel_tables(l)[channel]
    assert abs(_sympy_darkened_kernel(l, m, channel, a, b) - table[l, m + l]) < 1e-10


def test_photo_power_vs_grid_sum():
    """Weighted photometric kernel agrees with brute-force disk summation."""
    A_p = compute_A_lm_photo(4, power=1)
    A_p_grid = numerical_A_lm_photo(4, n_grid=800, power=1)
    assert np.allclose(A_p, A_p_grid, atol=5e-3)


def test_selection_rules_only_for_uniform_disk():
    """A darkened disk has non-zero kernels at the degrees the uniform
    disk is blind to."""
    A_p = compute_A_lm_photo(L, power=1)
    assert not np.allclose(A_p[3], 0)
    A_x, A_y = compute_A_lm(L, power=1)
    assert not (np.allclose(A_x[4], 0) and np.allclose(A_y[4], 0))


def test_uniform_disk_flux_under_limb_darkening():
    fm = ForwardModel(TIMES, L, limb_darkening=QUAD)
    s = np.zeros((L + 1) ** 2, dtype=complex)
    s[0] = 2 * np.sqrt(np.pi)
    signals = fm.observe(s, INC, channels="xyp", stacked=False)
    np.testing.assert_allclose(signals["p"], QUAD.disk_flux(), atol=1e-12)
    np.testing.assert_allclose(signals["x"], 0, atol=1e-12)
    np.testing.assert_allclose(signals["y"], 0, atol=1e-12)


def _disk_quadrature(n_mu, n_psi):
    """Sky-plane quadrature grid in the layout of build_projection_grid,
    plus weights: Gauss-Legendre in v = sqrt(mu), uniform in azimuth, with
    dA = mu dmu dpsi = 2 v^3 dv dpsi. A band-limited surface times a
    law with integer or half-integer powers of mu is a polynomial in v
    after the azimuthal sum, so the rule is exact."""
    v, wv = np.polynomial.legendre.leggauss(n_mu)
    v, wv = 0.5 * (v + 1), 0.5 * wv
    psi = np.arange(n_psi) * 2 * np.pi / n_psi
    V, PSI = np.meshgrid(v, psi)
    WV = np.meshgrid(wv, psi)[0]
    MU = V ** 2
    R = np.sqrt(1 - MU ** 2)
    X, Y, Z = MU, R * np.cos(PSI), R * np.sin(PSI)
    THETA = np.arccos(np.clip(Z, -1, 1))
    PHI = np.arctan2(Y, X)
    W = 2 * V ** 3 * WV * (2 * np.pi / n_psi)
    return X, Y, Z, THETA, PHI, np.zeros_like(X, dtype=bool), W


@pytest.mark.parametrize("law", [None, QUAD, NONLINEAR],
                         ids=["uniform", "quadratic", "nonlinear"])
def test_forward_model_reproduces_limb_darkened_disk(law):
    """Render the spotted surface on the sky plane at each time, darken it
    by w(mu) with mu the depth coordinate, and integrate the flux and the
    two first moments with an exact quadrature. The forward model must
    reproduce all three channels to the kernel quadrature tolerance; the
    limb-darkening effect itself is ~3e-2, so the test also separates the
    darkened model from the uniform-disk one."""
    s = multispot_surface([(30, 40, 20.0), (-20, 200, 15.0)], L,
                          drop_monopole=False)
    model = ForwardModel(TIMES, L, limb_darkening=law).observe(
        s, INC, channels="xyp", stacked=False)

    X, Y, Z, THETA, PHI, mask, W = _disk_quadrature(n_mu=24, n_psi=48)
    w = 1.0 if law is None else law(X)
    for n, t in enumerate(TIMES):
        img = W * w * render_surface_fast(s, L, INC, THETA, PHI, mask, X, Y, Z, t=t)
        assert abs(img.sum() - model["p"][n]) < 1e-10
        assert abs((Y * img).sum() - model["x"][n]) < 1e-10
        assert abs((Z * img).sum() - model["y"][n]) < 1e-10

    if law is not None:
        uniform = ForwardModel(TIMES, L).observe(s, INC, channels="xyp", stacked=False)
        for c in "xyp":
            assert np.max(np.abs(model[c] - uniform[c])) > 1e-2
