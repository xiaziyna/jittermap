"""Forward-model tests: fast paths vs reference, realness of signals,
and agreement with the historical research implementation semantics."""

import numpy as np
import pytest

from jittermap.forward.design import (design_matrix_reference,
                                      precompute_vandermonde,
                                      design_matrix_vandermonde,
                                      ForwardModel)
from jittermap.forward.kernels import compute_A_lm, compute_A_lm_photo
from jittermap.harmonics.surfaces import multispot_surface, random_surface

L = 6
N = 40
TIMES = np.linspace(0, 2 * np.pi, N, endpoint=False)
INC = 0.6


@pytest.fixture(scope="module")
def kernels():
    A_x, A_y = compute_A_lm(L)
    return {"x": A_x, "y": A_y, "p": compute_A_lm_photo(L)}


@pytest.fixture(scope="module")
def precomp():
    return precompute_vandermonde(TIMES, L)


@pytest.mark.parametrize("channel", ["x", "y", "p"])
def test_vandermonde_matches_reference(channel, kernels, precomp):
    D_ref = design_matrix_reference(TIMES, L, INC, kernels[channel])
    D_fast = design_matrix_vandermonde(L, INC, kernels[channel], *precomp)
    assert np.allclose(D_ref, D_fast, atol=1e-12)


@pytest.mark.parametrize("channel", ["x", "y", "p"])
def test_real_surface_gives_real_signal(channel, kernels, precomp):
    """For coefficients with real-surface conjugate symmetry the signal
    must be real up to numerical precision."""
    s = multispot_surface([(30, 120, 12.0), (-20, 250, 8.0)], L,
                          texture_amplitude=0.01, texture_seed=1)
    D = design_matrix_vandermonde(L, INC, kernels[channel], *precomp)
    y = D.dot(s)
    assert np.max(np.abs(y.imag)) < 1e-10 * max(1.0, np.max(np.abs(y.real)))


def test_forward_model_class(kernels):
    fm = ForwardModel(TIMES, L)
    s = random_surface(L, seed=0)
    y_dict = fm.observe(s, INC, channels="xyp", stacked=False)
    y_stacked = fm.observe(s, INC, channels="xyp", stacked=True)
    assert set(y_dict) == {"x", "y", "p"}
    assert np.allclose(y_stacked,
                       np.concatenate([y_dict["x"], y_dict["y"], y_dict["p"]]))
    D = fm.design(INC, channels="xy", stacked=True)
    assert D.shape == (2 * N, (L + 1) ** 2)


def test_pole_on_astrometry_static(kernels):
    """Pole-on (beta = pi/2 in this convention is pole-on? equator-on at 0):
    at inclination such that the spin axis points at the observer, rotation
    cannot modulate the photometric signal: p(t) is constant."""
    fm = ForwardModel(TIMES, L)
    s = multispot_surface([(45, 90, 15.0)], L)
    y = fm.observe(s, inclination=np.pi / 2, channels="p", stacked=True)
    assert np.std(y) < 1e-10 * max(1.0, np.abs(np.mean(y)))
