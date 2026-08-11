"""Fourier compression tests: the compressed design must reproduce the
time-domain forward model exactly for uniform full-period sampling."""

import numpy as np
import pytest

from jittermap.forward.design import precompute_vandermonde, design_matrix_vandermonde
from jittermap.forward.fourier import (compress, fourier_design_matrix,
                                       fourier_to_real, real_to_fourier)
from jittermap.forward.kernels import compute_A_lm, compute_A_lm_photo
from jittermap.harmonics.surfaces import multispot_surface

L = 5
N = 32
TIMES = np.linspace(0, 2 * np.pi, N, endpoint=False)
INC = 0.7


@pytest.mark.parametrize("channel", ["x", "y", "p"])
def test_fourier_design_equals_compressed_time_design(channel):
    A_x, A_y = compute_A_lm(L)
    A = {"x": A_x, "y": A_y, "p": compute_A_lm_photo(L)}[channel]
    precomp = precompute_vandermonde(TIMES, L)
    D = design_matrix_vandermonde(L, INC, A, *precomp)
    F = fourier_design_matrix(L, INC, A)

    s = multispot_surface([(25, 140, 14.0)], L, texture_amplitude=0.01,
                          texture_seed=2)
    y = D.dot(s)
    f_from_time = compress(y, L)
    f_direct = F.dot(s)
    assert np.allclose(f_from_time, f_direct, atol=1e-12)


def test_real_packing_roundtrip():
    rng = np.random.default_rng(0)
    f = np.zeros(2 * L + 1, dtype=complex)
    f[L] = rng.normal()
    f[L + 1:] = rng.normal(size=L) + 1j * rng.normal(size=L)
    f[:L] = np.conj(f[L + 1:][::-1])
    assert np.allclose(real_to_fourier(fourier_to_real(f)), f)


def test_compress_requires_enough_samples():
    with pytest.raises(ValueError):
        compress(np.zeros(2 * L), L)


def test_conjugate_symmetry_of_real_signal():
    rng = np.random.default_rng(1)
    y = rng.normal(size=N)
    f = compress(y, L)
    assert np.allclose(f[:L], np.conj(f[L + 1:][::-1]), atol=1e-12)
