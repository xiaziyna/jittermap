"""Kernel tests: the three independent computation paths must agree,
and the selection rules must hold."""

import numpy as np
import pytest

from jittermap.forward.kernels import (compute_A_lm, compute_A_lm_sympy,
                                       compute_A_lm_photo,
                                       numerical_A_lm_photo,
                                       compute_k_x, compute_k_y,
                                       compute_k_x_sympy, compute_k_y_sympy)


LP = 6


def test_numeric_vs_sympy_astrometric():
    A_x, A_y = compute_A_lm(LP)
    A_x_s, A_y_s = compute_A_lm_sympy(LP)
    assert np.allclose(A_x, A_x_s, atol=1e-10)
    assert np.allclose(A_y, A_y_s, atol=1e-10)


@pytest.mark.parametrize("l,m", [(1, 0), (1, 1), (2, 1), (3, 2), (5, 3)])
def test_elementwise_quad_vs_sympy(l, m):
    assert abs(compute_k_x(l, m) - complex(compute_k_x_sympy(l, m)).real) < 1e-10
    assert abs(compute_k_y(l, m) - complex(compute_k_y_sympy(l, m)).real) < 1e-10


def test_selection_rule_even_l():
    """Astrometric kernels vanish for even l > 2."""
    A_x, A_y = compute_A_lm(LP)
    for ln in range(3, LP + 1):
        if ln % 2 == 0:
            assert np.allclose(A_x[ln], 0)
            assert np.allclose(A_y[ln], 0)


def test_photo_selection_rule_odd_l():
    """Photometric kernel vanishes for odd l > 2."""
    A_p = compute_A_lm_photo(LP)
    for ln in range(3, LP + 1):
        if ln % 2 == 1:
            assert np.allclose(A_p[ln], 0)


def test_photo_vs_grid_sum():
    """Analytic photometric kernel agrees with brute-force disk summation."""
    A_p = compute_A_lm_photo(4)
    A_p_grid = numerical_A_lm_photo(4, n_grid=800)
    # the disk-edge discretization error of the grid sum is O(1/n_grid)
    assert np.allclose(A_p, A_p_grid, atol=5e-3)


def test_monopole_photo_value():
    """k^p_{0,0}: radial integral of sqrt(1-x^2) is pi/2, azimuthal
    integral of cos(phi) over [-pi/2, pi/2] is 2, normalization N_00."""
    A_p = compute_A_lm_photo(2)
    expected = np.sqrt(1 / (4 * np.pi)) * (np.pi / 2) * 2
    assert abs(A_p[0, 2] - expected) < 1e-10
