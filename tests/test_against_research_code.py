"""Equivalence against the original research implementation.

These tests only run when the research repository is present (set
JITTERMAP_RESEARCH_DIR or rely on the default sibling location); they
guarantee the library reproduces the exact numbers behind the paper.
"""

import os
import sys

import numpy as np
import pytest

RESEARCH_DIR = os.environ.get(
    "JITTERMAP_RESEARCH_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "astrometry"))

research_available = os.path.isdir(RESEARCH_DIR)
pytestmark = pytest.mark.skipif(not research_available,
                                reason="research repo not available")

if research_available:
    sys.path.insert(0, RESEARCH_DIR)
    # The research code imports the legacy scipy.special.sph_harm, removed
    # in scipy >= 1.15; inject the compatible shim before importing it.
    import scipy.special
    if not hasattr(scipy.special, "sph_harm"):
        from jittermap.harmonics._compat import sph_harm as _sph_harm_shim
        scipy.special.sph_harm = _sph_harm_shim


@pytest.mark.parametrize("l,inc", [(4, 0.3), (8, 1.1)])
def test_design_matrix_bit_equivalence(l, inc):
    from design_utils import (precompute_B_vandermonde as pv_old,
                              astrom_inc_rot_all_vfast as vfast_old)
    from integral_moments import compute_A_lm as compute_A_lm_old

    from jittermap.forward.design import (precompute_vandermonde,
                                          design_matrix_vandermonde)
    from jittermap.forward.kernels import compute_A_lm

    times = np.linspace(0, 2 * np.pi, 50, endpoint=True)
    A_x_old, A_y_old = compute_A_lm_old(l)
    A_x_new, A_y_new = compute_A_lm(l)
    assert np.allclose(A_x_old, A_x_new, atol=1e-14)
    assert np.allclose(A_y_old, A_y_new, atol=1e-14)

    D_old = vfast_old(l, inc, A_x_old, *pv_old(times, l))
    D_new = design_matrix_vandermonde(l, inc, A_x_new,
                                      *precompute_vandermonde(times, l))
    assert np.allclose(D_old, D_new, atol=1e-12)


def test_spot_generator_equivalence():
    sys.path.insert(0, os.path.join(RESEARCH_DIR, "simulate_spots_jamila"))
    from spot_generator import generate_spot as generate_spot_old
    from jittermap.harmonics.spots import generate_spot

    # The research generator carries a spurious (-1)^m in d^l_{m0},
    # equivalent to a 180-degree longitude shift; the library fixes it,
    # so old(lat, lon) corresponds to new(lat, lon + 180).
    c_old = generate_spot_old(33.0, 121.0, 9.5, 12)
    c_new = generate_spot(33.0, 121.0 + 180.0, 9.5, 12)
    assert np.allclose(c_old, c_new, atol=1e-12)


def test_gmrf_prior_equivalence():
    from gmrf_util import gmrf_precision_diag as q_old
    from jittermap.inference.inversion import gmrf_precision_diag as q_new
    assert np.allclose(q_old(9, alpha=1.3, scale=2.5), q_new(9, alpha=1.3, scale=2.5))


def test_rotated_kernel_pyramid_equivalence():
    """The fast M_l C(beta) pyramid must match the research repo's
    sympy Wigner-composition version (figure_scripts/pyramid_rot_kernel)."""
    sys.path.insert(0, os.path.join(RESEARCH_DIR, "figure_scripts"))
    from pyramid_rot_kernel import compute_Bbeta_pyramid
    from jittermap.forward.kernels import compute_A_lm
    from jittermap.plotting.pyramids import rotated_kernel_pyramid

    l_max, inc = 3, 0.6
    A_x, _ = compute_A_lm(l_max)
    mine = np.nan_to_num(rotated_kernel_pyramid(l_max, inc, A_x))
    hers = np.nan_to_num(compute_Bbeta_pyramid(l_max, inc, A_x))
    assert np.allclose(mine, hers, atol=1e-12)
