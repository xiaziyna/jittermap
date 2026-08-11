"""Wigner rotation tests.

The exact-formula M_l is validated against an independent symbolic
computation (sympy.physics.quantum.spin.Rotation), and the phase-diagonal
factorization identities against direct D-matrix evaluation.
"""

import numpy as np
import pytest
import sympy as sp
from sympy.physics.quantum.spin import Rotation

from jittermap.forward.wigner import (_wigner_d_pi2_exact, compute_M_exact,
                                      wigner_M, rotation_funcs)


@pytest.mark.parametrize("l", [0, 1, 2, 3, 5])
def test_d_pi2_matches_sympy(l):
    d_mine = _wigner_d_pi2_exact(l)
    for m in range(-l, l + 1):
        for mp in range(-l, l + 1):
            d_ref = float(Rotation.d(l, m, mp, sp.pi / 2).doit().evalf(30))
            assert abs(d_mine[m + l, mp + l] - d_ref) < 1e-14


@pytest.mark.parametrize("l", [1, 3])
def test_M_matches_symbolic_D(l):
    """M_l must equal D^l(pi/2, pi/2, pi/2) in the sympy convention."""
    M = compute_M_exact(l)
    for m in range(-l, l + 1):
        for mp in range(-l, l + 1):
            ref = complex(Rotation.D(l, m, mp, sp.pi / 2, sp.pi / 2,
                                     sp.pi / 2).doit().evalf(30))
            assert abs(M[m + l, mp + l] - ref) < 1e-14


@pytest.mark.parametrize("l", [1, 2, 4])
def test_phase_factorization(l):
    """B(t) and C(beta) from rotation_funcs must equal the direct
    symbolic D-matrix evaluation at the corresponding Euler angles."""
    B_func, C_func = rotation_funcs(l)
    t_val, inc_val = 0.7, 0.4
    for m in range(-l, l + 1):
        for mp in range(-l, l + 1):
            B_ref = complex(Rotation.D(l, m, mp, sp.Float(t_val) + sp.pi / 2,
                                       sp.pi / 2, sp.pi / 2).doit().evalf(30))
            C_ref = complex(Rotation.D(l, m, mp, -sp.Float(inc_val) + sp.pi / 2,
                                       sp.pi / 2, sp.pi / 2).doit().evalf(30))
            assert abs(B_func(t_val)[m + l, mp + l] - B_ref) < 1e-12
            assert abs(C_func(inc_val)[m + l, mp + l] - C_ref) < 1e-12


def test_M_unitary():
    for l in [1, 5, 12, 25, 40]:
        M = wigner_M(l)
        assert np.allclose(M @ M.conj().T, np.eye(2 * l + 1), atol=1e-12)


def test_cache_covers_shipped_range():
    for l in [0, 10, 30, 40]:
        M = wigner_M(l)
        assert M.shape == (2 * l + 1, 2 * l + 1)
        assert M.dtype == np.complex128
