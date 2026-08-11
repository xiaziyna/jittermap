"""Inference tests: noiseless recovery on the identifiable subspace,
inclination estimation, and consistency of the solvers."""

import numpy as np
import pytest

from jittermap.forward.design import ForwardModel
from jittermap.harmonics.indexing import (SHIndexer, build_real_surface_transform,
                                          project_real_surface)
from jittermap.harmonics.surfaces import multispot_surface
from jittermap.inference.inversion import (gmrf_precision_diag, solve_ridge,
                                           solve_ridge_real_constraint)
from jittermap.inference.reconstruct import reconstruct

L = 6
N = 60
TIMES = np.linspace(0, 2 * np.pi, N, endpoint=False)
INC = 0.6


@pytest.fixture(scope="module")
def setup():
    fm = ForwardModel(TIMES, L)
    s_true = multispot_surface([(35, 140, 14.0)], L, texture_amplitude=0.005,
                               texture_seed=7)
    y = fm.observe(s_true, INC, channels="xyp", stacked=False, sigma=0.0)
    return fm, s_true, y


def test_solve_ridge_lam0_is_lstsq(setup):
    fm, s_true, y = setup
    D = fm.design(INC, channels="xyp", stacked=True)
    y_vec = np.concatenate([y[c] for c in "xyp"])
    q = gmrf_precision_diag(L)
    s0 = solve_ridge(D, y_vec, q, 0)
    s_lstsq, *_ = np.linalg.lstsq(D, y_vec, rcond=None)
    assert np.allclose(s0, s_lstsq)


def test_noiseless_joint_recovery(setup):
    """With all three channels, tiny regularization, and the true
    inclination, the fit must reproduce the observations essentially
    exactly, and recover the observable component of the surface."""
    fm, s_true, y = setup
    res = reconstruct(y, TIMES, L, channels="xyp", inclination=INC,
                      lam=1e-10, model=fm)
    D = fm.design(INC, channels="xyp", stacked=True)
    y_vec = np.concatenate([y[c] for c in "xyp"])
    assert np.linalg.norm(y_vec - D.dot(res.s_hat)) < 1e-8 * np.linalg.norm(y_vec)
    # Observable subspace: project both s onto the row space of D
    _, sv, Vh = np.linalg.svd(D, full_matrices=False)
    rank = int(np.sum(sv > sv[0] * 1e-10))
    P = Vh[:rank].conj().T @ Vh[:rank]
    assert np.allclose(P @ res.s_hat, P @ s_true, atol=1e-6)


def test_inclination_recovered(setup):
    fm, s_true, y = setup
    res = reconstruct(y, TIMES, L, channels="xyp", inclination=None,
                      lam=2e-6, model=fm)
    assert abs(res.inclination - INC) < 0.02


def test_real_constraint_gives_real_surface(setup):
    fm, s_true, y = setup
    D = fm.design(INC, channels="xyp", stacked=True)
    y_vec = np.concatenate([y[c] for c in "xyp"])
    q = gmrf_precision_diag(L)
    R = build_real_surface_transform(L)
    s_hat = solve_ridge_real_constraint(D, y_vec, q, 1e-6, R)
    assert np.allclose(s_hat, project_real_surface(s_hat, L), atol=1e-8)


def test_astrometry_only_and_photometry_only_run(setup):
    fm, s_true, y = setup
    res_xy = reconstruct({c: y[c] for c in "xy"}, TIMES, L, channels="xy",
                         inclination=INC, lam=1e-6, model=fm)
    res_p = reconstruct({"p": y["p"]}, TIMES, L, channels="p",
                        inclination=INC, lam=1e-6, model=fm)
    assert res_xy.s_hat.shape == ((L + 1) ** 2,)
    assert res_p.s_hat.shape == ((L + 1) ** 2,)
