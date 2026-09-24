"""Limb-darkening laws and their effect on the forward model.

A limb-darkening law scales the emergent intensity by w(mu), where
mu = cos(viewing angle) is 1 at disk centre and 0 at the limb. Every
standard law is a short power series::

    w(mu) = sum_k u_k mu^{p_k}

and in the kernel frame mu = sin(theta) cos(phi), so each monomial keeps
the radial x azimuthal separation of the kernel integrals (see
jittermap.forward.kernels). The darkened kernel table is therefore the same
linear combination of per-power tables::

    k^{LD}_{lm} = sum_k u_k k^{(p_k)}_{lm}

and the design matrix D(t, beta) = B(t) C(beta) k is built exactly as for
the uniform disk with k replaced. Limb darkening lives in the observer
frame, after every rotation, so the Wigner blocks are untouched.

The parity selection rules (astrometry blind to even l >= 4, photometry to
odd l >= 3) hold only for the uniform disk; a darkened disk has small but
non-zero kernels there.
"""

import numpy as np

from jittermap.forward.kernels import compute_A_lm, compute_A_lm_photo


class LimbDarkening:
    """Limb-darkening law w(mu) = sum_k coeffs[k] * mu**powers[k].

    Use the constructors ``linear``, ``quadratic`` and ``nonlinear`` for the
    standard parameterizations, or pass powers and coefficients directly.

    Parameters
    ----------
    powers : sequence of float
        Exponents p_k >= 0 of mu.
    coeffs : sequence of float
        Coefficients u_k, same length as ``powers``.
    name : str
        Label used in ``repr``.

    Examples
    --------
    >>> ld = LimbDarkening.quadratic(0.4, 0.26)
    >>> fm = ForwardModel(times, l_max=10, limb_darkening=ld)
    """

    def __init__(self, powers, coeffs, name="power-series"):
        powers = np.atleast_1d(np.asarray(powers, dtype=float))
        coeffs = np.atleast_1d(np.asarray(coeffs, dtype=float))
        if powers.shape != coeffs.shape:
            raise ValueError("powers and coeffs must have the same length")
        if np.any(powers < 0):
            raise ValueError("powers must be non-negative")
        self.powers = powers
        self.coeffs = coeffs
        self.name = name

    # -- standard laws -------------------------------------------------------

    @classmethod
    def linear(cls, u):
        """w(mu) = 1 - u (1 - mu)."""
        return cls([0, 1], [1 - u, u], name=f"linear(u={u:g})")

    @classmethod
    def quadratic(cls, a, b):
        """w(mu) = 1 - a (1 - mu) - b (1 - mu)^2."""
        return cls([0, 1, 2], [1 - a - b, a + 2 * b, -b],
                   name=f"quadratic(a={a:g}, b={b:g})")

    @classmethod
    def nonlinear(cls, c1, c2, c3, c4):
        """Claret four-parameter law
        w(mu) = 1 - sum_{j=1}^{4} c_j (1 - mu^{j/2})."""
        return cls([0, 0.5, 1, 1.5, 2], [1 - c1 - c2 - c3 - c4, c1, c2, c3, c4],
                   name=f"nonlinear(c1={c1:g}, c2={c2:g}, c3={c3:g}, c4={c4:g})")

    # -- evaluation ------------------------------------------------------------

    def __call__(self, mu):
        """Intensity scaling w(mu) for mu in [0, 1]."""
        mu = np.asarray(mu, dtype=float)
        return sum(u * mu ** p for u, p in zip(self.coeffs, self.powers))

    def disk_flux(self):
        """Flux of a uniform unit-intensity disk under this law,
        F_0 = 2 pi int_0^1 mu w(mu) dmu = 2 pi sum_k u_k / (p_k + 2).
        Equals pi for the uniform disk."""
        return 2 * np.pi * np.sum(self.coeffs / (self.powers + 2))

    def kernel_tables(self, lp):
        """Darkened kernel tables {'x': A_lm_x, 'y': A_lm_y, 'p': A_lm_photo}
        for max degree lp, each the coefficient-weighted sum of the
        per-power tables from jittermap.forward.kernels."""
        shape = (lp + 1, 2 * lp + 1)
        tables = {c: np.zeros(shape, dtype=complex) for c in "xyp"}
        for u, p in zip(self.coeffs, self.powers):
            A_x, A_y = compute_A_lm(lp, power=p)
            tables["x"] += u * A_x
            tables["y"] += u * A_y
            tables["p"] += u * compute_A_lm_photo(lp, power=p)
        return tables

    def __repr__(self):
        return f"LimbDarkening({self.name})"
