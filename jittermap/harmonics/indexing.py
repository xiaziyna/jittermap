"""Spherical harmonic indexing and real-surface algebra.

Complex SH coefficients s_{l,m} are stored in a flat vector with the
standard linear index (l, m) -> l^2 + l + m, so degree l occupies the
contiguous block [l^2, (l+1)^2).

A real-valued surface satisfies the conjugate symmetry
    s_{l,-m} = (-1)^m conj(s_{l,m}),   s_{l,0} real.
"""

import numpy as np


class SHIndexer:
    """Minimal spherical harmonics indexer for (l, m) pairs."""

    def __init__(self, l_max):
        self.l_max = l_max
        self.total_coeffs = (l_max + 1) ** 2

    def get_index(self, l, m):
        """(l, m) -> linear index."""
        return l * l + l + m

    def get_lm(self, index):
        """Linear index -> (l, m)."""
        l = int(np.sqrt(index))
        m = index - l * l - l
        return l, m

    def get_l_coeffs(self, l):
        """Number of coefficients for given l."""
        return 2 * l + 1

    def get_l_indices(self, l):
        """All linear indices for given l."""
        return np.arange(l * l, (l + 1) ** 2)

    def get_keep_idx(self):
        """Indices of degrees observable by the astrometric/photometric
        kernels: keep l <= 2 and odd l (even l > 2 are in the null space).
        """
        keep_mask = np.zeros(self.total_coeffs, dtype=bool)
        for lv in range(self.l_max + 1):
            keep = (lv <= 2) or (lv % 2 == 1)
            keep_mask[self.get_l_indices(lv)] = keep
        return np.where(keep_mask)[0]


def build_real_surface_transform(l_max):
    """Build the reparameterization matrix R such that s = R r (with r real)
    enforces a real surface.

    For each degree l:
      - m=0: s_{l,0} is real
      - m>0: s_{l,m} = a + ib, s_{l,-m} = (-1)^m (a - ib)

    Returns
    -------
    R : ndarray, shape (p, p), complex
    """
    sh = SHIndexer(l_max=l_max)
    p = sh.total_coeffs
    R = np.zeros((p, p), dtype=complex)
    col = 0
    for l in range(l_max + 1):
        idx0 = sh.get_index(l, 0)
        R[idx0, col] = 1.0
        col += 1
        for m in range(1, l + 1):
            idx_pos = sh.get_index(l, m)
            idx_neg = sh.get_index(l, -m)
            R[idx_pos, col] = 1.0
            R[idx_neg, col] = (-1) ** m
            col += 1
            R[idx_pos, col] = 1j
            R[idx_neg, col] = -1j * ((-1) ** m)
            col += 1
    return R


def project_real_surface(s, l_max):
    """L2 projection of a coefficient vector onto the real-surface subspace:
    s_{l,-m} = (-1)^m conj(s_{l,m}), and s_{l,0} real.
    """
    sh = SHIndexer(l_max=l_max)
    s_proj = s.copy()
    for l in range(l_max + 1):
        idx0 = sh.get_index(l, 0)
        s_proj[idx0] = np.real(s_proj[idx0])
        for m in range(1, l + 1):
            idx_pos = sh.get_index(l, m)
            idx_neg = sh.get_index(l, -m)
            a = s[idx_pos]
            b = s[idx_neg]
            a_proj = 0.5 * (a + ((-1) ** m) * np.conj(b))
            s_proj[idx_pos] = a_proj
            s_proj[idx_neg] = ((-1) ** m) * np.conj(a_proj)
    return s_proj


def mse(a, b):
    """Sum of squared moduli of (a - b), zero-padding b if shorter."""
    if b.shape != a.shape:
        b_zero = np.zeros_like(a)
        b_zero[: b.shape[0]] = b
        return float(np.sum(np.abs(a - b_zero) ** 2))
    return float(np.sum(np.abs(a - b) ** 2))
