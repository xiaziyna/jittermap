"""Design matrix construction for the astrometric / photometric forward model.

The measured signal for a channel with kernel table A_lm is
    y = D(times, beta) s,        D[n, (l,m)] built per degree l as
    D_l = diag(e^{-i m omega t_n}) M_l C(beta) a_l
using the Vandermonde decomposition B(t) = diag(e^{-i m omega t}) B(0)
(see jittermap.forward.wigner). For a real surface s the signal y is real
up to numerical precision.

Two paths are provided:
  - design_matrix_reference: direct per-timepoint evaluation (slow, used
    as ground truth in tests)
  - precompute_vandermonde + design_matrix_vandermonde: the fast path,
    O(N L^2) elementwise operations with no per-timepoint matmul
and a high-level ForwardModel class wrapping channel assembly.
"""

import numpy as np

from jittermap.harmonics.indexing import SHIndexer
from jittermap.forward.wigner import rotation_funcs, wigner_M
from jittermap.forward.kernels import compute_A_lm, compute_A_lm_photo

CHANNEL_KEYS = ("x", "y", "p")


def design_matrix_reference(times, l, inclination, A_lm, omega=1.0):
    """Reference design matrix by direct evaluation of B(t) C(beta) a_l
    at every timepoint. Slow; kept as the ground-truth implementation.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Rotation phases omega * t.
    l : int
        Maximum spherical harmonic degree.
    inclination : float
        Stellar inclination beta in radians.
    A_lm : ndarray, shape (l+1, 2*l+1)
        Kernel table from compute_A_lm / compute_A_lm_photo.

    Returns
    -------
    design_matrix : complex ndarray, shape (N, (l+1)^2)
    """
    sh = SHIndexer(l_max=l)
    design_matrix = np.zeros((times.shape[0], sh.total_coeffs), dtype=complex)
    for lp in range(l + 1):
        B_func, C_func = rotation_funcs(lp, omega)
        C = C_func(inclination)
        C_a = C.dot(A_lm[lp, l - lp: l + lp + 1])
        for t_idx, t in enumerate(times):
            design_matrix[t_idx, sh.get_l_indices(lp)] += B_func(t).dot(C_a)
    return design_matrix


def precompute_vandermonde(times, l_max, omega=1.0):
    """Precompute the pieces of the Vandermonde decomposition.

    Returns
    -------
    B0_list : list of (2lp+1, 2lp+1) complex arrays
        B(0) = M_lp per degree.
    C_funcs : list of callables
        C_funcs[lp](inclination) -> (2lp+1, 2lp+1) matrix.
    phases_list : list of (N, 2lp+1) arrays
        phases_list[lp][n, m+lp] = e^{-i m omega t_n}.
    """
    B0_list = []
    C_funcs = []
    phases_list = []
    for lp in range(l_max + 1):
        _, C_func = rotation_funcs(lp, omega)
        B0_list.append(wigner_M(lp))
        C_funcs.append(C_func)
        m_vals = np.arange(-lp, lp + 1)
        phases_list.append(np.exp(-1j * np.outer(omega * times, m_vals)))
    return B0_list, C_funcs, phases_list


def design_matrix_vandermonde(l, inclination, A_lm, B0_list, C_funcs, phases_list):
    """Fast design matrix via the Vandermonde decomposition.

    D[:, l-block] = phases * (B(0) C(beta) a_l): O(N L^2) elementwise
    operations, no per-timepoint matrix multiply.
    """
    N = phases_list[0].shape[0]
    sh = SHIndexer(l_max=l)
    design_matrix = np.zeros((N, sh.total_coeffs), dtype=complex)
    for lp in range(l + 1):
        C = C_funcs[lp](inclination)
        C_a = C.dot(A_lm[lp, l - lp: l + lp + 1])
        v = B0_list[lp] @ C_a
        design_matrix[:, sh.get_l_indices(lp)] += phases_list[lp] * v[np.newaxis, :]
    return design_matrix


class ForwardModel:
    """Forward model mapping SH surface coefficients to time series of
    astrometric photocenter shifts (x, y) and disk-integrated photometry (p).

    Parameters
    ----------
    times : ndarray, shape (N,)
        Observation times; the rotation phase is omega * t.
    l_max : int
        Maximum spherical harmonic degree of the surface model.
    omega : float
        Rotation rate (radians per unit time).

    Examples
    --------
    >>> fm = ForwardModel(np.linspace(0, 2*np.pi, 100), l_max=10)
    >>> D = fm.design(inclination=0.6, channels='xyp')   # stacked (3N, p)
    >>> y = fm.observe(s, inclination=0.6, channels='xyp')
    """

    def __init__(self, times, l_max, omega=1.0):
        self.times = np.asarray(times, dtype=float)
        self.l_max = l_max
        self.omega = omega
        self.n_times = len(self.times)
        self._precomp = precompute_vandermonde(self.times, l_max, omega)
        self._kernels = None

    @property
    def kernels(self):
        """Dict of kernel tables {'x': A_lm_x, 'y': A_lm_y, 'p': A_lm_photo}."""
        if self._kernels is None:
            A_lm_x, A_lm_y = compute_A_lm(self.l_max)
            self._kernels = {
                "x": A_lm_x,
                "y": A_lm_y,
                "p": compute_A_lm_photo(self.l_max),
            }
        return self._kernels

    def design_channel(self, inclination, channel):
        """Design matrix for one channel ('x', 'y' or 'p'), shape (N, p)."""
        if channel not in CHANNEL_KEYS:
            raise ValueError(f"unknown channel {channel!r}; expected one of {CHANNEL_KEYS}")
        return design_matrix_vandermonde(self.l_max, inclination,
                                         self.kernels[channel], *self._precomp)

    def design(self, inclination, channels="xyp", stacked=True):
        """Design matrices for a set of channels.

        Parameters
        ----------
        inclination : float
            Stellar inclination beta in radians.
        channels : str
            Subset of 'xyp', e.g. 'xy' for astrometry only.
        stacked : bool
            If True, vertically stack the per-channel matrices
            (shape (len(channels)*N, p)); otherwise return a dict.
        """
        mats = {c: self.design_channel(inclination, c) for c in channels}
        if stacked:
            return np.vstack([mats[c] for c in channels])
        return mats

    def observe(self, s, inclination, channels="xyp", sigma=0.0, rng=None,
                stacked=True):
        """Simulate observations y = D s (+ noise) for each channel.

        Parameters
        ----------
        s : complex ndarray, shape ((l_max+1)^2,)
            Surface coefficients (real-surface symmetric).
        sigma : float or dict
            Gaussian noise standard deviation, per channel if a dict.
        rng : np.random.Generator or None

        Returns
        -------
        Stacked real array (len(channels)*N,) if stacked, else dict of
        per-channel arrays (N,).
        """
        if rng is None:
            rng = np.random.default_rng()
        out = {}
        for c in channels:
            y = self.design_channel(inclination, c).dot(s)
            y = np.real(y)
            sig = sigma[c] if isinstance(sigma, dict) else sigma
            if sig and sig > 0:
                y = y + rng.normal(0, sig, y.shape)
            out[c] = y
        if stacked:
            return np.concatenate([out[c] for c in channels])
        return out
