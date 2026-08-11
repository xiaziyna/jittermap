"""Rendering of the visible stellar hemisphere.

The observer sits on the +X axis; image axes are Y (horizontal) and Z
(vertical). Surfaces are drawn in the observer frame at spin phase
omega*t and inclination beta, using the same rotation convention as the
forward model.
"""

import numpy as np
from jittermap.harmonics._compat import sph_harm

from jittermap.harmonics.indexing import SHIndexer
from jittermap.forward.wigner import precompute_rotations


def complex_sph_harm(l, m, theta, phi):
    """Complex spherical harmonic Y_l^m(theta, phi) with scipy's
    (m, l, phi, theta) argument convention."""
    return sph_harm(m, l, phi, theta)


def build_projection_grid(n_grid=600):
    """Build (X, Y, Z, THETA, PHI, mask) for the +X visible hemisphere."""
    y = np.linspace(-1, 1, n_grid)
    z = np.linspace(-1, 1, n_grid)
    Y, Z = np.meshgrid(y, z)
    R = np.hypot(Y, Z)
    mask = R > 1
    X = np.sqrt(np.clip(1 - Y ** 2 - Z ** 2, 0, None))
    X[mask] = np.nan
    Y[mask] = np.nan
    Z[mask] = np.nan
    THETA = np.arccos(np.clip(Z, -1, 1))
    PHI = np.arctan2(Y, X)
    return X, Y, Z, THETA, PHI, mask


def render_surface(x_coeffs, l_max, inclination, THETA, PHI, mask,
                   t=0.0, omega=1.0, rotations=None):
    """Render the surface map via Wigner rotation of the coefficients.

    Reference implementation; render_surface_fast is preferred for
    large grids or high degree.
    """
    if rotations is None:
        rotations = precompute_rotations(l_max, omega)
    sh = SHIndexer(l_max=l_max)
    Y_img = np.zeros_like(THETA, dtype=complex)
    for lp in range(l_max + 1):
        B_func, C_func = rotations[lp]
        F = C_func(inclination).T @ B_func(t).T
        coeff_rot = F @ x_coeffs[sh.get_l_indices(lp)]
        for m in range(-lp, lp + 1):
            Y_img += coeff_rot[m + lp] * complex_sph_harm(lp, m, THETA, PHI)
    Y_real = np.real(Y_img)
    Y_real[mask] = np.nan
    return Y_real


def render_surface_fast(x_coeffs, l_max, inclination, THETA, PHI, mask,
                        X_grid, Y_grid, Z_grid, t=0.0, omega=1.0):
    """Fast rendering by rotating the observer grid into the body frame
    with a 3x3 rotation (matching the Wigner D convention) and evaluating
    the SH expansion directly."""
    sh = SHIndexer(l_max=l_max)

    def _Rz(a):
        ca, sa = np.cos(a), np.sin(a)
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])

    def _Ry(b):
        cb, sb = np.cos(b), np.sin(b)
        return np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])

    B_rot = _Rz(omega * t + np.pi / 2) @ _Ry(np.pi / 2) @ _Rz(np.pi / 2)
    C_rot = _Rz(-inclination + np.pi / 2) @ _Ry(np.pi / 2) @ _Rz(np.pi / 2)
    R = B_rot @ C_rot

    coords = np.stack([X_grid, Y_grid, Z_grid], axis=-1)
    body = coords @ R.T
    theta_body = np.arccos(np.clip(body[..., 2], -1, 1))
    phi_body = np.arctan2(body[..., 1], body[..., 0])

    flat_theta = theta_body.ravel()
    flat_phi = phi_body.ravel()
    Y_basis = np.zeros((len(flat_theta), sh.total_coeffs), dtype=complex)
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Y_basis[:, sh.get_index(l, m)] = sph_harm(m, l, flat_phi, flat_theta)
    Y_img = (Y_basis @ x_coeffs).reshape(THETA.shape)

    Y_real = Y_img.real
    Y_real[mask] = np.nan
    return Y_real


def draw_latlon_grid(ax, inclination_rad, num_meridians=7, num_parallels=5,
                     color="k", alpha=0.25, lw=0.8):
    """Overlay latitude/longitude grid lines on the projected sphere."""
    cb, sb = np.cos(inclination_rad), np.sin(inclination_rad)
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])

    theta = np.linspace(0, np.pi, 200)
    for phi0 in np.linspace(-np.pi, np.pi, num_meridians):
        coords = np.vstack([np.sin(theta) * np.cos(phi0),
                            np.sin(theta) * np.sin(phi0),
                            np.cos(theta)])
        rot = Ry @ coords
        vis = rot[0] >= 0
        if np.any(vis):
            ax.plot(rot[1, vis], rot[2, vis], color=color, alpha=alpha, lw=lw)

    phi = np.linspace(-np.pi, np.pi, 400)
    for theta0 in np.linspace(np.pi / 6, 5 * np.pi / 6, num_parallels):
        coords = np.vstack([np.sin(theta0) * np.cos(phi),
                            np.sin(theta0) * np.sin(phi),
                            np.cos(theta0) * np.ones_like(phi)])
        rot = Ry @ coords
        vis = rot[0] >= 0
        if np.any(vis):
            ax.plot(rot[1, vis], rot[2, vis], color=color, alpha=alpha, lw=lw)
