"""Pyramid plots of spherical-harmonic tables.

The pyramid layout displays one value per (l, m) cell, rows l = 0..L,
columns m = -l..l centered, with the degree printed in each row — the
figure style used throughout the accompanying paper for surface
coefficients, measurement kernels, and rotated-kernel magnitudes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from jittermap.harmonics.indexing import SHIndexer
from jittermap.forward.wigner import wigner_M, rotation_funcs


def coefficient_pyramid(s, l_max):
    """Pyramid array of |s_{l,m}| from a flat coefficient vector."""
    sh = SHIndexer(l_max)
    grid_size = 2 * l_max + 1
    pyramid = np.full((l_max + 1, grid_size), np.nan, dtype=float)
    for l in range(l_max + 1):
        start = (grid_size - (2 * l + 1)) // 2
        pyramid[l, start:start + 2 * l + 1] = np.abs(s[sh.get_l_indices(l)])
    return pyramid


def kernel_pyramid(A_lm, l_max):
    """Pyramid array of |k_{l,m}| from a kernel table of shape
    (l_max+1, 2*l_max+1) (compute_A_lm / compute_A_lm_photo output)."""
    grid_size = 2 * l_max + 1
    pyramid = np.full((l_max + 1, grid_size), np.nan, dtype=float)
    for l in range(l_max + 1):
        start = (grid_size - (2 * l + 1)) // 2
        pyramid[l, start:start + 2 * l + 1] = np.abs(
            A_lm[l, l_max - l: l_max + l + 1])
    return pyramid


def rotated_kernel_pyramid(l_max, inclination, A_lm, threshold=0.0,
                           omega=1.0):
    """Pyramid of the inclination-rotated kernel magnitudes |c^l_m| with
    c^l = M_l C(beta) k^h_l — the diagonal entries of the B_beta operator
    (paper Eq. A9). Cells below threshold are zeroed (rendered white)."""
    grid_size = 2 * l_max + 1
    pyramid = np.full((l_max + 1, grid_size), np.nan, dtype=float)
    for lp in range(l_max + 1):
        _, C_func = rotation_funcs(lp, omega)
        c = wigner_M(lp) @ (C_func(inclination)
                            @ A_lm[lp, l_max - lp: l_max + lp + 1])
        vals = np.abs(c)
        vals[vals < threshold] = 0.0
        start = (grid_size - (2 * lp + 1)) // 2
        pyramid[lp, start:start + 2 * lp + 1] = vals
    return pyramid


def plot_pyramid(ax, grid, title="", vmax=None, cmap="GnBu",
                 font_scale=1.0):
    """Draw a pyramid array in the paper's style: white masked cells,
    light-gray cell outlines, degree labels down the center, order labels
    along the bottom."""
    l_max = grid.shape[0] - 1
    grid_size = grid.shape[1]
    if vmax is None:
        vmax = np.nanmax(grid)

    norm_vals = np.clip(grid / vmax, 0.0, 1.0)
    norm_vals_masked = np.ma.masked_where(norm_vals <= 0.0, norm_vals)

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#ffffff")
    im = ax.imshow(norm_vals_masked, cmap=cm,
                   norm=mcolors.Normalize(vmin=0.0, vmax=1.0))
    ax.axis("off")

    rows = np.arange(0, l_max + 1)
    start_cols = (grid_size - (2 * rows + 1)) // 2
    end_cols = start_cols + (2 * rows)
    for l, s, e in zip(rows, start_cols, end_cols):
        ax.add_patch(plt.Rectangle((s - 0.5, l - 0.5), (e - s + 1), 1.0,
                                   fill=False, edgecolor="lightgray",
                                   linewidth=1.2))
        for c in range(s, e):
            ax.add_line(plt.Line2D([c + 0.5, c + 0.5], [l - 0.5, l + 0.5],
                                   color="lightgray", linewidth=0.6))

    center_col = grid_size // 2
    for l in range(l_max + 1):
        ax.text(center_col, l, f"{l}", ha="center", va="center",
                fontsize=20 * font_scale)
    for l in range(l_max + 1):
        start = (grid_size - (2 * l + 1)) // 2
        for i, m_val in enumerate(range(-l, l + 1)):
            ax.text(start + i, l_max + 1.1, f"{m_val}", ha="center",
                    fontsize=16 * font_scale)

    ax.text(grid_size / 2 - 0.55, l_max + 1.8, "m", ha="center",
            fontsize=16 * font_scale)
    ax.text(grid_size // 2, -0.8, r"$\ell$", ha="center", va="center",
            fontsize=20 * font_scale)
    if title:
        ax.text(-0.5, -0.8, title, ha="left", va="center",
                fontsize=22 * font_scale)
    return im
