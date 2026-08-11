"""Comparison panel figures: truth vs per-channel reconstructions."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from jittermap.plotting.render import draw_latlon_grid, render_surface_fast

# Star-like default: bright photosphere, dark starspots.
DEFAULT_CMAP = "inferno"
# Normalized surface values span [-1 (darkest spot), ~small positive];
# placing the zero-level background high in the colormap makes the disk
# glow while spots stay dark.
DEFAULT_CLIM = (-1.0, 0.25)


def normalize_map(m, negate=False):
    """Normalize a rendered map to unit max amplitude (spots are negative
    brightness; keep the sign so sequential star-like colormaps render the
    background bright and the spots dark). Set negate=True for diverging
    colormaps such as RdBu_r where spots should map to the red end."""
    peak = np.nanmax(np.abs(m))
    if peak == 0:
        return m
    return (-m if negate else m) / peak


def render_panel(ax, m, title, inclination, cmap=DEFAULT_CMAP, grid=True,
                 clim=DEFAULT_CLIM):
    """Render a single projected-surface panel with lat/lon overlay.

    Expects a map normalized by normalize_map (background near 0, spots
    toward -1). Pass clim=None to autoscale instead.
    """
    vmin, vmax = clim if clim is not None else (None, None)
    ax.imshow(np.flip(m, axis=0), cmap=cmap, extent=[-1, 1, -1, 1],
              vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(labelsize=8)
    if grid:
        draw_latlon_grid(ax, inclination_rad=inclination, color="w",
                         alpha=0.3)


def comparison_figure(true_coeffs, results, l_true, inc_true,
                      n_grid=300, l_fit=None, figsize_scale=3.2):
    """Truth-vs-reconstructions comparison figure.

    Parameters
    ----------
    true_coeffs : complex ndarray
        Ground-truth surface coefficients (degree l_true).
    results : dict
        {label: ReconstructionResult} — one panel per entry, in order.
    l_true : int
        Degree of the true surface.
    inc_true : float
        True inclination (radians).

    Returns
    -------
    fig, axes
    """
    from jittermap.plotting.render import build_projection_grid

    X, Y, Z, THETA, PHI, mask = build_projection_grid(n_grid=n_grid)
    n_panels = 1 + len(results)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(figsize_scale * n_panels, figsize_scale))

    true_map = normalize_map(render_surface_fast(
        true_coeffs, l_true, inc_true, THETA, PHI, mask, X, Y, Z))
    render_panel(axes[0], true_map,
                 r"Truth ($\beta$" + f"={inc_true:.2f})", inc_true)

    for ax, (label, res) in zip(axes[1:], results.items()):
        lf = l_fit if l_fit is not None else l_true
        m = normalize_map(render_surface_fast(
            res.s_hat, lf, res.inclination, THETA, PHI, mask, X, Y, Z))
        render_panel(ax, m,
                     f"{label} " + r"($\hat\beta$" + f"={res.inclination:.2f})",
                     res.inclination)
    fig.tight_layout()
    return fig, axes


def gallery_figure(results_by_snr, snr_values, l_fit, n_times,
                   method_keys=("map_p", "map_xy", "map_all"),
                   method_labels=("Photometry", "Astrometry", "Joint")):
    """Gallery layout from the paper's supplementary figures: rows are
    methods, columns are SNR levels, with the ground truth centered in a
    separate left column.

    Parameters
    ----------
    results_by_snr : list of dict
        One dict per SNR level with keys 'true_map', the method map keys,
        the matching inclination keys ('inc_p', 'inc_xy', 'inc_all',
        'inc_true'), all maps already rendered and normalized.
    snr_values : list
        SNR per column (None = noiseless).
    """
    n_snr = len(snr_values)
    n_methods = len(method_keys)
    inc_keys = [k.replace("map", "inc") for k in method_keys]

    fig = plt.figure(figsize=(3.2 * (1 + n_snr) + 0.6, 3.2 * n_methods))
    gs = GridSpec(n_methods, 2 + n_snr, figure=fig,
                  width_ratios=[1, 0.15] + [1] * n_snr,
                  wspace=0.08, hspace=0.25)

    r0 = results_by_snr[0]
    ax_true = fig.add_subplot(gs[min(1, n_methods - 1), 0])
    render_panel(ax_true, r0["true_map"],
                 r"$\mathbf{Ground\ Truth}$ ($\beta$" + f'={r0["inc_true"]:.2f})',
                 r0["inc_true"])
    for row in range(n_methods):
        if row != min(1, n_methods - 1):
            ax_empty = fig.add_subplot(gs[row, 0])
            ax_empty.axis("off")
            if row == 0:
                ax_empty.set_title(f"$L={l_fit},\\ N={n_times}$",
                                   fontsize=10, loc="left")
        fig.add_subplot(gs[row, 1]).axis("off")

    snr_col_labels = [r"$\mathbf{Noiseless}$" if s is None
                      else r"$\mathbf{" + str(s) + r"}$" for s in snr_values]
    for col_idx, (r, snr) in enumerate(zip(results_by_snr, snr_values)):
        for row_idx, (mk, ik, mlabel) in enumerate(
                zip(method_keys, inc_keys, method_labels)):
            ax = fig.add_subplot(gs[row_idx, col_idx + 2])
            inc_hat = r[ik]
            if row_idx == 0:
                header = (r"$\mathbf{SNR}$: " + snr_col_labels[col_idx]
                          if col_idx == 0 else snr_col_labels[col_idx])
                title = header + "\n" + r"$\hat{\beta}$" + f"={inc_hat:.2f}"
            else:
                title = r"$\hat{\beta}$" + f"={inc_hat:.2f}"
            render_panel(ax, r[mk], title, inc_hat)
            if col_idx == 0:
                ax.set_ylabel(mlabel, fontsize=11, fontweight="bold")
    return fig
