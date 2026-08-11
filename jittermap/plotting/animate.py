"""Spin animations of rendered surfaces."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from jittermap.plotting.render import build_projection_grid, render_surface_fast
from jittermap.plotting.panels import normalize_map, render_panel


def animate_spin(coeffs, l_max, inclination, n_frames=60, omega=1.0,
                 n_grid=200, cmap="inferno", output_path=None, fps=20,
                 negate=False, overlay=True, num_meridians=5):
    """Animate one full rotation of a surface.

    Parameters
    ----------
    coeffs : complex ndarray
        SH surface coefficients.
    l_max : int
        Degree of the surface.
    inclination : float
        Inclination beta (radians).
    output_path : str or None
        If given, save the animation (gif via pillow).
    overlay : bool
        Draw the projected spin axis and co-rotating meridians.
    num_meridians : int
        Number of meridian lines when overlay is enabled.

    Returns
    -------
    FuncAnimation
    """
    from jittermap.plotting.overlays import MeridianOverlay

    X, Y, Z, THETA, PHI, mask = build_projection_grid(n_grid=n_grid)
    t_vals = np.linspace(0, 2 * np.pi / omega, n_frames, endpoint=False)

    first = normalize_map(render_surface_fast(
        coeffs, l_max, inclination, THETA, PHI, mask, X, Y, Z,
        t=t_vals[0], omega=omega), negate=negate)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(np.flip(first, axis=0), cmap=cmap, vmin=-1.0, vmax=0.25,
                   extent=[-1, 1, -1, 1])
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])

    mo = MeridianOverlay(ax, inclination, omega=omega,
                         num_meridians=num_meridians) if overlay else None

    def update(i):
        frame = normalize_map(render_surface_fast(
            coeffs, l_max, inclination, THETA, PHI, mask, X, Y, Z,
            t=t_vals[i], omega=omega), negate=negate)
        im.set_data(np.flip(frame, axis=0))
        artists = [im]
        if mo is not None:
            artists.extend(mo.update(t_vals[i]))
        return tuple(artists)

    ani = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
    if output_path is not None:
        ani.save(output_path, writer="pillow", fps=fps)
    return ani


def animate_comparison(surfaces, l_maxes, inclinations, labels,
                       n_frames=60, omega=1.0, n_grid=200,
                       cmap="inferno", output_path=None, fps=20):
    """Animate several surfaces side by side over one rotation
    (e.g. Truth | Joint | Astrometry | Photometry).

    Parameters
    ----------
    surfaces : list of complex ndarray
        Coefficient vectors, one per panel.
    l_maxes : list of int
    inclinations : list of float
    labels : list of str
    """
    n_panels = len(surfaces)
    X, Y, Z, THETA, PHI, mask = build_projection_grid(n_grid=n_grid)
    t_vals = np.linspace(0, 2 * np.pi / omega, n_frames, endpoint=False)

    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.4))
    if n_panels == 1:
        axes = [axes]
    ims = []
    for ax, s, lm, inc, lab in zip(axes, surfaces, l_maxes, inclinations, labels):
        frame = normalize_map(render_surface_fast(
            s, lm, inc, THETA, PHI, mask, X, Y, Z, t=t_vals[0], omega=omega))
        im = ax.imshow(np.flip(frame, axis=0), cmap=cmap, vmin=-1.0, vmax=0.25,
                       extent=[-1, 1, -1, 1])
        ax.set_title(lab, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ims.append(im)

    def update(i):
        for im, s, lm, inc in zip(ims, surfaces, l_maxes, inclinations):
            frame = normalize_map(render_surface_fast(
                s, lm, inc, THETA, PHI, mask, X, Y, Z, t=t_vals[i], omega=omega))
            im.set_data(np.flip(frame, axis=0))
        return tuple(ims)

    ani = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
    if output_path is not None:
        ani.save(output_path, writer="pillow", fps=fps)
    return ani
