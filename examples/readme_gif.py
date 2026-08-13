"""Generate the README/docs header animation.

A realistic Monte Carlo spotted star (log-normal spot-size population in
activity belts, plus dominant belt spots) at degree L = 40, shown rotating
at three labeled inclinations with spin-axis, meridian, and parallel
overlays.

Rendering uses the Wigner coefficient-rotation path with a precomputed
spherical-harmonic basis matrix, so each frame is a single matrix-vector
product even at L = 40.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 72
matplotlib.rcParams["figure.dpi"] = 72
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import jittermap as jm
from jittermap.harmonics.indexing import SHIndexer
from jittermap.harmonics._compat import sph_harm
from jittermap.plotting.render import build_projection_grid
from jittermap.plotting.overlays import MeridianOverlay

# ---- settings -------------------------------------------------------------
L = 40
N_GRID = 130
N_FRAMES = 36
FPS = 14
BETAS = [np.deg2rad(10), np.deg2rad(45), np.deg2rad(75)]
BETA_LABELS = [r"$\beta = 10^\circ$ (near equator-on)",
               r"$\beta = 45^\circ$",
               r"$\beta = 75^\circ$ (near pole-on)"]
SEED = 7
OUT = "spin.gif"
# ---------------------------------------------------------------------------


def _separation_deg(lat1, lon1, lat2, lon2):
    """Great-circle separation between two (lat, lon) points, in degrees."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dlon = np.deg2rad(lon1 - lon2)
    cos_sep = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dlon)
    return float(np.rad2deg(np.arccos(np.clip(cos_sep, -1.0, 1.0))))


def draw_monte_carlo_star(rng, f_spot=0.05, r_med=0.055, lognorm_sigma=0.9,
                          r_min=0.061, r_max=0.16, belt_lat=18.0,
                          belt_sigma=6.0, lon_sigma=35.0, pair_prob=0.6,
                          max_spots=50, gap_deg=2.5):
    """Draw a solar-style spot population.

    Follows the observed statistics of sunspot groups:

    * log-normal spot sizes (Bogdan et al. 1988; Baumann & Solanki 2005),
      truncated below at r_min so every spot is resolved at the rendering
      degree (caps below ~180/L degrees truncate shallow, breaking the
      constant-contrast appearance);
    * Gaussian activity belts at +-belt_lat (the butterfly diagram at a
      fixed cycle phase; solar spots rarely appear above ~30 deg latitude);
    * two active longitudes ~180 deg apart, with Gaussian scatter;
    * bipolar emergence: large spots carry a smaller trailing companion
      slightly poleward (Joy's law tilt).

    All spots share the same contrast, so positions are rejection-sampled to
    be non-overlapping (overlapping caps stack coefficients and render
    darker than the nominal contrast).
    """
    spots = []
    active_lons = rng.uniform(0.0, 360.0) + np.array([0.0, 180.0])

    def draw_radius():
        while True:
            r = r_med * rng.lognormal(0.0, lognorm_sigma)
            if r_min <= r <= r_max:
                return r

    def clear(lat, lon, rad):
        return all(_separation_deg(lat, lon, la, lo) > rad + rd + gap_deg
                   for la, lo, rd in spots)

    def place(rad, tries=200):
        for _ in range(tries):
            hemi = rng.choice([-1.0, 1.0])
            lat = hemi * abs(rng.normal(belt_lat, belt_sigma))
            lon = (rng.choice(active_lons) + rng.normal(0.0, lon_sigma)) % 360.0
            if clear(lat, lon, rad):
                spots.append((lat, lon, rad))
                return lat, lon
        return None

    f = 0.0
    while f < f_spot and len(spots) < max_spots:
        r = draw_radius()
        placed = place(jm.rfrac_to_deg(r))
        if placed is None:
            break
        f += r ** 2
        # bipolar group: trailing companion at ~half the radius, offset in
        # longitude and slightly poleward of the leading spot
        if rng.uniform() < pair_prob and r > 1.5 * r_min:
            r_c = max(r * rng.uniform(0.45, 0.65), r_min)
            rad, rad_c = jm.rfrac_to_deg(r), jm.rfrac_to_deg(r_c)
            lat, lon = placed
            lon_c = (lon + (rad + rad_c + gap_deg + rng.uniform(1.0, 5.0))) % 360.0
            lat_c = lat + np.sign(lat) * rng.uniform(1.0, 5.0)
            if clear(lat_c, lon_c, rad_c):
                spots.append((lat_c, lon_c, rad_c))
                f += r_c ** 2
    return spots


def main():
    rng = np.random.default_rng(SEED)
    spots = draw_monte_carlo_star(rng)
    print(f"{len(spots)} spots, radii "
          f"{min(s[2] for s in spots):.1f}-{max(s[2] for s in spots):.1f} deg")
    s = jm.multispot_surface(spots, L, contrast=0.7, sigma_taper=True)

    sh = SHIndexer(L)
    X, Y, Z, THETA, PHI, mask = build_projection_grid(n_grid=N_GRID)
    flat_theta = THETA.ravel()
    flat_phi = PHI.ravel()

    print("precomputing SH basis...")
    basis = np.zeros((flat_theta.size, sh.total_coeffs), dtype=np.complex64)
    for l in range(L + 1):
        for m in range(-l, l + 1):
            basis[:, sh.get_index(l, m)] = sph_harm(m, l, flat_phi, flat_theta)

    rotations = jm.precompute_rotations(L)

    def frame_map(beta, t, peak=None):
        s_rot = np.zeros_like(s)
        for lp in range(L + 1):
            B_func, C_func = rotations[lp]
            F = C_func(beta).T @ B_func(t).T
            s_rot[sh.get_l_indices(lp)] = F @ s[sh.get_l_indices(lp)]
        img = (basis @ s_rot.astype(np.complex64)).real.reshape(THETA.shape)
        img = np.asarray(img, dtype=float)
        img[mask] = np.nan
        if peak is None:
            return img
        return img / peak

    # one normalization per panel (constant over frames: no brightness pumping)
    peaks = {}
    for beta in BETAS:
        p = max(np.nanmax(np.abs(frame_map(beta, t, peak=None)))
                for t in np.linspace(0, 2 * np.pi, 8, endpoint=False))
        peaks[beta] = p if p else 1.0

    t_vals = np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    fig.patch.set_facecolor("white")
    ims, overlays = [], []
    for ax, beta, label in zip(axes, BETAS, BETA_LABELS):
        img0 = frame_map(beta, t_vals[0], peak=peaks[beta])
        im = ax.imshow(np.flip(img0, axis=0), cmap="inferno",
                       vmin=-1.0, vmax=0.25, extent=[-1, 1, -1, 1])
        ax.set_title(label, fontsize=11)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        mo = MeridianOverlay(ax, beta, num_meridians=7, parallels=5,
                             lw=1.4)
        ims.append(im)
        overlays.append(mo)
    fig.tight_layout()

    def update(i):
        artists = []
        for im, mo, beta in zip(ims, overlays, BETAS):
            im.set_data(np.flip(frame_map(beta, t_vals[i], peak=peaks[beta]), axis=0))
            artists.append(im)
            artists.extend(mo.update(t_vals[i]))
        return tuple(artists)

    ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=60, blit=True)
    ani.save(OUT, writer="pillow", fps=FPS)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
