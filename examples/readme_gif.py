"""Generate the README/docs header animation.

The surface is the real Sun: every sunspot group of Carrington rotation
2156 (2014 Oct 9 - Nov 5, the most active rotation of cycle 24, including
AR 12192, the largest group in 24 years), taken at its central-meridian
crossing from the RGO/USAF/NOAA daily group record compiled by Hathaway
(solarcyclescience.com/AR_Database). Carrington longitudes and relative
spot areas are observed; a single knob (AREA_SCALE) multiplies the areas
to turn the 2014 Sun into a young active-Sun analog, since at true solar
scale only AR 12192 is resolvable at L = 40. Consistent with that
reading, the larger groups are displaced poleward from their observed
latitudes as flux-emergence theory predicts for rapid rotators (see
below). All spots share the same contrast of 0.7.

Shown at degree L = 40, rotating at three labeled inclinations with
spin-axis, meridian, and parallel overlays. Rendering uses the Wigner
coefficient-rotation path with a precomputed spherical-harmonic basis
matrix, so each frame is a single matrix-vector product even at L = 40.
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
AREA_SCALE = 9.0   # multiply observed group areas; 1.0 = the literal Sun
# Poleward displacement of the scaled-up groups (deg per decade of group
# area above POLEWARD_AREA0 uhem). At AREA_SCALE = 9 the surface stands in
# for a young, rapidly rotating Sun, and flux-emergence theory predicts
# that tubes carrying more flux deflect poleward as they rise (Schussler &
# Solanki 1992, A&A 264, L13; Schussler et al. 1996, A&A 314, 503; DeLuca,
# Fan & Saar 1997, ApJ 481, 369): high-latitude emergence is unavoidable
# for G/K stars with P_rot of about 5-10 d or less, and larger spots
# emerge at higher latitudes. Set POLEWARD_COEF = 0 for the observed
# solar latitudes.
POLEWARD_COEF = 18.0
POLEWARD_AREA0 = 100.0
OUT = "spin.gif"
# ---------------------------------------------------------------------------

# Carrington rotation 2156: (NOAA AR, latitude deg, Carrington longitude deg,
# corrected whole-spot group area in millionths of the solar hemisphere), each
# group at its central-meridian crossing. RGO/USAF/NOAA record via Hathaway's
# active region database.
CR2156_GROUPS = [
    (12192, -12.0, 248.0, 2700.0),
    (12187,  -9.0, 320.0,  200.0),
    (12186, -21.0,  19.0,  180.0),
    (12182, -14.0, 123.0,  170.0),
    (12193,   4.0, 284.0,   80.0),
    (12195,   7.0, 184.0,   80.0),
    (12178,  -1.0, 158.0,   60.0),
    (12194, -12.0, 209.0,   50.0),
    (12197, -13.0, 161.0,   30.0),
    (12201,  -5.0,  84.0,   30.0),
    (12189,  21.0, 342.0,   20.0),
    (12199, -17.0, 126.0,   20.0),
    (12200, -16.0, 101.0,   20.0),
    (12185, -13.0,  62.0,   10.0),
    (12188,  17.0,  26.0,   10.0),
    (12190,  22.0, 300.0,   10.0),
    (12191, -11.0, 304.0,   10.0),
    (12196,  -3.0, 161.0,   10.0),
    (12198, -14.0, 193.0,   10.0),
    (12202,  13.0, 134.0,   10.0),
]


def solar_cr2156_spots(area_scale=AREA_SCALE):
    """Observed CR 2156 sunspot groups as (lat, lon, cap_radius_deg) tuples.

    A group of area A (millionths of the solar hemisphere) covers a fraction
    A * 1e-6 of the hemisphere, so a circular cap of the same area has
    half-angle theta = arccos(1 - A * 1e-6). area_scale multiplies A,
    preserving every group's relative size and position in longitude.

    Latitudes: each group keeps its observed emergence hemisphere and
    longitude, but is displaced poleward by POLEWARD_COEF degrees per
    decade of scaled area above POLEWARD_AREA0 uhem, following the
    flux-emergence results cited above: on a rapid rotator the largest
    groups surface at high latitude while small groups still emerge in
    the low-latitude belts, giving a distribution over latitude rather
    than a single band.

    Displayed south pole up: CR 2156's activity was almost entirely in the
    southern hemisphere, so this keeps the active hemisphere in view when
    the visible pole tips toward the observer at high inclination.
    """
    spots = []
    for _, lat, lon, area in CR2156_GROUPS:
        a = area * area_scale
        theta = np.rad2deg(np.arccos(1.0 - min(a * 1e-6, 1.0)))
        shift = min(POLEWARD_COEF * np.log10(max(a, POLEWARD_AREA0) / POLEWARD_AREA0),
                    45.0)
        lat_eff = np.sign(lat) * min(abs(lat) + shift, 80.0)
        spots.append((-lat_eff, lon, theta))
    return spots


def main():
    spots = solar_cr2156_spots()
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
