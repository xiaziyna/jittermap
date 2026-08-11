"""Joint vs single-channel surface reconstruction.

Generates a two-spot surface, simulates noiseless astrometric (x, y) and
photometric time series over one rotation, reconstructs the surface from
(i) all three channels, (ii) astrometry only, (iii) photometry only —
estimating the inclination in each case — and renders the comparison.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")

import jittermap as jm
from jittermap.plotting.panels import comparison_figure

# ---- settings -------------------------------------------------------------
L = 10               # spherical harmonic degree
N = 100              # samples over one rotation
INC_TRUE = 0.6       # inclination (radians)
SPOTS = [(30, 315, 18.0), (-15, 30, 13.0)]   # (lat, lon, radius) in degrees
# ---------------------------------------------------------------------------

times = np.linspace(0, 2 * np.pi, N, endpoint=False)
s_true = jm.multispot_surface(SPOTS, L, texture_amplitude=0.002, texture_seed=3)

fm = jm.ForwardModel(times, l_max=L)
y = fm.observe(s_true, inclination=INC_TRUE, channels="xyp", stacked=False)

results = {}
for label, channels in [("Joint", "xyp"), ("Astrometry", "xy"),
                        ("Photometry", "p")]:
    res = jm.reconstruct({c: y[c] for c in channels}, times, L,
                         channels=channels, lam=2e-4, model=fm)
    print(f"{label:11s} channels={channels:3s}  "
          f"estimated inclination = {res.inclination:.3f} (true {INC_TRUE})")
    results[label] = res

fig, _ = comparison_figure(s_true, results, l_true=L, inc_true=INC_TRUE)
fig.savefig("reconstruction_demo.png", dpi=200, bbox_inches="tight")
print("saved reconstruction_demo.png")
