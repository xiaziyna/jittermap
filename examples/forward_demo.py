"""Forward model demo: signals of a single starspot.

Plots the astrometric photocenter shift (x, y) and photometric flux
variation over one rotation for a single spot, at several inclinations.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jittermap as jm

L = 10
N = 200
SPOT = (35, 90, 12.0)   # lat, lon, radius (deg)

times = np.linspace(0, 2 * np.pi, N, endpoint=False)
s = jm.multispot_surface([SPOT], L)
fm = jm.ForwardModel(times, l_max=L)

fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
for inc in [0.2, 0.6, 1.0, np.pi / 2]:
    y = fm.observe(s, inclination=inc, channels="xyp", stacked=False)
    for ax, c, name in zip(axes, "xyp",
                           ["photocenter x", "photocenter y", "flux"]):
        ax.plot(times, y[c], label=f"$\\beta$={inc:.2f}")
        ax.set_ylabel(name)
axes[0].legend(ncol=4, fontsize=8)
axes[-1].set_xlabel("rotation phase")
fig.suptitle("Single-spot astrometric and photometric signals")
fig.tight_layout()
fig.savefig("forward_demo.png", dpi=200)
print("saved forward_demo.png")
