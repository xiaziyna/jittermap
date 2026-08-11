"""Animate one rotation of a spotted surface."""

import matplotlib
matplotlib.use("Agg")

import jittermap as jm
from jittermap.plotting.animate import animate_spin

L = 15
SPOTS = [(30, 150, 12.0), (55, 40, 8.0), (-10, 260, 10.0)]

s = jm.multispot_surface(SPOTS, L, texture_amplitude=0.005, texture_seed=1)
animate_spin(s, L, inclination=0.6, n_frames=60, n_grid=200,
             output_path="spin.gif")
print("saved spin.gif")
