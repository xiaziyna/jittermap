"""Line overlays for rendered stars: spin axis and rotating meridians.

The MeridianOverlay draws the projected spin axis and a set of
longitudinal great circles that co-rotate with the stellar surface,
making the rotation state and axis tilt legible in animations.
"""

from typing import List

import numpy as np
import matplotlib.pyplot as plt


class MeridianOverlay:
    """Spin axis + rotating meridians atop the hemisphere projection.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes displaying the rendered star (image coords in [-1, 1]).
    inclination : float
        Stellar inclination beta (radians).
    omega : float
        Rotation rate.
    num_meridians : int
        Number of longitude lines; the central one is highlighted.
    """

    def __init__(self, ax, inclination, omega=1.0, num_meridians=5,
                 axis_color="w", line_color="#dddddd",
                 highlight_color="#ffd27f"):
        self.ax = ax
        self.inclination = inclination
        self.omega = omega
        self.num_meridians = num_meridians
        self.highlight_color = highlight_color
        self.line_color = line_color
        self.theta_range = np.linspace(0.0, np.pi, 80)
        rot_axis = np.array([0.0, np.sin(inclination), np.cos(inclination)])
        (self.axis_line,) = ax.plot([0.0, rot_axis[0]], [0.0, rot_axis[2]],
                                    color=axis_color, lw=2)
        self.meridian_lines: List[plt.Line2D] = []

    def update(self, t):
        """Redraw the meridians for spin phase omega * t; returns the
        artists (for FuncAnimation blitting)."""
        for line in self.meridian_lines:
            line.remove()
        self.meridian_lines.clear()

        phi_offsets = np.linspace(-np.pi / 2, np.pi / 2, self.num_meridians)
        alpha = np.pi / 2
        beta = self.inclination
        gamma = -self.omega * t

        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)
        Rz1 = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
        Rz2 = np.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]])
        R = Rz1 @ Ry @ Rz2

        for idx, phi_offset in enumerate(phi_offsets):
            x0 = np.sin(self.theta_range) * np.cos(phi_offset)
            y0 = np.sin(self.theta_range) * np.sin(phi_offset)
            z0 = np.cos(self.theta_range)
            rotated = R @ np.vstack([x0, y0, z0])
            x_rot, y_rot, z_rot = rotated
            visible = y_rot >= 0.0
            if np.any(visible):
                is_mid = idx == len(phi_offsets) // 2
                color = self.highlight_color if is_mid else self.line_color
                alpha_val = 0.9 if is_mid else 0.4
                (line,) = self.ax.plot(x_rot[visible], z_rot[visible],
                                       color=color, alpha=alpha_val,
                                       linewidth=1.6)
                self.meridian_lines.append(line)

        return [self.axis_line, *self.meridian_lines]
