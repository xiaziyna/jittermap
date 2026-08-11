"""Compatibility shims across scipy versions."""

try:  # scipy < 1.15
    from scipy.special import sph_harm as _sph_harm_legacy

    def sph_harm(m, l, azimuth, polar):
        """Complex spherical harmonic Y_l^m in the legacy scipy argument
        convention: sph_harm(m, l, azimuthal angle, polar angle)."""
        return _sph_harm_legacy(m, l, azimuth, polar)

except ImportError:  # scipy >= 1.15: sph_harm removed in favor of sph_harm_y
    from scipy.special import sph_harm_y as _sph_harm_y

    def sph_harm(m, l, azimuth, polar):
        """Complex spherical harmonic Y_l^m in the legacy scipy argument
        convention: sph_harm(m, l, azimuthal angle, polar angle)."""
        return _sph_harm_y(l, m, polar, azimuth)
