jittermap
=========

Stellar surface mapping from astrometric jitter and photometry.

jittermap implements a linear spherical-harmonic forward model for the
astrometric photocenter shift and disk-integrated photometry of a rotating
spotted star, and the corresponding inverse problem: GMRF-regularized MAP
reconstruction of the surface together with estimation of the stellar
inclination. Astrometry measures odd-degree surface modes and photometry
even-degree modes, so their joint use breaks degeneracies inherent to
either channel alone.

See the README for installation and a quickstart, and the ``examples/``
directory for worked scripts.

.. toctree::
   :maxdepth: 2

   api

Reference
---------

If you use jittermap, please cite Taaki, Corrales & Hero (2026),
"Using Astrometry to Break Degeneracies in Stellar Surface Mapping",
ApJ, arXiv:2601.11737.
