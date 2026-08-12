jittermap
=========

**Stellar surface mapping from astrometric jitter and photometry.**

.. image:: _images_repo/spin.gif
   :width: 680
   :align: center
   :alt: A Monte Carlo spotted star rotating at three labeled inclinations

Astrometric jitter arises when starspots on a rotating stellar surface move
in and out of view, shifting the observed photocenter. This jitter is a
limiting noise source for detecting small exoplanets around active stars —
but it also encodes the stellar surface. jittermap is the reference
implementation of the methods developed in

    Taaki, J. S., Corrales, L., & Hero, A. O. III (2026),
    *"Using Astrometry to Break Degeneracies in Stellar Surface Mapping"*,
    The Astrophysical Journal. `arXiv:2601.11737
    <https://arxiv.org/abs/2601.11737>`_,
    doi:10.48550/arXiv.2601.11737

which derives a linear spherical-harmonic forward model for the rotational
astrometric signal of an arbitrary stellar surface at any inclination, and
shows that astrometry and photometry probe complementary surface
information: photometry measures even-degree harmonics (symmetric about the
equator) while astrometry measures odd-degree harmonics. Their joint use
breaks long-standing degeneracies of light-curve inversion and helps
disentangle spot jitter from true reflex motion.

The library provides:

* the **forward model** — analytic astrometric/photometric moment kernels,
  Wigner-D rotations (precomputed numerically to :math:`L=40`), fast
  design-matrix assembly, and lossless Fourier-domain compression;
* the **inverse problem** — GMRF-regularized MAP surface reconstruction
  from any subset of the x/y/photometry channels, with profile-objective
  estimation of the stellar inclination;
* **surface generation** — analytic cap starspots and GMRF random textures;
* **plotting** — hemisphere rendering, comparison panels, spin animations.

**If you use jittermap, any part of its code, or results produced with it
in published work, please cite the paper above.**

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   theory
   examples

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/01_surfaces
   tutorials/02_forward_model
   tutorials/03_kernels_complementarity
   tutorials/04_reconstruction
   tutorials/05_inclination_and_fourier

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
