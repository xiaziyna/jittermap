Example scripts
===============

The ``examples/`` directory contains self-contained scripts (edit the
settings block at the top and re-run). They mirror the workflow of the
tutorial notebooks in plain-script form.

Joint vs. single-channel reconstruction
---------------------------------------

``examples/reconstruction_demo.py`` builds a two-spot surface, simulates
noiseless astrometric and photometric time series over one rotation, and
reconstructs the surface from all three channels, from astrometry alone,
and from photometry alone — estimating the inclination in each case:

.. code-block:: text

    Joint       channels=xyp  estimated inclination = 0.588 (true 0.6)
    Astrometry  channels=xy   estimated inclination = 0.484 (true 0.6)
    Photometry  channels=p    estimated inclination = 0.426 (true 0.6)

.. image:: _images_repo/reconstruction_demo.png
   :width: 100%
   :alt: Truth vs joint, astrometry-only and photometry-only reconstructions

The joint fit recovers both spots and the inclination; astrometry alone is
diffuser with a biased inclination; photometry alone shows the light-curve
inversion degeneracy (smearing and ghost structure).

Forward-model signals
---------------------

``examples/forward_demo.py`` plots the x/y photocenter shifts and the flux
variation of a single starspot at several inclinations, reproducing the
phenomenology of the paper's Figures 2–3 (circularized jitter and vanishing
photometric modulation toward pole-on viewing).

Spin animation
--------------

``examples/spin_animation_demo.py`` renders an animated rotation of a
multi-spot surface (``spin.gif``) with
:func:`jittermap.plotting.animate.animate_spin`.
