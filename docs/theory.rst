Model and background
====================

This page summarizes the forward model and the identifiability results from
the accompanying paper (Taaki, Corrales & Hero 2026, ApJ 1003, 226),
and maps the paper's notation onto the library API. Equation numbers refer
to the paper.

Surface representation
----------------------

The stellar surface brightness map is written in spherical coordinates as
:math:`s(\theta, \phi)` for polar angle :math:`\theta \in [0, \pi]` and
azimuth :math:`\phi \in [-\pi, \pi]`, and expanded in complex spherical
harmonics (Eqs. 1–2):

.. math::

   Y_l^m(\theta,\phi) = N_l^m P_l^m(\cos\theta)\, e^{im\phi}, \qquad
   s(\theta,\phi) = \sum_{l=0}^{L} \sum_{m=-l}^{l} s_l^m\, Y_l^m(\theta,\phi)

truncated at a maximum degree :math:`L` matched to the measurement noise
floor — higher degrees capture progressively finer spatial structure. The
coefficients are stored as a vector :math:`\mathbf{s} \in \mathbb{C}^{(L+1)^2}`
in the linear ordering :math:`(l, m) \rightarrow l^2 + l + m`
(:class:`jittermap.harmonics.indexing.SHIndexer`).

The complex basis is used for algebraic convenience when applying rotations,
but the physical surface is real, which is enforced by the conjugate
symmetry constraint

.. math::

   s_l^{-m} = (-1)^m (s_l^m)^*, \qquad s_l^0 \in \mathbb{R}

(:func:`jittermap.harmonics.indexing.project_real_surface`,
:func:`jittermap.harmonics.indexing.build_real_surface_transform`). This
halves the effective degrees of freedom and makes the modeled measurements
real.

Starspots are modeled analytically as spherical caps: an axisymmetric cap at
the pole has a closed-form expansion in Legendre integrals with only
:math:`m = 0` terms, which is rotated to any (latitude, longitude) with a
Wigner-d relation (:func:`jittermap.harmonics.spots.generate_spot`). Small
unresolved active regions can be emulated with a Gaussian random surface
whose per-degree variance follows a power law
(:func:`jittermap.harmonics.surfaces.random_surface`).

Rotation geometry
-----------------

The star's orientation is parameterized by Euler angles
:math:`R = (\alpha, \beta, \gamma)` in the :math:`z\!-\!y\!-\!z` convention
(paper Fig. 1): :math:`\beta \in [0, \pi/2]` is the inclination of the spin
axis — the observer faces the **equator** at :math:`\beta = 0` and the
**pole** at :math:`\beta = \pi/2` — while :math:`\gamma = \omega t` is the
time-dependent spin at rotation rate :math:`\omega = 2\pi / P`, and
:math:`\alpha` is the sky-plane tilt (taken to be zero). Rotations act on
the coefficients degree by degree through unitary Wigner-D matrices,
:math:`\mathbf{s}_l' = D^l(R)\, \mathbf{s}_l`, so surfaces are never rotated
pixel-wise.

In jittermap the two rotations are represented by the blocks

.. math::

   B(t) = D^l(\omega t + \tfrac{\pi}{2}, \tfrac{\pi}{2}, \tfrac{\pi}{2}),
   \qquad
   C(\beta) = D^l(-\beta + \tfrac{\pi}{2}, \tfrac{\pi}{2}, \tfrac{\pi}{2}).

Because the middle Euler angle is fixed at :math:`\pi/2`, both blocks are
phase-diagonal modulations of one fixed unitary matrix per degree,
:math:`M_l = D^l(\pi/2, \pi/2, \pi/2)`:

.. math::

   B(t) = \mathrm{diag}(e^{-im\omega t})\, M_l, \qquad
   C(\beta) = \mathrm{diag}(e^{+im\beta})\, M_l.

jittermap ships :math:`M_l` to :math:`L = 40` as exact numeric tables
(:func:`jittermap.forward.wigner.wigner_M`), so no symbolic algebra runs at
inference time.

Forward model
-------------

Astrometric measurements record the photocenter — the first moment of the
visible stellar disk, projected on the observer's x–y sky plane — while
photometry records the zeroth moment (disk-integrated flux). Applying the
first-moment operator to each rotated harmonic yields the astrometric
kernels :math:`k^h_{l,m}` for measurement axis :math:`h \in \{x, y\}`, and
the forward model becomes an inner product between the kernels and the
rotated coefficients (Eqs. 3–5):

.. math::

   \mu^h(R) = \sum_{l=0}^{L} \sum_{m=-l}^{l} \left[ D^l(R)\, \mathbf{s}_l \right]_m k^h_{l,m}.

For :math:`N` observation times the stacked x/y time series factor into a
time-dependent and an inclination-dependent matrix (Eqs. 8–9):

.. math::

   \boldsymbol{\mu} =
   \begin{bmatrix} \boldsymbol{\mu}^x \\ \boldsymbol{\mu}^y \end{bmatrix}
   = A(\beta)\,\mathbf{s}, \qquad
   A(\beta) = \begin{bmatrix} W_\omega & \\ & W_\omega \end{bmatrix}
   \begin{bmatrix} B_\beta^x \\ B_\beta^y \end{bmatrix},

where :math:`[W_\omega]_{n,m} = e^{-im\omega t_n}` depends only on the
sampling and :math:`B^h_\beta` folds the kernels and the inclination
rotation together. With at least :math:`2L+1` uniform samples over one
rotation period, :math:`W_\omega` is unitary (information-preserving) and
behaves as an inverse Fourier transform: the signal is a frequency comb at
harmonics :math:`m\omega`, :math:`m = -L..L`. The photometric channel has
the same structure with the photometric kernel swapped in.

API mapping:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Paper object
     - jittermap
   * - :math:`\mathbf{k}^x, \mathbf{k}^y` (astrometric kernels)
     - :func:`jittermap.forward.kernels.compute_A_lm`
   * - photometric kernel
     - :func:`jittermap.forward.kernels.compute_A_lm_photo`
   * - :math:`D^l` rotation blocks
     - :func:`jittermap.forward.wigner.rotation_funcs`
   * - :math:`A(\beta)` (time domain)
     - :meth:`jittermap.forward.design.ForwardModel.design`
   * - :math:`B^h_\beta` (Fourier domain)
     - :func:`jittermap.forward.fourier.fourier_design_matrix`
   * - :math:`W_\omega^H` comb projection
     - :func:`jittermap.forward.fourier.compress`
   * - :math:`\mathbf{y} = \boldsymbol{\mu} + \mathbf{n}`
     - :meth:`jittermap.forward.design.ForwardModel.observe`

Astrometry measures odd degrees; photometry even
------------------------------------------------

The key analytic result of the paper (Section 2.2): the astrometric kernels
are non-zero only for **odd** :math:`l` (plus :math:`l \le 2`), i.e.
harmonics anti-symmetric about the equator, whereas the photometric kernel
is non-zero only for **even** :math:`l` (plus :math:`l \le 2`), the
equator-symmetric harmonics (Cowan et al. 2013; Luger et al. 2021). In
order :math:`m`, the x-axis kernel is non-zero for odd :math:`m` and the
y-axis kernel for even :math:`m` (Eq. 10):

.. math::

   k^x_{l,m} &=
   \frac{I_{\phi,x}(m) N_l^m}{\pi} \int_{-1}^{1} (1-\eta^2)\, P_l^m(\eta)\, d\eta
   \quad \text{if } (l \text{ odd, } m \text{ odd}) \text{ or } (l=2, |m|=2),
   \\
   k^y_{l,m} &=
   \frac{I_{\phi,y}(m) N_l^m}{\pi} \int_{-1}^{1} \eta \sqrt{1-\eta^2}\, P_l^m(\eta)\, d\eta
   \quad \text{if } (l \text{ odd, } m \text{ even}) \text{ or } (l=2, |m|=1),

and zero otherwise. Intuitively, the photocenter integrates an
anti-symmetric position weighting over the visible disk, so only asymmetric
surface modes contribute. Astrometry and photometry therefore probe
orthogonal subspaces of the stellar surface, and **their joint use samples
every degree** — the basis for breaking light-curve inversion degeneracies.
jittermap evaluates the kernel integrals numerically (relative error
:math:`\sim 10^{-15}`), with independent SymPy and brute-force
cross-checks exercised in the test suite.

Inclination and the shape of the jitter signal
----------------------------------------------

The separable form isolates how inclination affects the information
content. Two consequences derived in the paper and reproducible with the
library (see the tutorials):

* For a pole-on observer (:math:`\beta = \pi/2`) the astrometric jitter of
  any surface is **circular**, and the photometric signal is constant —
  pole-on inclinations are the worst case for surface inversion.
* Starspots at higher latitudes produce smaller photocenter excursions, and
  spots out of view pin the photocenter at the origin between egress and
  ingress (paper Figs. 2–3).

Inference
---------

The surface is estimated by GMRF-regularized MAP inversion. The prior on
:math:`\mathbf{s}` is a diagonal Gaussian Markov random field in degree,
:math:`q_l = (l / \text{scale})^\alpha`, encoding that high-frequency
surface features have smaller amplitude
(:func:`jittermap.inference.inversion.gmrf_precision_diag`). The estimate
minimizes

.. math::

   \| D(\beta)\,\mathbf{s} - \mathbf{y} \|^2 + \lambda\, \mathbf{s}^H Q\, \mathbf{s}

over any subset of the channels (x, y, photometry) via augmented least
squares (:func:`jittermap.inference.inversion.solve_ridge`), optionally with
the real-surface constraint applied through the reparameterization
:math:`\mathbf{s} = R\mathbf{r}`
(:func:`jittermap.inference.inversion.solve_ridge_real_constraint`).

The inclination :math:`\beta` enters the design nonlinearly and is estimated
by minimizing the profile objective
:math:`J(\beta) = \|\mathbf{y} - D(\beta)\hat{\mathbf{s}}(\beta)\|^2 +
\lambda \hat{\mathbf{s}}^H Q \hat{\mathbf{s}}` with a bounded scalar search
(:func:`jittermap.inference.inclination.estimate_inclination`). Photometry
alone leaves an inclination ambiguity even at infinite SNR; adding
astrometry sharpens the profile objective and localizes spots in latitude
and longitude. The high-level entry point
:func:`jittermap.inference.reconstruct.reconstruct` combines both steps.

References
----------

* Taaki, J. S., Corrales, L., & Hero, A. O. III 2026, ApJ 1003, 226,
  "Using Astrometry to Break Degeneracies in Stellar Surface Mapping",
  doi:10.3847/1538-4357/ae66f7 (arXiv:2601.11737)
* Cowan, N. B., Fuentes, P. A., & Haggard, H. M. 2013, MNRAS 432, 2465 —
  photometric null space of light-curve inversion.
* Luger, R., Foreman-Mackey, D., & Hedges, C. 2021, AJ 162, 123 —
  degeneracies of rotational light curves.
* Morris, B. M., et al. 2018, AJ 156, 203 — starspot-induced photocenter
  jitter of nearby stars.
