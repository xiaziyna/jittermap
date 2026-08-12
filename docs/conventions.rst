Conventions
===========

All angle, rotation, indexing, and normalization conventions used
throughout jittermap, collected in one place. These match the accompanying
paper (Taaki, Corrales & Hero 2026) and are enforced by the test suite.

Coordinates and observer geometry
---------------------------------

* Spherical coordinates: polar angle :math:`\theta \in [0, \pi]` measured
  from the +z pole, azimuth :math:`\phi \in [-\pi, \pi]`.
* The observer lies along the **+x axis** of the spherical frame. The
  image (sky) plane is spanned by y (horizontal) and z (vertical);
  rendered maps and photocenter components (x, y) follow these axes.
* Latitude/longitude for spot placement: latitude in degrees, +90 at the
  north (+z) pole; longitude in degrees increasing eastward, with
  longitude 0 on the observer-facing meridian at spin phase t = 0 for an
  equator-on star.

Inclination and rotation
------------------------

* Euler angles :math:`R = (\alpha, \beta, \gamma)` in the
  **z–y–z convention**, applied right to left (paper Fig. 1).
* :math:`\beta \in [0, \pi/2]` is the **inclination of the spin axis**:
  :math:`\beta = 0` — the observer faces the **equator** (spin axis in the
  sky plane, vertical in the image); :math:`\beta = \pi/2` — the observer
  faces the **pole**. The projected spin axis is vertical with apparent
  length :math:`\cos\beta`.
* The spin phase is :math:`\gamma = \omega t` with rotation rate
  :math:`\omega = 2\pi/P`; the sky-plane tilt :math:`\alpha` is taken to
  be zero.
* Rotations act on coefficients degree by degree through Wigner-D matrices
  in the `sympy.physics.quantum.spin` convention,
  :math:`D^l_{mm'}(\alpha,\beta,\gamma) = e^{-im\alpha} d^l_{mm'}(\beta)
  e^{-im'\gamma}`, rows indexed by :math:`m = -l..l`:

  .. code-block:: text

     B(t)     = D^l(omega t + pi/2, pi/2, pi/2)     spin block
     C(beta)  = D^l(-beta + pi/2,   pi/2, pi/2)     inclination block
     design:    D_l(t, beta) = B(t) C(beta) a_l     (a_l = kernel vector)

* Point-space (grid) rotations are the **inverse** of coefficient-space
  rotations: the fast renderer uses :math:`R_z(-\omega t + \pi/2)` for the
  spin — note the sign relative to the coefficient block (regression-tested
  against the Wigner path in ``tests/test_plotting.py``).

Spherical harmonics and coefficient ordering
--------------------------------------------

* **Complex** spherical harmonics with the Condon–Shortley phase, matching
  ``scipy``:
  :math:`Y_l^m(\theta,\phi) = N_l^m P_l^m(\cos\theta) e^{im\phi}` with
  :math:`N_l^m = \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}`
  (orthonormal on the sphere).
* Flat coefficient ordering: :math:`(l, m) \rightarrow l^2 + l + m`, so
  degree l occupies the contiguous block :math:`[l^2, (l+1)^2)`
  (:class:`jittermap.harmonics.indexing.SHIndexer`).
* Real surfaces satisfy :math:`s_l^{-m} = (-1)^m (s_l^m)^*` with
  :math:`s_l^0` real. Surface generators return coefficients obeying this
  symmetry; :func:`jittermap.harmonics.indexing.project_real_surface`
  restores it after unconstrained fits.
* The monopole :math:`s_0^0` is an unobservable overall offset for
  astrometry and a pure flux normalization for photometry; surface
  generators drop it by default.

Units and normalization
-----------------------

* Times are in radians of rotation phase when :math:`\omega = 1`
  (the default); pass physical times together with
  :math:`\omega = 2\pi/P` otherwise.
* Photocenter outputs are normalized to the stellar angular radius for a
  unit-contrast surface. To convert to physical microarcseconds, multiply
  by :math:`(1/\pi)\,\Theta_\star\, c` where
  :math:`\Theta_\star = 4650\,(R_\star/R_\odot)/(d/\mathrm{pc})\;\mu\mathrm{as}`
  is the stellar angular radius and :math:`c` the spot contrast (the
  :math:`1/\pi` carries the visibility-kernel normalization of the paper's
  integrals).
* Spot ``contrast=1`` means fully dark relative to the photosphere;
  spot radii convert between fractions of the stellar radius and angular
  cap radius via :func:`jittermap.harmonics.surfaces.rfrac_to_deg`.

Selection rules (for quick reference)
-------------------------------------

* Astrometric kernels: non-zero only for odd :math:`l` and :math:`l \le 2`;
  the x-kernel needs odd :math:`m`, the y-kernel even :math:`m`.
* Photometric kernel: non-zero only for even :math:`l` and :math:`l \le 2`.
* Consequence: photometry is blind to odd :math:`l \ge 3`, astrometry to
  even :math:`l \ge 4`; jointly all degrees are sampled.
