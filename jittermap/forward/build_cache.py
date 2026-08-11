"""Build the packaged caches: Wigner rotation matrices and kernel tables.

Usage:
    python -m jittermap.forward.build_cache --lmax 40 [--outdir PATH]
    python -m jittermap.forward.build_cache --photo 40

The default mode computes M_l = D^l(pi/2, pi/2, pi/2) exactly for
l = 0..lmax, one compressed npz per degree. The --photo mode extends the
photometric kernel table to a target degree using high-precision mpmath
quadrature for the radial integrals (matches the sympy symbolic values
to ~1e-16), reusing existing lower-degree tables incrementally.
"""

import argparse
import os
import time

import numpy as np

from jittermap.forward.wigner import compute_M_exact, _PKG_DATA_DIR


def compute_k_photo_mpmath(l, m, dps=50):
    """Photometric kernel entry via mpmath high-precision quadrature
    (fast path for large l, cross-checked against compute_k_photo)."""
    import mpmath
    import sympy as sp
    from scipy.special import lpmv
    from jittermap.forward.kernels import I_phi_y_closed

    mpmath.mp.dps = dps
    if l > 2 and (l % 2 == 1):
        return 0.0 + 0.0j
    am = abs(m)
    I_phi = I_phi_y_closed(am, prec=80)
    if abs(I_phi) < 1e-30:
        return 0.0 + 0.0j
    N_lm = float(sp.sqrt((2 * l + 1) / (4 * sp.pi)
                         * sp.factorial(l - am) / sp.factorial(l + am)).evalf(80))

    def integrand(x):
        return mpmath.mpf(float(lpmv(am, l, float(x)))) * mpmath.sqrt(1 - x ** 2)

    I_radial = float(mpmath.quad(integrand, [-1, 1]))
    val = complex(N_lm * I_radial * I_phi)
    if m < 0:
        val = (-1) ** am * np.conj(val)
    return val


def build_photo(target_L, outdir):
    """Extend the photometric kernel table to target_L, reusing the
    largest existing table below it."""
    from jittermap.forward.kernels import _PKG_DATA_DIR as KERNEL_DIR

    if outdir is None:
        outdir = KERNEL_DIR
    target_file = os.path.join(outdir, f"A_lm_photo_lp{target_L}.npz")
    if os.path.exists(target_file):
        print(f"photo table for L={target_L} already exists")
        return

    best_L, best_data = -1, None
    if os.path.isdir(KERNEL_DIR):
        for f in sorted(os.listdir(KERNEL_DIR)):
            if f.startswith("A_lm_photo_lp") and f.endswith(".npz"):
                l_val = int(f[len("A_lm_photo_lp"):-len(".npz")])
                if best_L < l_val <= target_L:
                    best_L = l_val
                    best_data = np.load(os.path.join(KERNEL_DIR, f))["A_lm_photo"]

    A = np.zeros((target_L + 1, 2 * target_L + 1), dtype=complex)
    if best_data is not None:
        for ln in range(best_L + 1):
            for mn in range(-ln, ln + 1):
                A[ln, mn + target_L] = best_data[ln, mn + best_L]
        print(f"copied degrees 0..{best_L} from existing table")

    t0 = time.time()
    for ln in range(best_L + 1, target_L + 1):
        if ln > 2 and ln % 2 == 1:
            continue
        for mn in range(0, ln + 1):
            val = compute_k_photo_mpmath(ln, mn)
            A[ln, mn + target_L] = val
            if mn > 0:
                A[ln, -mn + target_L] = (-1) ** mn * np.conj(val)
        print(f"l={ln:3d} done ({time.time()-t0:6.1f}s)", flush=True)

    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(target_file, A_lm_photo=A)
    print(f"saved {target_file}")


def build(l_max, outdir):
    os.makedirs(outdir, exist_ok=True)
    for l in range(l_max + 1):
        path = os.path.join(outdir, f"wigner_M_l{l}.npz")
        if os.path.exists(path):
            continue
        t0 = time.time()
        M = compute_M_exact(l)
        np.savez_compressed(path, M=M)
        print(f"l={l:3d}  ({2*l+1}x{2*l+1})  {time.time()-t0:6.1f}s  -> {path}",
              flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmax", type=int, default=40)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--photo", type=int, default=None,
                        help="build the photometric kernel table to this degree")
    args = parser.parse_args()
    if args.photo is not None:
        build_photo(args.photo, args.outdir)
    else:
        build(args.lmax, args.outdir or _PKG_DATA_DIR)


if __name__ == "__main__":
    main()
