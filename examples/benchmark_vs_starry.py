"""Benchmark: photometric design-matrix assembly, jittermap vs starry.

Reproduces the comparison table in the README. jittermap timings run in
any environment with this package installed; the starry timings need an
environment with starry 1.2 (legacy theano stack — on modern numpy the
`np.bool = np.bool_` compatibility patch below is required).

Run each half in its own environment:
    python benchmark_vs_starry.py jittermap
    python benchmark_vs_starry.py starry
"""

import sys
import time

import numpy as np

N = 100
DEGREES = [10, 20, 30, 40]
REPEATS = 5


def bench_jittermap():
    import jittermap as jm
    times = np.linspace(0, 2 * np.pi, N, endpoint=False)
    for L in DEGREES:
        fm = jm.ForwardModel(times, l_max=L)   # one-time precompute
        fm.design_channel(0.6, "p")            # warm-up (cache load)
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            D = fm.design_channel(0.6, "p")
        dt = (time.perf_counter() - t0) / REPEATS
        print(f"jittermap L={L}: warm photometry design {dt*1e3:6.1f} ms, "
              f"shape {D.shape}")


def bench_starry():
    np.bool = np.bool_   # numpy >= 1.24 compatibility for theano
    import starry
    starry.config.lazy = False
    starry.config.quiet = True
    theta = np.linspace(0, 360, N)
    for ydeg in DEGREES:
        t0 = time.perf_counter()
        m = starry.Map(ydeg=ydeg)
        m.design_matrix(theta=theta)           # includes theano compile
        cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            X = m.design_matrix(theta=theta)
        dt = (time.perf_counter() - t0) / REPEATS
        print(f"starry ydeg={ydeg}: cold {cold:6.2f} s, "
              f"warm design {dt*1e3:6.1f} ms, shape {X.shape}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "jittermap"
    bench_starry() if which == "starry" else bench_jittermap()
