"""Benchmark one leapfrog step: naive Python loops vs NumPy vs Numba.

Usage::

    python benchmarks/bench_kernels.py            # default sizes
    python benchmarks/bench_kernels.py --sizes 64 128 --repeat 3

Prints a Markdown table (time per step and million cell-updates per second) that can
be pasted into the README. The Numba kernel is compiled before timing starts.
"""

from __future__ import annotations

import argparse
import os
import platform
import time

import numpy as np

from hifu_sim import PML, Grid, Medium, Simulation
from hifu_sim.kernels import NUMBA_AVAILABLE, numba_threads

NAIVE_MAX_CELLS = 128 * 128  # pure-Python loops get painfully slow beyond this


def make_sim(n: int, backend: str) -> Simulation:
    grid = Grid(n, n, 1e-4)
    rng = np.random.default_rng(0)
    medium = Medium.uniform(grid, 1500.0, 1000.0, alpha=5.0)
    medium.sound_speed += rng.uniform(0, 100, grid.shape)
    sim = Simulation(grid, medium, 1e6, pml=PML(min(20, n // 8)), backend=backend)
    sim.set_initial_pressure(rng.standard_normal(grid.shape))
    return sim


def time_per_step(n: int, backend: str, steps: int, repeat: int) -> float:
    sim = make_sim(n, backend)
    sim.run(2)  # warm-up (and JIT compilation for numba)
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        sim.run(steps)
        best = min(best, (time.perf_counter() - t0) / steps)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    backends = ["naive", "numpy"] + (["numba"] if NUMBA_AVAILABLE else [])
    print(f"python {platform.python_version()}, numpy {np.__version__}, "
          f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs, "
          f"numba threads: {numba_threads() if NUMBA_AVAILABLE else 'n/a'}\n")
    header = "| grid | " + " | ".join(f"{b} [ms/step]" for b in backends)
    header += " | numpy vs naive | numba vs numpy |"
    print(header)
    print("|" + "---|" * (len(backends) + 3))
    for n in args.sizes:
        cells = n * n
        times: dict[str, float] = {}
        for b in backends:
            if b == "naive" and cells > NAIVE_MAX_CELLS:
                continue
            steps = 3 if b == "naive" else max(20, int(2e7 / cells))
            times[b] = time_per_step(n, b, steps, args.repeat if b != "naive" else 1)
        cols = [f"{times[b] * 1e3:.3g}" if b in times else "-" for b in backends]
        speed_np = f"{times['naive'] / times['numpy']:.0f}x" if "naive" in times else "-"
        speed_nb = f"{times['numpy'] / times['numba']:.1f}x" if "numba" in times else "-"
        print(f"| {n}x{n} | " + " | ".join(cols) + f" | {speed_np} | {speed_nb} |")


if __name__ == "__main__":
    main()
