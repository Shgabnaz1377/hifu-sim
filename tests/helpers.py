"""Shared set-ups for the validation tests (small grids, fast)."""

from __future__ import annotations

import math

import numpy as np

from hifu_sim import PML, Grid, Medium, Simulation, cw_waveform, plane_source


def lock_in(trace: np.ndarray, t: np.ndarray, frequency: float, periods: int) -> np.ndarray:
    """Complex amplitude at ``frequency`` from the last ``periods`` whole periods."""
    dt = t[1] - t[0]
    n = round(periods / (frequency * dt))
    w = np.exp(-2j * math.pi * frequency * t[-n:])
    result: np.ndarray = 2.0 / n * (trace[..., -n:] @ w)
    return result


def plane_wave_run(
    c: float,
    rho: float = 1000.0,
    alpha: float = 0.0,
    frequency: float = 1e6,
    dx: float = 1e-4,
    receivers_mm: tuple[float, ...] = (15.0, 35.0),
    amplitude: float = 1e3,
) -> tuple[np.ndarray, float, float]:
    """1D-like CW plane wave: 2 lateral cells between rigid walls, PML only along x.

    Returns the complex lock-in amplitudes at the receivers, the time step and the
    effective Courant number.
    """
    grid = Grid(nx=500, ny=2, dx=dx)
    medium = Medium.uniform(grid, c, rho, alpha)
    sim = Simulation(grid, medium, frequency, pml=PML(40, thickness_y=0), cfl=0.3,
                     backend="numpy")
    sim.add_source(plane_source(grid, 5e-3, amplitude, cw_waveform(frequency, 3)))
    idx = [np.ravel_multi_index(grid.index_of(r * 1e-3, 0.0), grid.shape) for r in receivers_mm]
    t_end = 3 / frequency + 1.3 * grid.size[0] / c + 6 / frequency
    t, traces = sim.record(math.ceil(t_end / sim.dt), idx)
    return lock_in(traces, t, frequency, 4), sim.dt, c * sim.dt / dx
