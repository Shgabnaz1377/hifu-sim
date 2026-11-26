"""Time-step selection and the stability bound of the leapfrog scheme."""

import math

import numpy as np
import pytest

from hifu_sim import PML, Grid, Medium, Simulation, cfl_limit, cfl_time_step


def test_cfl_limit_values():
    assert cfl_limit(1e-4) == pytest.approx(1 / math.sqrt(2))
    # anisotropic cells: c dt sqrt(1/dx^2 + 1/dy^2) <= 1, expressed relative to dx
    assert cfl_limit(1e-4, 2e-4) == pytest.approx(1 / math.sqrt(1 + 0.25))


def test_time_step_matches_courant_number():
    dt = cfl_time_step(1500.0, 1e-4, cfl=0.3)
    assert 1500.0 * dt / 1e-4 == pytest.approx(0.3)


def test_time_step_snaps_to_integer_steps_per_period():
    f = 1e6
    dt = cfl_time_step(1540.0, 1e-4, cfl=0.3, frequency=f)
    steps = 1.0 / (f * dt)
    assert steps == pytest.approx(round(steps), abs=1e-9)
    assert 1540.0 * dt / 1e-4 <= 0.3 + 1e-12  # never exceeds the requested Courant number


@pytest.mark.parametrize("cfl", [0.0, 0.71, 1.0, -0.1])
def test_invalid_cfl_rejected(cfl):
    with pytest.raises(ValueError):
        cfl_time_step(1500.0, 1e-4, cfl=cfl)


def test_simulation_rejects_unstable_dt():
    grid = Grid(32, 32, 1e-4)
    medium = Medium.uniform(grid, 1500.0, 1000.0)
    dt_bad = 0.75 * 1e-4 / 1500.0
    with pytest.raises(ValueError, match="stability"):
        Simulation(grid, medium, 1e6, pml=PML(0), dt=dt_bad)


def test_cfl_uses_fastest_sound_speed():
    grid = Grid(32, 32, 1e-4)
    medium = Medium.uniform(grid, 1500.0, 1000.0)
    medium.sound_speed[10, 10] = 3000.0
    sim = Simulation(grid, medium, 1e6, pml=PML(0), cfl=0.3)
    assert 3000.0 * sim.dt / 1e-4 <= 0.3 + 1e-12


def _growth(courant_fraction: float, steps: int = 400) -> float:
    """Max |p| after ``steps`` relative to the start, rigid lossless box, noisy start."""
    grid = Grid(48, 48, 1e-4)
    medium = Medium.uniform(grid, 1500.0, 1000.0)
    dt = courant_fraction * cfl_limit(grid.dx) * grid.dx / 1500.0
    sim = Simulation(grid, medium, 1e6, pml=PML(0), dt=min(dt, 0.999 * cfl_limit(grid.dx)
                                                           * grid.dx / 1500.0))
    if courant_fraction >= 1.0:
        # bypass the guard on purpose to show that the bound is sharp
        sim.dt = dt
        sim.coefficients = sim._build_coefficients()
    p0 = np.random.default_rng(0).standard_normal(grid.shape)
    sim.set_initial_pressure(p0)
    sim.run(steps)
    return float(np.abs(sim.pressure).max() / np.abs(p0).max())


def test_stability_bound_is_sharp():
    assert _growth(0.98) < 5.0
    assert _growth(1.03) > 1e6
