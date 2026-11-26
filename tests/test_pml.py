"""PML absorption: energy leaves a PML-bounded box, but is conserved in a rigid box."""

import numpy as np
import pytest

from hifu_sim import PML, Grid, Medium, Simulation


def _pulse_sim(pml: PML) -> Simulation:
    grid = Grid(160, 160, 1e-4)
    medium = Medium.uniform(grid, 1500.0, 1000.0)
    sim = Simulation(grid, medium, 1e6, pml=pml, cfl=0.3, backend="numpy")
    xx, yy = grid.mesh()
    cx, cy = grid.size[0] / 2, grid.size[1] / 2
    sigma = 3 * grid.dx
    sim.set_initial_pressure(1e3 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))
    return sim


def _steps(sim: Simulation, t: float) -> int:
    return round(t / sim.dt)


def test_energy_conserved_before_wave_reaches_boundary():
    sim = _pulse_sim(PML(20))
    sim.step()
    e0 = sim.energy()
    sim.run(_steps(sim, 2.5e-6))  # front travels ~3.8 mm < 6 mm to the PML
    assert sim.energy() == pytest.approx(e0, rel=1e-2)


def test_energy_decays_in_pml_box():
    sim = _pulse_sim(PML(20))
    sim.step()
    e0 = sim.energy()
    sim.run(_steps(sim, 40e-6))  # ~ 2.5 domain diagonals
    assert sim.energy() / e0 < 1e-3


def test_energy_trapped_in_rigid_box():
    sim = _pulse_sim(PML(0))
    sim.step()
    e0 = sim.energy()
    sim.run(_steps(sim, 40e-6))
    assert sim.energy() == pytest.approx(e0, rel=2e-2)


def test_pml_profile_shape():
    pml = PML(10, order=2, reflection=1e-4)
    sig = pml.profile(50, 10, 1e-4, 1500.0, staggered=False)
    assert sig.shape == (50,)
    assert np.all(sig[10:40] == 0)
    assert sig[0] > sig[5] > 0
    assert sig[0] == pytest.approx(sig[-1])
    assert pml.profile(50, 10, 1e-4, 1500.0, staggered=True).shape == (51,)


def test_pml_must_fit():
    grid = Grid(30, 30, 1e-4)
    with pytest.raises(ValueError):
        Simulation(grid, Medium.uniform(grid, 1500, 1000), 1e6, pml=PML(15))
