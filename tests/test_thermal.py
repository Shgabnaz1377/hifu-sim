"""CEM43 thermal dose and the Pennes bioheat solver vs closed-form solutions."""

import math

import numpy as np
import pytest

from hifu_sim import BioheatSolver, Grid, Medium, absorbed_power_density, cem43, cem43_rate
from hifu_sim.thermal import BLOOD_SPECIFIC_HEAT


@pytest.mark.parametrize(
    ("temp", "expected"),
    [(43.0, 1.0), (44.0, 2.0), (46.0, 8.0), (42.0, 0.25), (41.0, 0.0625), (37.0, 0.25**6)],
)
def test_cem43_one_minute_at_constant_temperature(temp, expected):
    history = np.full(600, temp)  # 60 s sampled every 0.1 s
    assert float(cem43(history, dt=0.1)) == pytest.approx(expected, rel=1e-12)


def test_cem43_is_additive_and_vectorised():
    hist = np.concatenate([np.full(60, 44.0), np.full(120, 50.0)])  # 1 min @44, 2 min @50
    assert float(cem43(hist, 1.0)) == pytest.approx(2.0 + 2 * 2**7)
    stacked = np.stack([hist, np.full(180, 43.0)], axis=1)
    np.testing.assert_allclose(cem43(stacked, 1.0), [2.0 + 2 * 2**7, 3.0])


def test_cem43_rate_switches_base_at_43():
    np.testing.assert_allclose(cem43_rate([42.0, 43.0, 45.0]), [0.25, 1.0, 4.0])


def test_absorbed_power():
    assert float(absorbed_power_density(np.array(5.0), np.array(1e4))) == pytest.approx(1e5)


def _uniform_tissue(n=12, perfusion=0.0):
    grid = Grid(n, n, 1e-3)
    medium = Medium.homogeneous(grid, "soft_tissue")
    medium.perfusion[...] = perfusion
    return grid, medium


def test_pennes_uniform_heating_without_losses():
    grid, medium = _uniform_tissue()
    q = 2e5
    res = BioheatSolver(grid, medium).simulate(
        np.full(grid.shape, q), heating_time=10.0, dt=0.05, conduction=False
    )
    rho_c = medium.density[0, 0] * medium.specific_heat[0, 0]
    assert res.temperature_end_of_heating[5, 5] == pytest.approx(37.0 + q * 10.0 / rho_c)


def test_pennes_perfusion_exponential_approach():
    """dT/dt = (Q - w c_b (T - Ta)) / (rho C)  =>  T - Ta = Q/(w c_b) (1 - e^{-t/tau})."""
    grid, medium = _uniform_tissue(perfusion=2.0)
    q, dt, t_heat = 5e5, 0.01, 60.0
    res = BioheatSolver(grid, medium).simulate(
        np.full(grid.shape, q), heating_time=t_heat, dt=dt, conduction=False
    )
    w = 2.0 * BLOOD_SPECIFIC_HEAT
    rho_c = medium.density[0, 0] * medium.specific_heat[0, 0]
    expected = q / w * (1 - math.exp(-w * t_heat / rho_c))
    assert res.temperature_end_of_heating[3, 3] - 37.0 == pytest.approx(expected, rel=1e-3)


def test_heat_conduction_conserves_energy_and_spreads_as_diffusion():
    """A Gaussian hot spot keeps its heat and its variance grows as s0^2 + 2 kappa t."""
    grid = Grid(81, 81, 2e-4)
    medium = Medium.homogeneous(grid, "soft_tissue")
    medium.perfusion[...] = 0.0
    solver = BioheatSolver(grid, medium)
    xx, yy = grid.mesh()
    cx = cy = grid.x[40]
    s0 = 4 * grid.dx
    dt = solver.stable_dt()
    q = 1e8 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s0**2))
    t_cool = 1.0
    res = solver.simulate(q, heating_time=dt, cooling_time=t_cool, dt=dt)

    rho_c = medium.density[0, 0] * medium.specific_heat[0, 0]
    kappa = medium.thermal_conductivity[0, 0] / rho_c
    rise = res.temperature_final - 37.0
    deposited = q.sum() * res.heating_time * grid.dx**2
    stored = rho_c * rise.sum() * grid.dx**2
    assert stored == pytest.approx(deposited, rel=1e-2)

    var = (rise * (xx - cx) ** 2).sum() / rise.sum()
    elapsed = res.times[-1] - 0.5 * res.heating_time
    assert var == pytest.approx(s0**2 + 2 * kappa * elapsed, rel=3e-2)


def test_unstable_thermal_step_rejected():
    grid, medium = _uniform_tissue()
    solver = BioheatSolver(grid, medium)
    with pytest.raises(ValueError, match="stability"):
        solver.simulate(np.zeros(grid.shape), 1.0, dt=10 * solver.stable_dt())
