"""Homogeneous medium vs the analytic 2D Green's function of a point (line) source."""

import math

import numpy as np
import pytest

from hifu_sim import PML, Grid, Medium, Simulation, cw_waveform, point_source


@pytest.fixture(scope="module")
def point_source_field():
    f, c, dx, a = 1e6, 1500.0, 1e-4, 1e3
    grid = Grid(200, 200, dx)
    sim = Simulation(grid, Medium.uniform(grid, c, 1000.0), f, pml=PML(20), backend="numpy")
    centre = (grid.x[100], grid.y[100])
    sim.add_source(point_source(grid, centre, a, cw_waveform(f, 3)))
    return grid, sim.run_cw(), (f, c, dx, a)


@pytest.mark.parametrize("direction", [(1, 0), (0, 1), (-1, 0), (1, 1)])
@pytest.mark.parametrize("r_cells", [30, 50, 70])
def test_cylindrical_spreading_amplitude(point_source_field, direction, r_cells):
    """Far field of a 2D monopole: |p(r)| = A dx sqrt(k / (2 pi r)).

    A single source cell injecting S = 2 c A / dx is a monopole of strength
    q = 2 c A dx, whose field is p = (k A dx / 2) H0(kr); the large-argument form of
    the Hankel function gives the expression above.
    """
    _, result, (f, c, dx, a) = point_source_field
    di, dj = direction
    steps = r_cells if di * dj == 0 else round(r_cells / math.sqrt(2))
    i, j = 100 + di * steps, 100 + dj * steps
    r = math.hypot(i - 100, j - 100) * dx
    k = 2 * math.pi * f / c
    expected = a * dx * math.sqrt(k / (2 * math.pi * r))
    assert result.amplitude[i, j] == pytest.approx(expected, rel=0.03)


def test_phase_advances_with_radius(point_source_field):
    _, result, (f, c, dx, _) = point_source_field
    k = 2 * math.pi * f / c
    dphi = (result.phase[150, 100] - result.phase[160, 100]) % (2 * math.pi)
    assert dphi == pytest.approx((k * 10 * dx) % (2 * math.pi), abs=0.05)


def test_field_is_isotropic(point_source_field):
    _, result, _ = point_source_field
    a = result.amplitude
    np.testing.assert_allclose(a[160, 100], a[100, 160], rtol=1e-3)
    np.testing.assert_allclose(a[160, 100], a[40, 100], rtol=2e-2)
