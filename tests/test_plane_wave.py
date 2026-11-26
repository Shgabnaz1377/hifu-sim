"""Plane waves: phase speed, absorption and interface reflection vs theory."""

import math

import numpy as np
import pytest

from hifu_sim import PML, Grid, Medium, Simulation, gaussian_pulse, plane_source

from .helpers import plane_wave_run

DX_MM = 20.0e-3  # distance between the two receivers [m]


def _numerical_phase_speed(c, f, dt, dx):
    """Phase speed predicted by the discrete dispersion relation of the 1D scheme."""
    courant = c * dt / dx
    k = 2 / dx * math.asin(math.sin(math.pi * f * dt) / courant)
    return 2 * math.pi * f / k


@pytest.mark.parametrize("c", [1500.0, 2500.0])
def test_plane_wave_phase_speed(c):
    f = 1e6
    amps, dt, _ = plane_wave_run(c, frequency=f)
    phi = float(np.angle(amps[0] / amps[1]))  # = k * dx  (mod 2 pi)
    n = round((2 * math.pi * f / c * DX_MM - phi) / (2 * math.pi))
    c_measured = 2 * math.pi * f * DX_MM / (phi + 2 * math.pi * n)
    assert c_measured == pytest.approx(c, rel=1e-2)
    # and it is exactly the speed the discrete dispersion relation predicts
    assert c_measured == pytest.approx(_numerical_phase_speed(c, f, dt, 1e-4), rel=5e-4)


def test_plane_wave_amplitude_calibration():
    """A row of source cells with amplitude A radiates a plane wave of amplitude A."""
    amps, _, _ = plane_wave_run(1500.0, amplitude=1e3)
    np.testing.assert_allclose(np.abs(amps), 1e3, rtol=2e-3)


def test_plane_wave_attenuation():
    alpha = 20.0  # Np/m
    amps, _, _ = plane_wave_run(1500.0, alpha=alpha)
    ratio = abs(amps[1]) / abs(amps[0])
    assert ratio == pytest.approx(math.exp(-alpha * DX_MM), rel=1e-2)


def test_interface_reflection_and_transmission():
    """Normal incidence on a water/bone-like interface: R and T from impedances."""
    f, dx = 1e6, 5e-5
    grid = Grid(nx=1000, ny=2, dx=dx)
    medium = Medium.uniform(grid, 1500.0, 1000.0)
    x_int = 25e-3
    right = grid.x >= x_int
    medium.sound_speed[right] = 2800.0
    medium.density[right] = 1850.0
    sim = Simulation(grid, medium, f, pml=PML(60, thickness_y=0), cfl=0.3, backend="numpy")
    sim.add_source(plane_source(grid, 5e-3, 1e3, gaussian_pulse(f, cycles=2)))
    rx = [np.ravel_multi_index(grid.index_of(x, 0.0), grid.shape) for x in (15e-3, 40e-3)]
    t, tr = sim.record(round(30e-6 / sim.dt), rx)

    z1, z2 = 1500.0 * 1000.0, 2800.0 * 1850.0
    r_theory = (z2 - z1) / (z2 + z1)
    t_theory = 2 * z2 / (z1 + z2)
    split = 16e-6  # incident pulse has passed the receiver, the reflection is not back yet
    inc = np.abs(tr[0, t < split]).max()
    ref = np.abs(tr[0, t >= split]).max()
    trans = np.abs(tr[1]).max()
    assert ref / inc == pytest.approx(r_theory, rel=3e-2)
    assert trans / inc == pytest.approx(t_theory, rel=3e-2)
