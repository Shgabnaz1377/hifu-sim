"""Focusing in homogeneous water: focal position, gain, focal width and steering."""

import math

import numpy as np
import pytest

from hifu_sim import PML, ArcTransducer, Grid, Medium, Simulation, focal_metrics, width_at_level

F0 = 1e6
C0 = 1500.0
DX = 1e-4  # 15 points per wavelength
RADIUS = 0.025
APERTURE_DEG = 60.0
FOCUS = (0.030, 0.016)


def _run(steer=(0.0, 0.0)):
    grid = Grid.from_size(0.036, 0.032, DX)
    medium = Medium.homogeneous(grid, "water", F0)
    tr = ArcTransducer(F0, radius=RADIUS, aperture_deg=APERTURE_DEG, focus=FOCUS,
                       amplitude=1e5, n_elements=24, steer=steer)
    sim = Simulation(grid, medium, F0, pml=PML(20), cfl=0.3, backend="numpy")
    sim.add_source(tr.to_source(grid, C0))
    result = sim.run_cw()
    return tr, result, focal_metrics(result, tr)


@pytest.fixture(scope="module")
def unsteered():
    return _run()


def test_focal_spot_near_geometric_focus(unsteered):
    _, _, m = unsteered
    # Lateral: on axis within one cell. Axial: the maximum sits slightly *before* the
    # geometric focus (focal shift of a finite-Fresnel-number aperture), never after.
    assert abs(m["lateral_offset_m"]) <= DX
    assert -0.1 * RADIUS <= m["axial_offset_m"] <= DX


def test_focal_gain_matches_rayleigh_integral(unsteered):
    """Amplitude at the geometric focus vs the 2D Rayleigh integral, L / sqrt(lambda R)."""
    tr, r, _ = unsteered
    i, j = r.grid.index_of(*FOCUS)
    gain = r.amplitude[i, j] / tr.amplitude
    assert gain == pytest.approx(tr.focal_gain_estimate(C0), rel=0.05)


def test_lateral_focal_width_matches_diffraction_integral(unsteered):
    """Focal-plane profile vs |integral_{-theta}^{theta} exp(i k y sin(phi)) d(phi)|."""
    _, r, _ = unsteered
    i, _ = r.grid.index_of(*FOCUS)
    k = 2 * math.pi * F0 / C0
    half = math.radians(APERTURE_DEG / 2)
    phi = np.linspace(-half, half, 2001)
    y = np.linspace(-3e-3, 3e-3, 1201)
    theory = np.abs(np.exp(1j * k * y[:, None] * np.sin(phi)).sum(axis=1))  # uniform phi grid
    expected = width_at_level(theory, y[1] - y[0])
    measured = width_at_level(r.amplitude[i, r.interior[1]], DX)
    assert measured == pytest.approx(expected, rel=0.05)


def test_focal_spot_is_elongated(unsteered):
    _, _, m = unsteered
    assert m["fwhm_axial_m"] > 4 * m["fwhm_lateral_m"]


def test_peak_pressures_are_symmetric_in_linear_acoustics(unsteered):
    _, r, _ = unsteered
    pos = r.peak_positive[r.interior].max()
    neg = -r.peak_negative[r.interior].min()
    assert pos == pytest.approx(neg, rel=0.02)
    assert np.max(r.amplitude[r.interior]) == pytest.approx(pos, rel=0.02)


def test_electronic_steering_moves_focus(unsteered):
    steer = (0.0, 3e-3)
    _, _, m = _run(steer)
    _, _, m0 = unsteered
    assert m["peak_y_m"] == pytest.approx(FOCUS[1] + steer[1], abs=2 * DX)
    assert m["peak_x_m"] == pytest.approx(m0["peak_x_m"], abs=2 * DX)
    # steering away from the geometric focus costs a little gain
    assert 0.9 * m0["focal_gain"] < m["focal_gain"] < m0["focal_gain"]
