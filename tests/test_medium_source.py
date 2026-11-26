"""Material maps, geometric regions and transducer geometry."""

import math

import numpy as np
import pytest

from hifu_sim import MATERIALS, PML, ArcTransducer, Grid, Medium, Region, Simulation
from hifu_sim.medium import db_cm_to_np_m


def test_db_to_neper_conversion():
    # 1 dB/cm = 100 dB/m = 100 / 8.686 Np/m
    assert db_cm_to_np_m(1.0) == pytest.approx(11.5129, rel=1e-4)


def test_power_law_attenuation():
    tissue = MATERIALS["soft_tissue"]
    a1 = tissue.alpha_np_per_m(1e6)
    a2 = tissue.alpha_np_per_m(2e6)
    assert a2 / a1 == pytest.approx(2**tissue.alpha_power)


def test_regions_paint_in_order():
    grid = Grid(100, 80, 1e-4)
    regions = [
        Region(kind="layer", material="soft_tissue", x_min=4e-3, x_max=1.0),
        Region(kind="circle", material="bone", center=(6e-3, 4e-3), radius=1e-3),
    ]
    medium = Medium.from_regions(grid, "water", regions, 1e6)
    assert medium.material_names == ["water", "soft_tissue", "bone"]
    i, j = grid.index_of(6e-3, 4e-3)
    assert medium.sound_speed[i, j] == MATERIALS["bone"].sound_speed
    assert medium.sound_speed[5, 5] == MATERIALS["water"].sound_speed
    assert medium.sound_speed[90, 5] == MATERIALS["soft_tissue"].sound_speed
    n_bone = int((medium.labels == 2).sum())
    assert n_bone == pytest.approx(math.pi * 10**2, rel=0.05)


def test_region_from_dict_validation():
    r = Region.from_dict({"kind": "circle", "material": "bone", "center": [1, 2], "radius": 3})
    assert r.center == (1.0, 2.0)
    with pytest.raises(ValueError):
        Region.from_dict({"kind": "hexagon", "material": "bone"})
    with pytest.raises(ValueError):
        Region.from_dict({"kind": "layer", "material": "bone", "thickness": 1})


def test_unknown_material():
    with pytest.raises(ValueError, match="unknown material"):
        Medium.homogeneous(Grid(10, 10, 1e-4), "unobtainium")


def test_arc_geometry_and_rasterisation():
    grid = Grid.from_size(0.04, 0.04, 1e-4)
    tr = ArcTransducer(1e6, radius=0.025, aperture_deg=60, focus=(0.03, 0.02))
    x, y, _ = tr.arc_points(50)
    np.testing.assert_allclose(np.hypot(x - 0.03, y - 0.02), 0.025)
    src = tr.to_source(grid, 1500.0)
    # weights add up to the arc length measured in cells
    assert src.weights.sum() == pytest.approx(tr.arc_length() / grid.dx, rel=1e-3)
    np.testing.assert_allclose(src.delays, 0.0, atol=1e-15)  # no steering: all in phase
    assert set(np.unique(src.element_ids)) == set(range(tr.n_elements))


def test_steering_delays_equalise_arrival_times():
    c0 = 1500.0
    tr = ArcTransducer(1e6, radius=0.03, aperture_deg=70, focus=(0.04, 0.02), steer=(2e-3, 4e-3))
    ex, ey = tr.element_centres()
    tx, ty = tr.target
    arrival = tr.element_delays(c0) + np.hypot(ex - tx, ey - ty) / c0
    np.testing.assert_allclose(arrival, arrival[0], rtol=1e-12)
    assert tr.element_delays(c0).min() == 0.0


def test_source_outside_grid_or_in_pml_rejected():
    grid = Grid.from_size(0.02, 0.02, 1e-4)
    tr = ArcTransducer(1e6, radius=0.03, aperture_deg=60, focus=(0.015, 0.01))
    with pytest.raises(ValueError, match="outside"):
        tr.to_source(grid, 1500.0)
    small = ArcTransducer(1e6, radius=0.012, aperture_deg=60, focus=(0.0135, 0.01))
    sim = Simulation(grid, Medium.homogeneous(grid), 1e6, pml=PML(20))
    with pytest.raises(ValueError, match="PML"):
        sim.add_source(small.to_source(grid, 1500.0))


def test_invalid_transducer():
    with pytest.raises(ValueError):
        ArcTransducer(1e6, radius=0.03, aperture_deg=190, focus=(0, 0))
