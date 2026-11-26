"""The naive, NumPy and Numba kernels implement the same update."""

import warnings

import numpy as np
import pytest

from hifu_sim import PML, Grid, Medium, Simulation, kernels


def _random_medium(grid: Grid, seed: int = 1) -> Medium:
    rng = np.random.default_rng(seed)
    medium = Medium.uniform(grid, 1500.0, 1000.0, alpha=5.0)
    medium.sound_speed += rng.uniform(0, 1300, grid.shape)
    medium.density += rng.uniform(0, 850, grid.shape)
    medium.alpha += rng.uniform(0, 100, grid.shape)
    return medium


def _final_pressure(backend: str, shape=(64, 48), steps=40) -> np.ndarray:
    grid = Grid(*shape, dx=1e-4)
    sim = Simulation(grid, _random_medium(grid), 1e6, pml=PML(8), backend=backend)
    sim.set_initial_pressure(np.random.default_rng(2).standard_normal(grid.shape))
    sim.run(steps)
    return sim.pressure


def test_naive_matches_numpy():
    ref = _final_pressure("numpy")
    out = _final_pressure("naive")
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12 * np.abs(ref).max())


def test_numba_matches_numpy():
    pytest.importorskip("numba")
    ref = _final_pressure("numpy", shape=(120, 90), steps=200)
    out = _final_pressure("numba", shape=(120, 90), steps=200)
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-10 * np.abs(ref).max())


def test_single_lateral_cell_grid():
    """ny == 1 (pure 1D) works and the lateral velocity stays zero."""
    grid = Grid(50, 1, dx=1e-4)
    sim = Simulation(grid, Medium.uniform(grid, 1500, 1000), 1e6, pml=PML(5, thickness_y=0),
                     backend="numpy")
    p0 = np.zeros(grid.shape)
    p0[25, 0] = 1.0
    sim.set_initial_pressure(p0)
    sim.run(10)
    assert np.all(sim.fields.uy == 0)
    assert np.abs(sim.pressure).max() > 0


def test_auto_backend_resolves():
    assert kernels.resolve_backend("auto") in ("numpy", "numba")
    assert kernels.resolve_backend("numpy") == "numpy"
    with pytest.raises(ValueError):
        kernels.resolve_backend("cuda")


def test_numba_request_falls_back_without_numba(monkeypatch):
    monkeypatch.setattr(kernels, "NUMBA_AVAILABLE", False)
    with pytest.warns(RuntimeWarning, match="falling back"):
        assert kernels.resolve_backend("numba") == "numpy"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert kernels.resolve_backend("auto") == "numpy"
