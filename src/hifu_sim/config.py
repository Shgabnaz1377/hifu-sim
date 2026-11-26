"""YAML run configuration and the end-to-end pipeline used by the CLI."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .analysis import focal_metrics
from .grid import Grid
from .medium import Medium, Region
from .pml import PML
from .solver import CWResult, Simulation
from .source import ArcTransducer
from .thermal import BioheatSolver, ThermalResult, absorbed_power_density


def _take(d: dict[str, Any], section: str, allowed: set[str]) -> dict[str, Any]:
    raw = d.get(section) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"[{section}] must be a mapping")
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown keys in [{section}]: {sorted(unknown)}")
    return dict(raw)


@dataclass
class GridConfig:
    size: tuple[float, float] = (0.05, 0.044)
    spacing: float = 1e-4


@dataclass
class PMLConfig:
    thickness: int = 20
    order: int = 3
    reflection: float = 1e-5


@dataclass
class MediumConfig:
    background: str = "water"
    regions: list[Region] = field(default_factory=list)


@dataclass
class SourceConfig:
    frequency: float = 1e6
    amplitude: float = 1e5
    radius: float = 0.03
    aperture_deg: float = 70.0
    n_elements: int = 32
    focus: tuple[float, float] = (0.038, 0.022)
    steer: tuple[float, float] = (0.0, 0.0)
    ramp_cycles: float = 3.0


@dataclass
class SolverConfig:
    cfl: float = 0.3
    record_cycles: int = 4
    backend: str = "auto"
    t_end: float | None = None


@dataclass
class ThermalConfig:
    enabled: bool = True
    heating_time: float = 3.0
    cooling_time: float = 3.0
    initial_temperature: float = 37.0


@dataclass
class RunConfig:
    name: str = "run"
    description: str = ""
    grid: GridConfig = field(default_factory=GridConfig)
    pml: PMLConfig = field(default_factory=PMLConfig)
    medium: MediumConfig = field(default_factory=MediumConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    simulation: SolverConfig = field(default_factory=SolverConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunConfig:
        d = dict(data)
        top = {"name", "description", "grid", "pml", "medium", "source", "simulation", "thermal"}
        unknown = set(d) - top
        if unknown:
            raise ValueError(f"unknown top-level keys: {sorted(unknown)}")

        g = _take(d, "grid", {"size", "spacing"})
        grid = GridConfig(
            size=_pair(g.get("size", GridConfig.size), "grid.size"),
            spacing=float(g.get("spacing", GridConfig.spacing)),
        )
        pml = PMLConfig(**_floatify(_take(d, "pml", {"thickness", "order", "reflection"})))
        pml.thickness, pml.order = int(pml.thickness), int(pml.order)

        m = _take(d, "medium", {"background", "regions"})
        medium = MediumConfig(
            background=str(m.get("background", "water")),
            regions=[Region.from_dict(r) for r in (m.get("regions") or [])],
        )

        s = _take(
            d, "source",
            {"frequency", "amplitude", "radius", "aperture_deg", "n_elements", "focus", "steer",
             "ramp_cycles"},
        )
        source = SourceConfig(
            frequency=float(s.get("frequency", SourceConfig.frequency)),
            amplitude=float(s.get("amplitude", SourceConfig.amplitude)),
            radius=float(s.get("radius", SourceConfig.radius)),
            aperture_deg=float(s.get("aperture_deg", SourceConfig.aperture_deg)),
            n_elements=int(s.get("n_elements", SourceConfig.n_elements)),
            focus=_pair(s.get("focus", SourceConfig.focus), "source.focus"),
            steer=_pair(s.get("steer", SourceConfig.steer), "source.steer"),
            ramp_cycles=float(s.get("ramp_cycles", SourceConfig.ramp_cycles)),
        )

        sim = _take(d, "simulation", {"cfl", "record_cycles", "backend", "t_end"})
        solver = SolverConfig(
            cfl=float(sim.get("cfl", SolverConfig.cfl)),
            record_cycles=int(sim.get("record_cycles", SolverConfig.record_cycles)),
            backend=str(sim.get("backend", SolverConfig.backend)),
            t_end=None if sim.get("t_end") is None else float(sim["t_end"]),
        )

        th = _take(d, "thermal", {"enabled", "heating_time", "cooling_time", "initial_temperature"})
        thermal = ThermalConfig(
            enabled=bool(th.get("enabled", True)),
            heating_time=float(th.get("heating_time", ThermalConfig.heating_time)),
            cooling_time=float(th.get("cooling_time", ThermalConfig.cooling_time)),
            initial_temperature=float(
                th.get("initial_temperature", ThermalConfig.initial_temperature)
            ),
        )
        return cls(
            name=str(d.get("name", "run")),
            description=str(d.get("description", "")),
            grid=grid, pml=pml, medium=medium, source=source, simulation=solver, thermal=thermal,
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["medium"]["regions"] = [
            {k: v for k, v in asdict(r).items() if v not in (0.0, (0.0, 0.0))}
            for r in self.medium.regions
        ]
        return out


def _pair(value: Any, name: str) -> tuple[float, float]:
    try:
        a, b = value
        return (float(a), float(b))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a pair of numbers, got {value!r}") from None


def _floatify(d: dict[str, Any]) -> dict[str, Any]:
    return {k: float(v) for k, v in d.items()}


def load_config(path: str | Path) -> RunConfig:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return RunConfig.from_dict(data)


@dataclass
class Setup:
    grid: Grid
    medium: Medium
    pml: PML
    transducer: ArcTransducer
    simulation: Simulation


def build(config: RunConfig, backend: str | None = None) -> Setup:
    """Instantiate grid, medium, transducer and solver from a configuration."""
    grid = Grid.from_size(*config.grid.size, config.grid.spacing)
    f0 = config.source.frequency
    medium = Medium.from_regions(grid, config.medium.background, config.medium.regions, f0)
    pml = PML(config.pml.thickness, order=config.pml.order, reflection=config.pml.reflection)
    s = config.source
    transducer = ArcTransducer(
        frequency=f0, radius=s.radius, aperture_deg=s.aperture_deg, focus=s.focus,
        amplitude=s.amplitude, n_elements=s.n_elements, steer=s.steer, ramp_cycles=s.ramp_cycles,
    )
    sim = Simulation(
        grid, medium, f0, pml=pml, cfl=config.simulation.cfl,
        backend=backend or config.simulation.backend,
    )
    i, j = np.unravel_index(transducer.to_source(grid, 1.0).indices, grid.shape)
    c0 = float(np.median(medium.sound_speed[i, j]))
    sim.add_source(transducer.to_source(grid, c0))
    return Setup(grid, medium, pml, transducer, sim)


@dataclass
class RunOutput:
    config: RunConfig
    setup: Setup
    acoustic: CWResult
    thermal: ThermalResult | None
    metrics: dict[str, float]
    files: list[Path]


def run(
    config: RunConfig,
    out_dir: str | Path,
    backend: str | None = None,
    plots: bool = True,
    thermal: bool | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> RunOutput:
    """Run acoustics (+ optional thermal) and write ``<name>.npz``, JSON metrics and PNGs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup = build(config, backend)
    acoustic = setup.simulation.run_cw(
        t_end=config.simulation.t_end,
        record_periods=config.simulation.record_cycles,
        ramp_cycles=config.source.ramp_cycles,
        progress=progress,
    )
    metrics = focal_metrics(acoustic, setup.transducer)

    do_thermal = config.thermal.enabled if thermal is None else thermal
    thermal_result: ThermalResult | None = None
    q = absorbed_power_density(setup.medium.alpha, acoustic.intensity)
    if do_thermal:
        q_interior = np.zeros_like(q)
        q_interior[acoustic.interior] = q[acoustic.interior]
        bio = BioheatSolver(setup.grid, setup.medium, config.thermal.initial_temperature)
        thermal_result = bio.simulate(
            q_interior,
            heating_time=config.thermal.heating_time,
            cooling_time=config.thermal.cooling_time,
            initial_temperature=config.thermal.initial_temperature,
        )
        metrics["temperature_max_C"] = float(np.max(thermal_result.temperature_max))
        metrics["cem43_max_min"] = float(np.max(thermal_result.cem43))
        metrics["thermal_dt_s"] = thermal_result.dt

    files: list[Path] = []
    npz_path = out / f"{config.name}.npz"
    arrays: dict[str, Any] = {
        "x": setup.grid.x,
        "y": setup.grid.y,
        "sound_speed": setup.medium.sound_speed,
        "density": setup.medium.density,
        "alpha_np_per_m": setup.medium.alpha,
        "labels": setup.medium.labels,
        "pressure_amplitude": acoustic.amplitude,
        "pressure_phase": acoustic.phase,
        "peak_positive": acoustic.peak_positive,
        "peak_negative": acoustic.peak_negative,
        "intensity": acoustic.intensity,
        "heat_source": q,
        "pml_cells": np.array([setup.pml.nx_cells, setup.pml.ny_cells]),
    }
    if thermal_result is not None:
        arrays.update(
            temperature_max=thermal_result.temperature_max,
            cem43=thermal_result.cem43,
            probe_times=thermal_result.times,
            probe_temperature=thermal_result.probe_temperature,
        )
    np.savez_compressed(npz_path, **arrays)
    files.append(npz_path)

    meta_path = out / f"{config.name}_metrics.json"
    meta = {"config": config.to_dict(), "metrics": metrics,
            "material_names": setup.medium.material_names}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    files.append(meta_path)

    if plots:
        from .plotting import save_run_figures

        files.extend(save_run_figures(out, config.name, setup, acoustic, thermal_result))
    return RunOutput(config, setup, acoustic, thermal_result, metrics, files)
