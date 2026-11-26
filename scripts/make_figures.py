"""Regenerate the figures in docs/figures used by the README.

    python scripts/make_figures.py

Runs the three example configurations, a steering sweep and two analytic validation
cases, writes PNGs to ``docs/figures`` and a summary of the numbers quoted in the README
to ``docs/figures/metrics.json``. Takes about a minute with Numba installed.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hifu_sim import (
    PML,
    Grid,
    Medium,
    Simulation,
    cw_waveform,
    focal_metrics,
    plane_source,
    point_source,
)
from hifu_sim.config import RunConfig, build, load_config, run
from hifu_sim.plotting import (
    DIVERGING,
    SERIES,
    TEXT,
    TEXT_2,
    acoustic_figure,
    plot_amplitude,
    plot_medium,
    show_map,
    style,
    thermal_figure,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"


def water_overview(metrics: dict[str, object]) -> None:
    cfg = load_config(ROOT / "configs" / "water.yaml")
    setup = build(cfg)
    res = setup.simulation.run_cw(record_periods=cfg.simulation.record_cycles)
    water = focal_metrics(res, setup.transducer)
    water["analytic_gain"] = setup.transducer.focal_gain_estimate(1500.0)
    i, j = setup.grid.index_of(*setup.transducer.focus)
    water["gain_at_geometric_focus"] = float(res.amplitude[i, j] / setup.transducer.amplitude)
    metrics["water"] = water
    snapshot = res.amplitude * np.cos(res.phase)  # p(t) at a phase-zero instant
    vmax = float(np.max(np.abs(snapshot[res.interior])))
    with style():
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
        plot_medium(axes[0], setup, res)
        show_map(axes[1], snapshot / 1e6, res, DIVERGING, "p [MPa]", vmin=-vmax / 1e6,
                  vmax=vmax / 1e6)
        axes[1].set_title("Instantaneous pressure (steady state)", loc="left")
        plot_amplitude(axes[2], res, setup.transducer.target)
    fig.savefig(FIG / "water_focus.png")
    plt.close(fig)


def steering_sweep(metrics: dict[str, object]) -> None:
    base = load_config(ROOT / "configs" / "water.yaml").to_dict()
    steers = [(0.0, -5e-3), (0.0, 0.0), (-4e-3, 5e-3)]
    out = []
    with style():
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
        for ax, steer in zip(axes, steers, strict=True):
            d = dict(base)
            d["source"] = dict(base["source"], steer=list(steer))
            d["medium"] = {"background": "water", "regions": []}
            setup = build(RunConfig.from_dict(d))
            res = setup.simulation.run_cw()
            m = focal_metrics(res, setup.transducer)
            out.append({"steer_mm": [s * 1e3 for s in steer], **m})
            title = f"steer = ({steer[0] * 1e3:+.0f}, {steer[1] * 1e3:+.0f}) mm"
            plot_amplitude(ax, res, setup.transducer.target, title=title)
            ex, ey = setup.transducer.element_centres()
            delays = setup.transducer.element_delays(1500.0) * 1e6
            ax.scatter(ex * 1e3, ey * 1e3, c=delays, cmap="Greys", s=12, vmin=0,
                       vmax=max(float(np.max(delays)), 1e-3), edgecolors=TEXT, linewidths=0.4)
    fig.savefig(FIG / "steering.png")
    plt.close(fig)
    metrics["steering"] = out


def config_runs(metrics: dict[str, object], tmp: Path) -> None:
    for name in ("tissue_layer", "bone_inclusion"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        result = run(cfg, tmp / name, plots=False)
        metrics[name] = result.metrics
        fig = acoustic_figure(result.setup, result.acoustic)
        fig.savefig(FIG / f"{name}_acoustic.png")
        plt.close(fig)
        assert result.thermal is not None
        fig = thermal_figure(result.acoustic, result.thermal)
        fig.savefig(FIG / f"{name}_thermal.png")
        plt.close(fig)


def validation(metrics: dict[str, object]) -> None:
    f, c, dx, a = 1e6, 1500.0, 1e-4, 1e3
    k = 2 * math.pi * f / c

    grid = Grid(240, 240, dx)
    sim = Simulation(grid, Medium.uniform(grid, c, 1000.0), f, pml=PML(20))
    sim.add_source(point_source(grid, (grid.x[120], grid.y[120]), a, cw_waveform(f)))
    res = sim.run_cw()
    idx = np.arange(125, 220)
    r = (idx - 120) * dx
    sim_amp = res.amplitude[idx, 120]
    theory = a * dx * np.sqrt(k / (2 * math.pi * r))

    grid1 = Grid(500, 2, dx)
    alpha = 20.0
    sim1 = Simulation(grid1, Medium.uniform(grid1, c, 1000.0, alpha), f,
                      pml=PML(40, thickness_y=0))
    x0 = 5e-3
    sim1.add_source(plane_source(grid1, x0, a, cw_waveform(f)))
    res1 = sim1.run_cw()
    sl = slice(70, 450)
    x1 = grid1.x[sl]
    amp1 = res1.amplitude[sl, 0]
    i0 = grid1.index_of(x0, 0.0)[0]
    theory1 = a * np.exp(-alpha * (x1 - grid1.x[i0]))

    err_point = float(np.max(np.abs(sim_amp[25:] / theory[25:] - 1)))
    err_plane = float(np.max(np.abs(amp1 / theory1 - 1)))
    metrics["validation"] = {
        "point_source_max_rel_error_r_ge_3mm": err_point,
        "plane_wave_attenuation_max_rel_error": err_plane,
    }

    with style():
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
        ax = axes[0]
        ax.plot(r * 1e3, theory, color=TEXT_2, lw=1.5, ls="--",
                label=r"theory $A\,\Delta x\sqrt{k/2\pi r}$")
        ax.plot(r[::4] * 1e3, sim_amp[::4], "o", ms=5, color=SERIES[0], label="FDTD")
        ax.set_xlabel("distance from point source r [mm]")
        ax.set_ylabel("|p| [Pa]")
        ax.set_title("2D point source: cylindrical spreading", loc="left")
        ax.legend()
        ax.grid(True, axis="y")
        ax = axes[1]
        ax.plot(x1 * 1e3, theory1, color=TEXT_2, lw=1.5, ls="--",
                label=r"theory $A\,e^{-\alpha x}$")
        ax.plot(x1[::15] * 1e3, amp1[::15], "o", ms=5, color=SERIES[0], label="FDTD")
        ax.set_xlabel("axial x [mm]")
        ax.set_ylabel("|p| [Pa]")
        ax.set_title(r"Plane wave, $\alpha$ = 20 Np/m: absorption", loc="left")
        ax.legend()
        ax.grid(True, axis="y")
    fig.savefig(FIG / "validation.png")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {}


    with tempfile.TemporaryDirectory() as tmp:
        water_overview(metrics)
        steering_sweep(metrics)
        config_runs(metrics, Path(tmp))
        validation(metrics)
    (FIG / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
