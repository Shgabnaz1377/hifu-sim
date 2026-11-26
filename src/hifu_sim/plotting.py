"""Matplotlib figures for simulation results (PML cropped, axes in mm)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

from .grid import FloatArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .config import Setup
    from .solver import CWResult
    from .thermal import ThermalResult

TEXT = "#191919"
TEXT_2 = "#666666"
GRID = "#d9d9d9"
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]

#: Single-hue sequential ramp (light -> dark blue) for magnitudes.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "hifu_blue",
    ["#ffffff", "#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"],
)
#: Second sequential context (temperature, dose): light -> dark orange.
SEQ_ORANGE = LinearSegmentedColormap.from_list(
    "hifu_orange", ["#ffffff", "#fde0cf", "#f9b48e", "#f47230", "#c9531a", "#8f3608", "#5c2203"]
)
#: Diverging blue <-> red with a neutral grey midpoint (signed pressure).
DIVERGING = LinearSegmentedColormap.from_list(
    "hifu_div", ["#0e3d7e", "#338ced", "#efefef", "#ec4647", "#8a1415"]
)

#: Fixed colour per material so a tissue keeps its colour across figures.
MATERIAL_COLORS = {
    "water": SERIES[0],
    "soft_tissue": SERIES[1],
    "bone": SERIES[2],
    "fat": SERIES[6],
    "custom": "#9c9c9c",
}

STYLE: dict[str, Any] = {
    "axes.edgecolor": TEXT_2,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 130,
    "font.size": 10,
    "lines.linewidth": 2.0,
}


def style() -> Any:
    """Context manager applying :data:`STYLE`."""
    return plt.rc_context(STYLE)  # type: ignore[arg-type]


def _extent_mm(
    result: CWResult,
) -> tuple[tuple[float, float, float, float], tuple[slice, slice]]:
    sl = result.interior
    x = result.grid.x[sl[0]] * 1e3
    y = result.grid.y[sl[1]] * 1e3
    dx, dy = result.grid.dx * 1e3 / 2, result.grid.spacing_y * 1e3 / 2
    return (x[0] - dx, x[-1] + dx, y[0] - dy, y[-1] + dy), sl


def show_map(
    ax: Axes, data: FloatArray, result: CWResult, cmap: Any, label: str, **kw: Any
) -> None:
    extent, sl = _extent_mm(result)
    im = ax.imshow(
        data[sl].T, origin="lower", extent=extent, cmap=cmap, aspect="equal",
        interpolation="nearest", **kw,
    )
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(label, color=TEXT)
    cb.outline.set_edgecolor(GRID)  # type: ignore[operator]
    cb.ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.set_xlabel("axial x [mm]")
    ax.set_ylabel("lateral y [mm]")


def plot_medium(ax: Axes, setup: Setup, result: CWResult) -> None:
    names = setup.medium.material_names
    colors = [MATERIAL_COLORS.get(n, "#9c9c9c") for n in names]
    extent, sl = _extent_mm(result)
    ax.imshow(
        setup.medium.labels[sl].T, origin="lower", extent=extent, aspect="equal",
        cmap=ListedColormap(colors), vmin=-0.5, vmax=len(names) - 0.5, interpolation="nearest",
    )
    ex, ey = setup.transducer.element_centres()
    ax.plot(ex * 1e3, ey * 1e3, "o", ms=3, color=TEXT, label="array elements")
    fx, fy = setup.transducer.focus
    ax.plot(fx * 1e3, fy * 1e3, "+", ms=10, mew=2, color=TEXT)
    handles = [
        Patch(color=c, label=n.replace("_", " ")) for n, c in zip(names, colors, strict=True)
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True, facecolor="white",
              edgecolor=GRID)
    ax.set_title("Medium and transducer", loc="left")
    ax.set_xlabel("axial x [mm]")
    ax.set_ylabel("lateral y [mm]")


def plot_amplitude(ax: Axes, result: CWResult, target: tuple[float, float] | None = None,
                   title: str = "Pressure amplitude") -> None:
    show_map(ax, result.amplitude / 1e6, result, SEQ_BLUE, "|p| [MPa]", vmin=0)
    if target is not None:
        ax.plot(target[0] * 1e3, target[1] * 1e3, "+", ms=10, mew=1.5, color=SERIES[1])
    ax.set_title(title, loc="left")


def acoustic_figure(setup: Setup, result: CWResult) -> Figure:
    with style():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
        plot_medium(axes[0], setup, result)
        plot_amplitude(axes[1], result, setup.transducer.target)
    return fig


def profiles_figure(result: CWResult) -> Figure:
    fx, fy = result.peak_location()
    i, j = result.grid.index_of(fx, fy)
    sl = result.interior
    amp = result.amplitude / result.amplitude[i, j]
    with style():
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
        axes[0].plot(result.grid.x[sl[0]] * 1e3, amp[sl[0], j], color=SERIES[0])
        axes[0].set_xlabel("axial x [mm]")
        axes[0].set_title(f"Axial profile through peak (y = {fy * 1e3:.1f} mm)", loc="left")
        axes[1].plot(result.grid.y[sl[1]] * 1e3, amp[i, sl[1]], color=SERIES[0])
        axes[1].set_xlabel("lateral y [mm]")
        axes[1].set_title(f"Lateral profile through peak (x = {fx * 1e3:.1f} mm)", loc="left")
        for ax in axes:
            ax.set_ylabel("|p| / |p|max")
            ax.axhline(0.5, color=TEXT_2, lw=1, ls="--")
            ax.text(ax.get_xlim()[0], 0.52, " -6 dB", color=TEXT_2, fontsize=8, va="bottom")
            ax.grid(True, axis="y")
            ax.set_ylim(0, 1.05)
    return fig


def thermal_figure(result: CWResult, thermal: ThermalResult) -> Figure:
    heating_time = thermal.heating_time
    with style():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        t0 = thermal.probe_temperature[0]
        show_map(axes[0], thermal.temperature_max - t0, result, SEQ_ORANGE,
                  "max temperature rise [K]", vmin=0)
        extent, sl = _extent_mm(result)
        dose = thermal.cem43[sl].T
        if np.max(dose) >= 240.0:
            axes[0].contour(dose, levels=[240.0], extent=extent, colors=[TEXT], linewidths=1.2)
            axes[0].text(0.02, 0.96, "contour: CEM43 = 240 min", transform=axes[0].transAxes,
                         color=TEXT, fontsize=8, va="top")
        axes[0].set_title("Peak temperature rise", loc="left")

        ax = axes[1]
        ax.plot(thermal.times, thermal.probe_temperature, color=SERIES[1])
        ax.axvspan(0, heating_time, color="#efefef", zorder=0)
        ax.text(heating_time / 2, ax.get_ylim()[0], "sonication", ha="center", va="bottom",
                color=TEXT_2, fontsize=8)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("temperature [°C]")
        ax.grid(True, axis="y")
        i, j = thermal.probe_index
        ax.set_title(
            f"At hottest-source cell (x={result.grid.x[i] * 1e3:.1f}, "
            f"y={result.grid.y[j] * 1e3:.1f} mm)", loc="left",
        )
    return fig


def save_run_figures(
    out_dir: Path, name: str, setup: Setup, result: CWResult, thermal: ThermalResult | None
) -> list[Path]:
    paths = []
    figs = [("acoustic", acoustic_figure(setup, result)), ("profiles", profiles_figure(result))]
    if thermal is not None:
        figs.append(("thermal", thermal_figure(result, thermal)))
    for suffix, fig in figs:
        path = out_dir / f"{name}_{suffix}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths
