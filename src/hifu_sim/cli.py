"""Command-line interface: ``hifu-sim run config.yaml``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .kernels import BACKENDS, NUMBA_AVAILABLE, numba_threads
from .medium import MATERIALS


def _progress(done: int, total: int) -> None:
    width = 30
    filled = int(width * done / total)
    sys.stderr.write(f"\r  [{'#' * filled}{'.' * (width - filled)}] {done}/{total} steps")
    if done == total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _cmd_run(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")  # files only, never open windows
    from .config import load_config, run

    cfg = load_config(args.config)
    if args.name:
        cfg.name = args.name
    out_dir = Path(args.output) if args.output else Path("results") / cfg.name
    print(f"hifu-sim {__version__}: running '{cfg.name}' from {args.config}")
    thermal = False if args.no_thermal else None
    result = run(
        cfg, out_dir, backend=args.backend, plots=not args.no_plots, thermal=thermal,
        progress=None if args.quiet else _progress,
    )
    m = result.metrics
    g = result.setup.grid
    print(f"  grid {g.nx} x {g.ny} (dx = {g.dx * 1e3:.3f} mm), {int(m['n_steps'])} steps, "
          f"dt = {m['dt_s'] * 1e9:.2f} ns, backend = {result.acoustic.backend}, "
          f"{m['elapsed_s']:.1f} s")
    print(f"  peak |p| = {m['peak_amplitude_Pa'] / 1e6:.3f} MPa at "
          f"({m['peak_x_m'] * 1e3:.2f}, {m['peak_y_m'] * 1e3:.2f}) mm, "
          f"gain = {m['focal_gain']:.2f}")
    print(f"  -6 dB width: axial {m['fwhm_axial_m'] * 1e3:.2f} mm, "
          f"lateral {m['fwhm_lateral_m'] * 1e3:.2f} mm")
    if "temperature_max_C" in m:
        print(f"  max temperature {m['temperature_max_C']:.2f} degC, "
              f"max CEM43 {m['cem43_max_min']:.3g} min")
    for path in result.files:
        print(f"  wrote {path}")
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    print(f"hifu-sim {__version__}")
    if NUMBA_AVAILABLE:
        print(f"numba: available ({numba_threads()} threads)")
    else:
        print("numba: not installed (NumPy kernel will be used)")
    print("materials:")
    for m in MATERIALS.values():
        print(f"  {m.name:12s} c={m.sound_speed:6.0f} m/s  rho={m.density:6.0f} kg/m^3  "
              f"alpha0={m.alpha0:g} dB/(cm MHz^{m.alpha_power:g})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hifu-sim", description="2D FDTD simulation of focused ultrasound in tissue."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run a simulation from a YAML config")
    run_p.add_argument("config", type=Path, help="YAML configuration file")
    run_p.add_argument("-o", "--output", help="output directory (default results/<name>)")
    run_p.add_argument("--name", help="override the run name used for output files")
    run_p.add_argument("--backend", choices=BACKENDS, default=None,
                       help="kernel backend (default: value in config, else auto)")
    run_p.add_argument("--no-plots", action="store_true", help="skip PNG figures")
    run_p.add_argument("--no-thermal", action="store_true", help="skip the bioheat step")
    run_p.add_argument("-q", "--quiet", action="store_true", help="no progress bar")
    run_p.set_defaults(func=_cmd_run)

    info_p = sub.add_parser("info", help="show backends and the material library")
    info_p.set_defaults(func=_cmd_info)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
