"""Finite-difference time-domain solver for 2D linear acoustics.

Governing equations (linear, lossy, heterogeneous)::

    rho du/dt = -grad p
    dp/dt     = -rho c^2 div u - eta p + S,      eta = 2 alpha c

On a staggered grid (pressure at cell centres, velocity on faces) with leapfrog time
stepping. The pressure is split into ``px + py`` so that a perfectly matched layer
with damping ``sigma_x(x)`` and ``sigma_y(y)`` can be applied per direction. Damping
terms are integrated semi-implicitly (Crank-Nicolson on the decay term), which is
unconditionally stable for any ``sigma`` or ``eta``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .grid import FloatArray, Grid, cfl_limit, cfl_time_step
from .kernels import Coefficients, Fields, Workspace, get_step_function, resolve_backend
from .medium import Medium
from .pml import PML
from .source import PressureSource


@dataclass
class CWResult:
    """Steady-state (continuous-wave) field quantities on the full grid.

    ``amplitude`` and ``phase`` come from a lock-in (single-bin DFT) over whole periods
    at the drive frequency; ``peak_positive``/``peak_negative`` are the extreme
    pressures seen at any time during the run. ``intensity`` is the time-averaged
    plane-wave intensity ``|p|^2 / (2 rho c)`` [W/m^2].
    """

    grid: Grid
    pml: PML
    frequency: float
    amplitude: FloatArray
    phase: FloatArray
    peak_positive: FloatArray
    peak_negative: FloatArray
    intensity: FloatArray
    dt: float
    n_steps: int
    elapsed: float
    backend: str
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def interior(self) -> tuple[slice, slice]:
        return self.pml.interior(self.grid.shape)

    def peak_location(self, mask: NDArray[np.bool_] | None = None) -> tuple[float, float]:
        """``(x, y)`` [m] of the largest amplitude outside the PML (and inside ``mask``)."""
        amp = np.zeros_like(self.amplitude)
        sl = self.interior
        amp[sl] = self.amplitude[sl]
        if mask is not None:
            amp = np.where(mask, amp, 0.0)
        i, j = np.unravel_index(int(np.argmax(amp)), amp.shape)
        return float(self.grid.x[i]), float(self.grid.y[j])


class Simulation:
    """Time-domain acoustic simulation on a 2D grid.

    Parameters
    ----------
    grid, medium:
        Geometry and property maps (``medium`` arrays must have ``grid.shape``).
    frequency:
        Reference frequency [Hz]. Used to snap ``dt`` to an integer number of steps
        per period (when ``dt`` is not given) and by :meth:`run_cw`. The attenuation
        map in ``medium`` should already be evaluated at this frequency.
    pml:
        Absorbing layer description; it occupies the outermost cells of the grid.
    cfl:
        Courant number ``c_max dt / dx``. Must be below ``1/sqrt(2)`` for ``dx == dy``.
    backend:
        ``"auto"``, ``"numpy"``, ``"numba"`` or ``"naive"``.
    """

    def __init__(
        self,
        grid: Grid,
        medium: Medium,
        frequency: float,
        pml: PML | None = None,
        cfl: float = 0.3,
        dt: float | None = None,
        backend: str = "auto",
    ) -> None:
        if medium.sound_speed.shape != grid.shape:
            raise ValueError(
                f"medium shape {medium.sound_speed.shape} does not match grid {grid.shape}"
            )
        self.grid = grid
        self.medium = medium
        self.frequency = float(frequency)
        self.pml = pml if pml is not None else PML()
        if 2 * self.pml.nx_cells >= grid.nx or (
            self.pml.ny_cells > 0 and 2 * self.pml.ny_cells >= grid.ny
        ):
            raise ValueError("PML is thicker than half the grid")

        dy = grid.spacing_y
        if dt is None:
            self.dt = cfl_time_step(medium.c_max, grid.dx, cfl, dy, frequency)
        else:
            courant = medium.c_max * dt / grid.dx
            limit = cfl_limit(grid.dx, dy)
            if courant >= limit:
                raise ValueError(
                    f"dt={dt:.3e} s gives Courant number {courant:.3f} >= stability limit "
                    f"{limit:.3f}"
                )
            self.dt = float(dt)

        self.backend = resolve_backend(backend)
        self._step_fn = get_step_function(self.backend)
        self.coefficients = self._build_coefficients()
        self.fields = Fields.zeros(grid.nx, grid.ny)
        self._work = Workspace(grid.nx, grid.ny)
        self.sources: list[PressureSource] = []
        self._source_scale: list[FloatArray] = []
        self.n = 0

    # ------------------------------------------------------------------ setup
    def _build_coefficients(self) -> Coefficients:
        g, m, dt = self.grid, self.medium, self.dt
        dx, dy = g.dx, g.spacing_y
        c_ref = m.c_max
        sx_c = self.pml.profile(g.nx, self.pml.nx_cells, dx, c_ref, staggered=False)
        sx_f = self.pml.profile(g.nx, self.pml.nx_cells, dx, c_ref, staggered=True)[1:-1]
        sy_c = self.pml.profile(g.ny, self.pml.ny_cells, dy, c_ref, staggered=False)
        sy_f = self.pml.profile(g.ny, self.pml.ny_cells, dy, c_ref, staggered=True)[1:-1]

        rho = m.density
        rho_fx = 0.5 * (rho[1:, :] + rho[:-1, :])
        rho_fy = 0.5 * (rho[:, 1:] + rho[:, :-1])
        bulk = rho * m.sound_speed**2
        eta = 2.0 * m.alpha * m.sound_speed

        def decay(sigma: FloatArray) -> tuple[FloatArray, FloatArray]:
            h = 0.5 * sigma * dt
            return (1.0 - h) / (1.0 + h), dt / (1.0 + h)

        a, b = decay(np.broadcast_to(sx_f[:, None], rho_fx.shape))
        aux, cux = a, b / (rho_fx * dx)
        a, b = decay(np.broadcast_to(sy_f[None, :], rho_fy.shape))
        auy, cuy = a, b / (rho_fy * dy)
        a, b = decay(sx_c[:, None] + eta)
        apx, cpx = a, b * bulk / dx
        a, b = decay(sy_c[None, :] + eta)
        apy, cpy = a, b * bulk / dy

        def c64(arr: FloatArray) -> FloatArray:
            return np.ascontiguousarray(arr, dtype=np.float64)

        return Coefficients(
            aux=c64(aux), cux=c64(cux), auy=c64(auy), cuy=c64(cuy),
            apx=c64(apx), cpx=c64(cpx), apy=c64(apy), cpy=c64(cpy),
        )

    def add_source(self, source: PressureSource) -> None:
        """Register an additive pressure source. Its cells must lie outside the PML."""
        i, j = np.unravel_index(source.indices, self.grid.shape)
        ax, ay = self.pml.nx_cells, self.pml.ny_cells
        nx, ny = self.grid.shape
        if np.any((i < ax) | (i >= nx - ax) | (j < ay) | (j >= ny - ay)):
            raise ValueError("source cells overlap the PML; move the source inwards")
        c = self.medium.sound_speed.ravel()[source.indices]
        # 1D jump condition p = S dx / (2c), times cos(k dx / 2): on the staggered grid a
        # single-cell source radiates 1/cos(k dx/2) too much at wavenumber k (an
        # O((k dx)^2) discretisation effect, ~2 % at 15 points per wavelength).
        k_half = math.pi * self.frequency * self.grid.dx / c
        self.sources.append(source)
        self._source_scale.append(2.0 * c / self.grid.dx * np.cos(k_half))

    def reset(self) -> None:
        self.fields = Fields.zeros(self.grid.nx, self.grid.ny)
        self.n = 0

    def set_initial_pressure(self, p0: FloatArray) -> None:
        """Initial-value problem: ``p(t=0) = p0``, ``u(t=-dt/2) = 0``."""
        if p0.shape != self.grid.shape:
            raise ValueError("initial pressure has the wrong shape")
        self.fields.px[...] = 0.5 * p0
        self.fields.py[...] = 0.5 * p0

    # ---------------------------------------------------------------- stepping
    @property
    def time(self) -> float:
        """Time of the current pressure field [s]."""
        return self.n * self.dt

    @property
    def pressure(self) -> FloatArray:
        return self.fields.pressure

    def step(self) -> None:
        self._step_fn(self.fields, self.coefficients, self._work)
        self.n += 1
        if self.sources:
            t_mid = (self.n - 0.5) * self.dt
            px = self.fields.px.reshape(-1)
            py = self.fields.py.reshape(-1)
            for src, scale in zip(self.sources, self._source_scale, strict=True):
                inc = (0.5 * self.dt) * src.rate(t_mid, scale)
                px[src.indices] += inc
                py[src.indices] += inc

    def run(self, n_steps: int, callback: Callable[[Simulation], None] | None = None) -> None:
        for _ in range(n_steps):
            self.step()
            if callback is not None:
                callback(self)

    def energy(self) -> float:
        """Total acoustic energy per unit length in z [J/m]."""
        m, f = self.medium, self.fields
        p = f.pressure
        rho = m.density
        potential = np.sum(p**2 / (2.0 * rho * m.sound_speed**2))
        rho_fx = 0.5 * (rho[1:, :] + rho[:-1, :])
        rho_fy = 0.5 * (rho[:, 1:] + rho[:, :-1])
        kinetic = 0.5 * np.sum(rho_fx * f.ux[1:-1] ** 2) + 0.5 * np.sum(rho_fy * f.uy[:, 1:-1] ** 2)
        return float((potential + kinetic) * self.grid.dx * self.grid.spacing_y)

    def record(self, n_steps: int, flat_indices: Sequence[int]) -> tuple[FloatArray, FloatArray]:
        """Run ``n_steps`` and return ``(t, traces)`` with ``traces[k, n]`` = p at point k."""
        idx = np.asarray(flat_indices, dtype=np.intp)
        traces = np.empty((idx.size, n_steps))
        t = np.empty(n_steps)
        for n in range(n_steps):
            self.step()
            t[n] = self.time
            traces[:, n] = self.fields.px.reshape(-1)[idx] + self.fields.py.reshape(-1)[idx]
        return t, traces

    def default_duration(self, ramp_cycles: float = 3.0, record_periods: int = 4) -> float:
        """Run time long enough to reach steady state across the whole grid."""
        lx, ly = self.grid.size
        transit = math.hypot(lx, ly) / self.medium.c_min
        delay = max((s.max_delay for s in self.sources), default=0.0)
        periods = (ramp_cycles + record_periods) / self.frequency
        return periods + delay + 1.2 * transit

    def run_cw(
        self,
        t_end: float | None = None,
        record_periods: int = 4,
        ramp_cycles: float = 3.0,
        progress: Callable[[int, int], None] | None = None,
    ) -> CWResult:
        """Run a continuous-wave simulation from rest and extract steady-state maps."""
        if not self.sources:
            raise RuntimeError("add a source before calling run_cw()")
        self.reset()
        if t_end is None:
            t_end = self.default_duration(ramp_cycles, record_periods)
        steps_per_period = round(1.0 / (self.frequency * self.dt))
        n_record = record_periods * steps_per_period
        n_total = max(math.ceil(t_end / self.dt), n_record + 1)

        shape = self.grid.shape
        p = np.empty(shape)
        pmax = np.zeros(shape)
        pmin = np.zeros(shape)
        acc_re = np.zeros(shape)
        acc_im = np.zeros(shape)
        omega = 2.0 * math.pi * self.frequency
        start = time.perf_counter()
        for n in range(n_total):
            self.step()
            np.add(self.fields.px, self.fields.py, out=p)
            np.maximum(pmax, p, out=pmax)
            np.minimum(pmin, p, out=pmin)
            if n >= n_total - n_record:
                phase = omega * self.time
                acc_re += math.cos(phase) * p
                acc_im -= math.sin(phase) * p
            if progress is not None and (n % 200 == 0 or n == n_total - 1):
                progress(n + 1, n_total)
        elapsed = time.perf_counter() - start

        complex_amp = (acc_re + 1j * acc_im) * (2.0 / n_record)
        amplitude = np.abs(complex_amp)
        intensity = amplitude**2 / (2.0 * self.medium.density * self.medium.sound_speed)
        return CWResult(
            grid=self.grid,
            pml=self.pml,
            frequency=self.frequency,
            amplitude=amplitude,
            phase=np.angle(complex_amp),
            peak_positive=pmax,
            peak_negative=pmin,
            intensity=intensity,
            dt=self.dt,
            n_steps=n_total,
            elapsed=elapsed,
            backend=self.backend,
        )
