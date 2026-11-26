"""Acoustic sources: waveforms, generic additive pressure sources, arc transducers."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .grid import FloatArray, Grid

Waveform = Callable[[FloatArray], FloatArray]


def cw_waveform(frequency: float, ramp_cycles: float = 3.0) -> Waveform:
    """Continuous sine wave switched on with a raised-cosine ramp.

    ``s(t) = R(t) sin(2 pi f t)`` for ``t >= 0`` and zero before, where ``R`` rises from
    0 to 1 over ``ramp_cycles`` periods. The smooth start avoids exciting high
    wavenumbers the grid cannot resolve.
    """
    omega = 2.0 * math.pi * frequency
    t_ramp = ramp_cycles / frequency

    def s(t: FloatArray) -> FloatArray:
        t = np.asarray(t, dtype=np.float64)
        if t_ramp > 0:
            ramp = np.where(t < t_ramp, 0.5 * (1.0 - np.cos(np.pi * t / t_ramp)), 1.0)
        else:
            ramp = np.ones_like(t)
        out: FloatArray = np.where(t >= 0.0, ramp * np.sin(omega * t), 0.0)
        return out

    return s


def gaussian_pulse(frequency: float, cycles: float = 3.0) -> Waveform:
    """Gaussian-modulated sine pulse centred at ``t0 = 3 * width`` (starts near zero).

    ``width`` is chosen so the -6 dB duration is about ``cycles`` periods.
    """
    omega = 2.0 * math.pi * frequency
    width = cycles / (frequency * 2.355)
    t0 = 3.0 * width

    def s(t: FloatArray) -> FloatArray:
        t = np.asarray(t, dtype=np.float64)
        out: FloatArray = np.exp(-(((t - t0) / width) ** 2) / 2.0) * np.sin(omega * (t - t0))
        return out

    return s


@dataclass
class PressureSource:
    """Additive pressure (mass) source on a set of grid cells.

    The injected rate in cell ``n`` is ``S_n(t) = 2 c_n / dx * w_n * A * s(t - tau_n)``,
    which, for a straight row of cells with ``w_n = 1``, radiates a plane wave of
    pressure amplitude ``A`` to each side (1D jump condition ``p = S dx / (2c)``).
    ``weights`` correct for how much source line each cell represents. The solver also
    multiplies by ``cos(k dx / 2)`` to cancel the staggered grid's discrete source
    response at the drive frequency (see :meth:`Simulation.add_source`).
    """

    indices: NDArray[np.intp]  # flat indices into the (nx, ny) pressure array
    weights: FloatArray
    delays: FloatArray  # seconds
    amplitude: float  # Pa
    waveform: Waveform
    element_ids: NDArray[np.intp] | None = None

    def __post_init__(self) -> None:
        n = self.indices.size
        if self.weights.shape != (n,) or self.delays.shape != (n,):
            raise ValueError("indices, weights and delays must have the same length")
        if np.unique(self.indices).size != n:
            raise ValueError("source cell indices must be unique")

    def rate(self, t: float, scale: FloatArray) -> FloatArray:
        """Injection rate [Pa/s] per source cell at time ``t``.

        ``scale`` is the per-cell factor ``2 c / dx`` supplied by the solver.
        """
        result: FloatArray = scale * self.weights * self.amplitude * self.waveform(
            t - self.delays
        )
        return result

    @property
    def max_delay(self) -> float:
        return float(np.max(self.delays)) if self.delays.size else 0.0


def point_source(
    grid: Grid, position: tuple[float, float], amplitude: float, waveform: Waveform
) -> PressureSource:
    i, j = grid.index_of(*position)
    idx = np.array([np.ravel_multi_index((i, j), grid.shape)], dtype=np.intp)
    return PressureSource(idx, np.ones(1), np.zeros(1), amplitude, waveform)


def plane_source(
    grid: Grid, x: float, amplitude: float, waveform: Waveform
) -> PressureSource:
    """Source filling the full column of cells at axial position ``x``."""
    i, _ = grid.index_of(x, 0.0)
    idx = np.ravel_multi_index((np.full(grid.ny, i), np.arange(grid.ny)), grid.shape)
    n = grid.ny
    return PressureSource(idx.astype(np.intp), np.ones(n), np.zeros(n), amplitude, waveform)


@dataclass(frozen=True)
class ArcTransducer:
    """Cylindrically focused arc array (2D analogue of a spherical bowl).

    The arc has radius ``radius`` and is centred on the geometric ``focus``; it opens
    towards ``+x`` with a full aperture angle ``aperture_deg``. It is divided into
    ``n_elements`` equal-angle elements. With ``steer`` set, each element is delayed
    so that all wavefronts arrive at ``focus + steer`` simultaneously (phased-array
    steering); with no steering all delays are zero.
    """

    frequency: float
    radius: float
    aperture_deg: float
    focus: tuple[float, float]
    amplitude: float = 1e5
    n_elements: int = 32
    steer: tuple[float, float] = (0.0, 0.0)
    ramp_cycles: float = 3.0

    def __post_init__(self) -> None:
        if not 0 < self.aperture_deg < 180:
            raise ValueError("aperture_deg must be in (0, 180)")
        if self.radius <= 0 or self.frequency <= 0:
            raise ValueError("radius and frequency must be positive")
        if self.n_elements < 1:
            raise ValueError("n_elements must be >= 1")

    @property
    def half_angle(self) -> float:
        return math.radians(self.aperture_deg) / 2.0

    @property
    def target(self) -> tuple[float, float]:
        return (self.focus[0] + self.steer[0], self.focus[1] + self.steer[1])

    def arc_points(self, n: int) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``n`` points ``(x, y, theta)`` evenly spaced in angle along the arc."""
        theta = np.linspace(-self.half_angle, self.half_angle, n)
        x = self.focus[0] - self.radius * np.cos(theta)
        y = self.focus[1] + self.radius * np.sin(theta)
        return x, y, theta

    def element_centres(self) -> tuple[FloatArray, FloatArray]:
        edges = np.linspace(-self.half_angle, self.half_angle, self.n_elements + 1)
        theta = 0.5 * (edges[1:] + edges[:-1])
        return (
            self.focus[0] - self.radius * np.cos(theta),
            self.focus[1] + self.radius * np.sin(theta),
        )

    def element_delays(self, c0: float) -> FloatArray:
        """Per-element firing delays [s] focusing on :attr:`target` in speed ``c0``."""
        ex, ey = self.element_centres()
        tx, ty = self.target
        dist = np.hypot(ex - tx, ey - ty)
        delays: FloatArray = (np.max(dist) - dist) / c0
        return delays

    def arc_length(self) -> float:
        return 2.0 * self.radius * self.half_angle

    def to_source(self, grid: Grid, c0: float) -> PressureSource:
        """Rasterise the arc onto ``grid``.

        The arc is sampled at ``dx / 8`` spacing and each sample is assigned to the
        cell containing it. A cell's weight is the arc length it contains divided by
        ``dx``, so the total source strength per unit arc length is independent of how
        the staircase falls on the grid.
        """
        ds = min(grid.dx, grid.spacing_y) / 8.0
        n = max(math.ceil(self.arc_length() / ds) + 1, 2)
        x, y, theta = self.arc_points(n)
        seg = self.arc_length() / (n - 1)
        i = np.floor(x / grid.dx).astype(np.intp)
        j = np.floor(y / grid.spacing_y).astype(np.intp)
        if np.min(i) < 0 or np.min(j) < 0 or np.max(i) >= grid.nx or np.max(j) >= grid.ny:
            raise ValueError("transducer arc extends outside the computational grid")
        flat = np.ravel_multi_index((i, j), grid.shape)
        cells, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
        weights = counts * seg / grid.dx

        # element id of each cell: majority element of its samples (use mean angle)
        theta_sum = np.zeros(cells.size)
        np.add.at(theta_sum, inverse, theta)
        theta_cell = theta_sum / counts
        elem = np.floor(
            (theta_cell + self.half_angle) / (2.0 * self.half_angle) * self.n_elements
        ).astype(np.intp)
        elem = np.clip(elem, 0, self.n_elements - 1)
        delays = self.element_delays(c0)[elem]
        return PressureSource(
            indices=cells.astype(np.intp),
            weights=weights.astype(np.float64),
            delays=delays.astype(np.float64),
            amplitude=self.amplitude,
            waveform=cw_waveform(self.frequency, self.ramp_cycles),
            element_ids=elem,
        )

    def focal_gain_estimate(self, c0: float) -> float:
        """Linear 2D focal pressure gain ``L / sqrt(lambda R)`` (Fresnel approximation).

        Obtained from the 2D Rayleigh integral with the large-argument form of the
        Hankel function: every element of arc length ``ds`` contributes
        ``p0 sqrt(k / (2 pi R)) ds`` in phase at the geometric focus.
        """
        wavelength = c0 / self.frequency
        return self.arc_length() / math.sqrt(wavelength * self.radius)
