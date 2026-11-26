"""Computational grid and CFL time-step helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Grid:
    """Uniform 2D Cartesian grid.

    ``x`` is the axial (propagation) direction and ``y`` the lateral one. Pressure
    lives at cell centres ``(x_i, y_j) = ((i + 1/2) dx, (j + 1/2) dy)``; the velocity
    components live on the cell faces normal to their direction (staggered grid).
    """

    nx: int
    ny: int
    dx: float
    dy: float | None = None

    def __post_init__(self) -> None:
        if self.nx < 3 or self.ny < 1:
            raise ValueError(f"grid too small: nx={self.nx}, ny={self.ny}")
        if self.dx <= 0 or (self.dy is not None and self.dy <= 0):
            raise ValueError("grid spacing must be positive")

    @property
    def spacing_y(self) -> float:
        return self.dx if self.dy is None else self.dy

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nx, self.ny)

    @property
    def size(self) -> tuple[float, float]:
        """Physical extent ``(Lx, Ly)`` in metres."""
        return (self.nx * self.dx, self.ny * self.spacing_y)

    @property
    def x(self) -> FloatArray:
        """Cell-centre x coordinates [m]."""
        return np.asarray((np.arange(self.nx) + 0.5) * self.dx, dtype=np.float64)

    @property
    def y(self) -> FloatArray:
        """Cell-centre y coordinates [m]."""
        return np.asarray((np.arange(self.ny) + 0.5) * self.spacing_y, dtype=np.float64)

    def mesh(self) -> tuple[FloatArray, FloatArray]:
        """Return ``(X, Y)`` coordinate arrays of shape ``(nx, ny)``."""
        xx, yy = np.meshgrid(self.x, self.y, indexing="ij")
        return xx, yy

    def index_of(self, x: float, y: float) -> tuple[int, int]:
        """Index of the cell containing point ``(x, y)``."""
        i = math.floor(x / self.dx)
        j = math.floor(y / self.spacing_y)
        if not (0 <= i < self.nx and 0 <= j < self.ny):
            raise ValueError(f"point ({x:.4g}, {y:.4g}) m lies outside the grid")
        return i, j

    @classmethod
    def from_size(cls, size_x: float, size_y: float, spacing: float) -> Grid:
        """Build a grid covering ``size_x`` by ``size_y`` metres (rounded up)."""
        nx = math.ceil(size_x / spacing - 1e-9)
        ny = math.ceil(size_y / spacing - 1e-9)
        return cls(nx=nx, ny=ny, dx=spacing)


def cfl_limit(dx: float, dy: float | None = None) -> float:
    """Largest stable value of ``c_max * dt / dx`` for the 2nd-order staggered scheme.

    The von Neumann condition for the leapfrog staggered grid in 2D is
    ``c dt sqrt(1/dx^2 + 1/dy^2) <= 1``, which for ``dx == dy`` gives ``1/sqrt(2)``.
    """
    dy = dx if dy is None else dy
    return 1.0 / (dx * math.sqrt(1.0 / dx**2 + 1.0 / dy**2))


def cfl_time_step(
    c_max: float,
    dx: float,
    cfl: float = 0.3,
    dy: float | None = None,
    frequency: float | None = None,
) -> float:
    """Time step from the Courant number ``cfl = c_max dt / dx``.

    If ``frequency`` is given, ``dt`` is reduced slightly so that one acoustic period
    holds an integer number of steps; this makes the lock-in (DFT) amplitude
    estimate over whole periods exact.
    """
    if c_max <= 0:
        raise ValueError("c_max must be positive")
    limit = cfl_limit(dx, dy)
    if not 0 < cfl < limit:
        raise ValueError(
            f"CFL number {cfl:.3f} is outside the stable range (0, {limit:.3f}) "
            "of the staggered leapfrog scheme"
        )
    dt = cfl * dx / c_max
    if frequency is not None:
        steps_per_period = math.ceil(1.0 / (frequency * dt))
        dt = 1.0 / (frequency * steps_per_period)
    return dt
