"""Perfectly matched layer (split-field) absorption profiles."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .grid import FloatArray


@dataclass(frozen=True)
class PML:
    """Polynomially graded split-field PML placed *inside* the grid edges.

    The damping profile is ``sigma(d) = sigma_max (d / L)^order`` where ``d`` is the
    depth into the layer of thickness ``L``. ``sigma_max`` follows the usual choice
    ``(order + 1) c ln(1/R) / (2 L)`` for a target normal-incidence reflection ``R``.
    A thickness of zero on an axis leaves rigid walls (``u_n = 0``) on that axis.
    """

    thickness: int = 20
    thickness_y: int | None = None
    order: int = 3
    reflection: float = 1e-5

    def __post_init__(self) -> None:
        if self.thickness < 0 or (self.thickness_y is not None and self.thickness_y < 0):
            raise ValueError("PML thickness must be non-negative")
        if not 0 < self.reflection < 1:
            raise ValueError("PML target reflection must be in (0, 1)")

    @property
    def nx_cells(self) -> int:
        return self.thickness

    @property
    def ny_cells(self) -> int:
        return self.thickness if self.thickness_y is None else self.thickness_y

    def sigma_max(self, cells: int, spacing: float, c_ref: float) -> float:
        if cells == 0:
            return 0.0
        length = cells * spacing
        return (self.order + 1) * c_ref * math.log(1.0 / self.reflection) / (2.0 * length)

    def profile(
        self, n: int, cells: int, spacing: float, c_ref: float, staggered: bool
    ) -> FloatArray:
        """Damping [1/s] along one axis.

        ``staggered=False`` samples the ``n`` cell centres, ``staggered=True`` the
        ``n + 1`` cell faces (including both outer walls).
        """
        if cells == 0:
            return np.zeros(n + 1 if staggered else n)
        pos = np.arange(n + 1, dtype=np.float64) if staggered else np.arange(n) + 0.5
        pos = pos * spacing
        length = cells * spacing
        total = n * spacing
        depth = np.maximum(length - pos, 0.0) + np.maximum(pos - (total - length), 0.0)
        return self.sigma_max(cells, spacing, c_ref) * (depth / length) ** self.order

    def interior(self, shape: tuple[int, int]) -> tuple[slice, slice]:
        """Slices selecting the non-PML part of an array of ``shape``."""
        nx, ny = shape
        ax, ay = self.nx_cells, self.ny_cells
        return (slice(ax, nx - ax), slice(ay, ny - ay))
