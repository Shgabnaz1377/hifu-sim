"""Time-stepping kernels for the staggered-grid split-field scheme.

Three interchangeable implementations of one leapfrog step are provided:

* :func:`step_naive`: explicit Python loops, the readable reference (slow).
* :func:`step_numpy`: fully vectorised NumPy with preallocated buffers, no
  temporaries in the hot loop.
* :func:`step_numba`: the loop version compiled with Numba and parallelised over
  rows with ``prange``. Only available when Numba is installed.

All three perform exactly the same floating-point operations in the same order, so
their results agree to round-off.

Field layout for a grid of ``nx x ny`` cells::

    px, py : (nx, ny)      split pressure, p = px + py, at cell centres
    ux     : (nx + 1, ny)  x-velocity on x-faces; ux[0] and ux[-1] are rigid walls
    uy     : (nx, ny + 1)  y-velocity on y-faces; uy[:, 0] and uy[:, -1] are walls
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .grid import FloatArray

_numba: Any
try:  # pragma: no cover - exercised implicitly depending on the environment
    import numba as _numba
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _numba = None
    NUMBA_AVAILABLE = False


@dataclass
class Coefficients:
    """Precomputed update coefficients (all ``float64``, C-contiguous).

    Velocity: ``u <- a_u * u - c_u * (p[+] - p[-])``.
    Pressure: ``p_s <- a_p * p_s - c_p * (u[+] - u[-])`` for each split component ``s``.
    """

    aux: FloatArray  # (nx-1, ny)
    cux: FloatArray
    auy: FloatArray  # (nx, ny-1)
    cuy: FloatArray
    apx: FloatArray  # (nx, ny)
    cpx: FloatArray
    apy: FloatArray
    cpy: FloatArray


@dataclass
class Fields:
    px: FloatArray
    py: FloatArray
    ux: FloatArray
    uy: FloatArray

    @classmethod
    def zeros(cls, nx: int, ny: int) -> Fields:
        return cls(
            px=np.zeros((nx, ny)),
            py=np.zeros((nx, ny)),
            ux=np.zeros((nx + 1, ny)),
            uy=np.zeros((nx, ny + 1)),
        )

    @property
    def pressure(self) -> FloatArray:
        return self.px + self.py


class Workspace:
    """Scratch buffers for :func:`step_numpy`."""

    def __init__(self, nx: int, ny: int) -> None:
        self.p = np.empty((nx, ny))
        self.gx = np.empty((nx - 1, ny))
        self.gy = np.empty((nx, max(ny - 1, 0)))
        self.div = np.empty((nx, ny))


StepFunction = Callable[[Fields, Coefficients, Workspace], None]


def step_numpy(f: Fields, k: Coefficients, w: Workspace) -> None:
    """Advance velocity then pressure by one time step (vectorised, in place)."""
    p = w.p
    np.add(f.px, f.py, out=p)

    gx = w.gx
    np.subtract(p[1:], p[:-1], out=gx)
    gx *= k.cux
    ux_in = f.ux[1:-1]
    ux_in *= k.aux
    ux_in -= gx

    if w.gy.shape[1] > 0:
        gy = w.gy
        np.subtract(p[:, 1:], p[:, :-1], out=gy)
        gy *= k.cuy
        uy_in = f.uy[:, 1:-1]
        uy_in *= k.auy
        uy_in -= gy

    d = w.div
    np.subtract(f.ux[1:], f.ux[:-1], out=d)
    d *= k.cpx
    f.px *= k.apx
    f.px -= d

    np.subtract(f.uy[:, 1:], f.uy[:, :-1], out=d)
    d *= k.cpy
    f.py *= k.apy
    f.py -= d


def _step_loops(
    px: FloatArray,
    py: FloatArray,
    ux: FloatArray,
    uy: FloatArray,
    aux: FloatArray,
    cux: FloatArray,
    auy: FloatArray,
    cuy: FloatArray,
    apx: FloatArray,
    cpx: FloatArray,
    apy: FloatArray,
    cpy: FloatArray,
) -> None:
    nx, ny = px.shape
    for i in range(nx - 1):
        for j in range(ny):
            g = (px[i + 1, j] + py[i + 1, j]) - (px[i, j] + py[i, j])
            ux[i + 1, j] = ux[i + 1, j] * aux[i, j] - g * cux[i, j]
    for i in range(nx):
        for j in range(ny - 1):
            g = (px[i, j + 1] + py[i, j + 1]) - (px[i, j] + py[i, j])
            uy[i, j + 1] = uy[i, j + 1] * auy[i, j] - g * cuy[i, j]
    for i in range(nx):
        for j in range(ny):
            px[i, j] = px[i, j] * apx[i, j] - (ux[i + 1, j] - ux[i, j]) * cpx[i, j]
            py[i, j] = py[i, j] * apy[i, j] - (uy[i, j + 1] - uy[i, j]) * cpy[i, j]


def step_naive(f: Fields, k: Coefficients, w: Workspace) -> None:
    """Reference implementation with explicit Python loops (for testing/benchmarks)."""
    _step_loops(
        f.px, f.py, f.ux, f.uy, k.aux, k.cux, k.auy, k.cuy, k.apx, k.cpx, k.apy, k.cpy
    )


if NUMBA_AVAILABLE:

    @njit(parallel=True, cache=False, fastmath=False)  # type: ignore[misc]
    def _step_numba_impl(px, py, ux, uy, aux, cux, auy, cuy, apx, cpx, apy, cpy):  # type: ignore[no-untyped-def]
        nx, ny = px.shape
        for i in prange(nx - 1):
            for j in range(ny):
                g = (px[i + 1, j] + py[i + 1, j]) - (px[i, j] + py[i, j])
                ux[i + 1, j] = ux[i + 1, j] * aux[i, j] - g * cux[i, j]
        for i in prange(nx):
            for j in range(ny - 1):
                g = (px[i, j + 1] + py[i, j + 1]) - (px[i, j] + py[i, j])
                uy[i, j + 1] = uy[i, j + 1] * auy[i, j] - g * cuy[i, j]
        for i in prange(nx):
            for j in range(ny):
                px[i, j] = px[i, j] * apx[i, j] - (ux[i + 1, j] - ux[i, j]) * cpx[i, j]
                py[i, j] = py[i, j] * apy[i, j] - (uy[i, j + 1] - uy[i, j]) * cpy[i, j]


def step_numba(f: Fields, k: Coefficients, w: Workspace) -> None:
    """Numba-compiled, multithreaded version of :func:`step_naive`."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is not installed; pip install 'hifu-sim[numba]'")
    _step_numba_impl(
        f.px, f.py, f.ux, f.uy, k.aux, k.cux, k.auy, k.cuy, k.apx, k.cpx, k.apy, k.cpy
    )


BACKENDS = ("auto", "numpy", "numba", "naive")


def resolve_backend(name: str) -> str:
    """Map a requested backend to an available one.

    ``"auto"`` picks Numba when installed, else NumPy. Asking for ``"numba"`` without
    Numba installed falls back to NumPy with a :class:`RuntimeWarning`.
    """
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {BACKENDS}")
    if name == "auto":
        return "numba" if NUMBA_AVAILABLE else "numpy"
    if name == "numba" and not NUMBA_AVAILABLE:
        warnings.warn(
            "numba is not installed; falling back to the NumPy kernel",
            RuntimeWarning,
            stacklevel=2,
        )
        return "numpy"
    return name


def get_step_function(backend: str) -> StepFunction:
    resolved = resolve_backend(backend)
    return {"numpy": step_numpy, "numba": step_numba, "naive": step_naive}[resolved]


def numba_threads() -> int:
    """Number of threads Numba will use (0 if Numba is unavailable)."""
    if _numba is None:
        return 0
    return int(_numba.get_num_threads())
