"""Pennes bioheat equation and CEM43 thermal dose.

The temperature rise produced by the acoustic field is computed from

    rho C dT/dt = div(k grad T) - w_b c_b (T - T_a) + Q,      Q = 2 alpha I

where ``alpha`` is the amplitude attenuation [Np/m] and ``I`` the time-averaged
intensity, so ``Q`` [W/m^3] is the absorbed power density (Pennes, 1948). The thermal
dose is expressed in cumulative equivalent minutes at 43 degC (Sapareto & Dewey, 1984)

    CEM43 = sum_n  R^(43 - T_n) dt_n / 60,   R = 0.5 if T >= 43 else 0.25.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .grid import FloatArray, Grid
from .medium import Medium

#: Specific heat of blood [J/(kg K)].
BLOOD_SPECIFIC_HEAT = 3617.0

#: Commonly used lethal-dose threshold for soft tissue [min].
CEM43_THRESHOLD = 240.0


def absorbed_power_density(alpha: FloatArray, intensity: FloatArray) -> FloatArray:
    """Heat source ``Q = 2 alpha I`` [W/m^3] for amplitude attenuation ``alpha`` [Np/m]."""
    q: FloatArray = 2.0 * alpha * intensity
    return q


def cem43_rate(temperature: ArrayLike) -> FloatArray:
    """Dose rate ``R^(43 - T)`` in equivalent minutes per minute."""
    t = np.asarray(temperature, dtype=np.float64)
    r = np.where(t >= 43.0, 0.5, 0.25)
    rate: FloatArray = r ** (43.0 - t)
    return rate


def cem43(temperature_history: ArrayLike, dt: float) -> FloatArray:
    """Thermal dose [min] of a temperature history sampled every ``dt`` seconds.

    The first axis of ``temperature_history`` is time; the result has the remaining
    shape (a scalar array for a 1D history).
    """
    t = np.asarray(temperature_history, dtype=np.float64)
    dose: FloatArray = np.sum(cem43_rate(t), axis=0) * (dt / 60.0)
    return dose


@dataclass
class ThermalResult:
    times: FloatArray  # sample times [s]
    probe_temperature: FloatArray  # temperature at the probe cell vs time [degC]
    probe_index: tuple[int, int]
    temperature_max: FloatArray  # max temperature reached per cell [degC]
    temperature_end_of_heating: FloatArray
    temperature_final: FloatArray
    cem43: FloatArray  # accumulated thermal dose per cell [min]
    dt: float
    heating_time: float


class BioheatSolver:
    """Explicit finite-volume solver for the Pennes equation on the acoustic grid.

    Conductivity at a face is the harmonic mean of its neighbours, which keeps the heat
    flux continuous across material interfaces. The outer boundary is held at the
    initial temperature (body core far from the focus).
    """

    def __init__(
        self,
        grid: Grid,
        medium: Medium,
        arterial_temperature: float = 37.0,
        blood_specific_heat: float = BLOOD_SPECIFIC_HEAT,
    ) -> None:
        self.grid = grid
        self.medium = medium
        self.t_arterial = arterial_temperature
        self.rho_c = medium.density * medium.specific_heat
        self.perf = medium.perfusion * blood_specific_heat  # W/(m^3 K)
        k = medium.thermal_conductivity
        kp = np.pad(k, 1, mode="edge")
        self.kx = 2.0 * kp[1:, 1:-1] * kp[:-1, 1:-1] / (kp[1:, 1:-1] + kp[:-1, 1:-1])
        self.ky = 2.0 * kp[1:-1, 1:] * kp[1:-1, :-1] / (kp[1:-1, 1:] + kp[1:-1, :-1])

    def stable_dt(self, safety: float = 0.9) -> float:
        dx, dy = self.grid.dx, self.grid.spacing_y
        k = self.medium.thermal_conductivity
        denom = 2.0 * k * (1.0 / dx**2 + 1.0 / dy**2) + self.perf
        return float(safety * np.min(self.rho_c / denom))

    def _rhs(self, temp: FloatArray, q: FloatArray, t_boundary: float) -> FloatArray:
        dx, dy = self.grid.dx, self.grid.spacing_y
        tp = np.pad(temp, 1, mode="constant", constant_values=t_boundary)
        fx = self.kx * (tp[1:, 1:-1] - tp[:-1, 1:-1]) / dx  # flux on x faces (nx+1, ny)
        fy = self.ky * (tp[1:-1, 1:] - tp[1:-1, :-1]) / dy
        div = (fx[1:] - fx[:-1]) / dx + (fy[:, 1:] - fy[:, :-1]) / dy
        rhs: FloatArray = (div - self.perf * (temp - self.t_arterial) + q) / self.rho_c
        return rhs

    def simulate(
        self,
        heat_source: FloatArray,
        heating_time: float,
        cooling_time: float = 0.0,
        initial_temperature: float | None = None,
        dt: float | None = None,
        probe: tuple[int, int] | None = None,
        conduction: bool = True,
    ) -> ThermalResult:
        """Heat with ``heat_source`` [W/m^3] for ``heating_time`` s, then cool.

        ``probe`` is the cell whose temperature history is returned; by default the cell
        with the largest heat source. ``conduction=False`` drops the diffusion term
        (useful for analytic checks).
        """
        if heat_source.shape != self.grid.shape:
            raise ValueError("heat source has the wrong shape")
        t0 = self.t_arterial if initial_temperature is None else initial_temperature
        dt = self.stable_dt() if dt is None else dt
        if conduction and dt > self.stable_dt(safety=1.0):
            raise ValueError(f"thermal dt={dt:.3g} s exceeds the explicit stability limit")
        n_heat = int(np.ceil(heating_time / dt - 1e-9))
        n_cool = int(np.ceil(cooling_time / dt - 1e-9))
        if probe is None:
            pi, pj = np.unravel_index(int(np.argmax(heat_source)), heat_source.shape)
            probe = (int(pi), int(pj))

        temp = np.full(self.grid.shape, t0, dtype=np.float64)
        tmax = temp.copy()
        dose = np.zeros(self.grid.shape)
        zero = np.zeros_like(temp)
        times = np.empty(n_heat + n_cool + 1)
        trace = np.empty(n_heat + n_cool + 1)
        times[0], trace[0] = 0.0, t0
        t_heat_end = temp.copy()
        for n in range(n_heat + n_cool):
            q = heat_source if n < n_heat else zero
            if conduction:
                temp = temp + dt * self._rhs(temp, q, t0)
            else:
                temp = temp + dt * (q - self.perf * (temp - self.t_arterial)) / self.rho_c
            np.maximum(tmax, temp, out=tmax)
            dose += cem43_rate(temp) * (dt / 60.0)
            times[n + 1] = (n + 1) * dt
            trace[n + 1] = temp[probe]
            if n == n_heat - 1:
                t_heat_end = temp.copy()
        return ThermalResult(
            times=times,
            probe_temperature=trace,
            probe_index=probe,
            temperature_max=tmax,
            temperature_end_of_heating=t_heat_end,
            temperature_final=temp,
            cem43=dose,
            dt=dt,
            heating_time=n_heat * dt,
        )
