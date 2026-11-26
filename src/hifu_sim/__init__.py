"""hifu-sim: focused ultrasound in heterogeneous tissue (2D linear FDTD + bioheat)."""

from .analysis import focal_metrics, width_at_level
from .grid import Grid, cfl_limit, cfl_time_step
from .kernels import NUMBA_AVAILABLE
from .medium import MATERIALS, Material, Medium, Region
from .pml import PML
from .solver import CWResult, Simulation
from .source import (
    ArcTransducer,
    PressureSource,
    cw_waveform,
    gaussian_pulse,
    plane_source,
    point_source,
)
from .thermal import BioheatSolver, ThermalResult, absorbed_power_density, cem43, cem43_rate

__version__ = "0.1.0"

__all__ = [
    "MATERIALS",
    "NUMBA_AVAILABLE",
    "PML",
    "ArcTransducer",
    "BioheatSolver",
    "CWResult",
    "Grid",
    "Material",
    "Medium",
    "PressureSource",
    "Region",
    "Simulation",
    "ThermalResult",
    "__version__",
    "absorbed_power_density",
    "cem43",
    "cem43_rate",
    "cfl_limit",
    "cfl_time_step",
    "cw_waveform",
    "focal_metrics",
    "gaussian_pulse",
    "plane_source",
    "point_source",
    "width_at_level",
]
