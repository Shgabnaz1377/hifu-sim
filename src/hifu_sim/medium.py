"""Material library and heterogeneous medium maps."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np

from .grid import FloatArray, Grid

#: dB per neper (20 / ln 10).
DB_PER_NEPER = 8.685889638065035


@dataclass(frozen=True)
class Material:
    """Acoustic and thermal properties of one tissue type.

    Attenuation follows a power law ``alpha(f) = alpha0 * f_MHz**y`` with ``alpha0`` in
    dB/(cm MHz^y). Thermal properties are used only by the Pennes post-processing.
    """

    name: str
    sound_speed: float  # m/s
    density: float  # kg/m^3
    alpha0: float  # dB/(cm MHz^y)
    alpha_power: float = 1.0
    thermal_conductivity: float = 0.5  # W/(m K)
    specific_heat: float = 3600.0  # J/(kg K)
    perfusion: float = 0.0  # blood perfusion rate, kg/(m^3 s)

    def alpha_np_per_m(self, frequency: float) -> float:
        """Amplitude attenuation coefficient at ``frequency`` [Np/m]."""
        return db_cm_to_np_m(self.alpha0 * (frequency / 1e6) ** self.alpha_power)


def db_cm_to_np_m(alpha_db_cm: float) -> float:
    """Convert an attenuation in dB/cm to Np/m."""
    return alpha_db_cm * 100.0 / DB_PER_NEPER


# Representative values, rounded, of the kind tabulated in Duck (1990) and the
# IT'IS tissue database. They are meant for qualitative studies, not treatment planning.
MATERIALS: dict[str, Material] = {
    "water": Material(
        "water", 1500.0, 1000.0, 0.0022, 2.0,
        thermal_conductivity=0.6, specific_heat=4180.0, perfusion=0.0,
    ),
    "soft_tissue": Material(
        "soft_tissue", 1540.0, 1050.0, 0.6, 1.1,
        thermal_conductivity=0.52, specific_heat=3600.0, perfusion=0.5,
    ),
    "fat": Material(
        "fat", 1450.0, 920.0, 0.5, 1.0,
        thermal_conductivity=0.21, specific_heat=2350.0, perfusion=0.3,
    ),
    "bone": Material(
        "bone", 2800.0, 1850.0, 8.0, 1.0,
        thermal_conductivity=0.32, specific_heat=1300.0, perfusion=0.1,
    ),
}


def get_material(name_or_material: str | Material) -> Material:
    if isinstance(name_or_material, Material):
        return name_or_material
    try:
        return MATERIALS[name_or_material]
    except KeyError:
        known = ", ".join(sorted(MATERIALS))
        raise ValueError(f"unknown material {name_or_material!r}; known: {known}") from None


@dataclass(frozen=True)
class Region:
    """A geometric region filled with one material.

    ``kind`` is one of

    * ``"layer"``: ``x_min <= x < x_max`` across the full lateral extent,
    * ``"rectangle"``: ``x_min <= x < x_max`` and ``y_min <= y < y_max``,
    * ``"circle"``: ``|r - center| <= radius``.
    """

    kind: Literal["layer", "rectangle", "circle"]
    material: str | Material
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    center: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.0

    def mask(self, grid: Grid) -> np.ndarray:
        xx, yy = grid.mesh()
        if self.kind == "layer":
            return (xx >= self.x_min) & (xx < self.x_max)
        if self.kind == "rectangle":
            return (
                (xx >= self.x_min) & (xx < self.x_max) & (yy >= self.y_min) & (yy < self.y_max)
            )
        if self.kind == "circle":
            return (xx - self.center[0]) ** 2 + (yy - self.center[1]) ** 2 <= self.radius**2
        raise ValueError(f"unknown region kind {self.kind!r}")

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Region:
        data = dict(d)
        kind = str(data.pop("kind", data.pop("shape", "")))
        if kind not in ("layer", "rectangle", "circle"):
            raise ValueError(f"region kind must be layer, rectangle or circle, got {kind!r}")
        material = str(data.pop("material"))
        kwargs: dict[str, object] = {}
        for key in ("x_min", "x_max", "y_min", "y_max", "radius"):
            if key in data:
                kwargs[key] = float(data.pop(key))  # type: ignore[arg-type]
        if "center" in data:
            center = cast("Sequence[float]", data.pop("center"))
            cx, cy = center
            kwargs["center"] = (float(cx), float(cy))
        if data:
            raise ValueError(f"unknown region keys: {sorted(data)}")
        return cls(kind=kind, material=material, **kwargs)  # type: ignore[arg-type]


@dataclass
class Medium:
    """Per-cell property maps, each of shape ``grid.shape``.

    ``alpha`` is the amplitude attenuation in Np/m *at the simulation frequency*.
    ``labels`` holds an integer material id per cell (index into ``material_names``).
    """

    sound_speed: FloatArray
    density: FloatArray
    alpha: FloatArray
    thermal_conductivity: FloatArray
    specific_heat: FloatArray
    perfusion: FloatArray
    labels: np.ndarray = field(repr=False)
    material_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        shape = self.sound_speed.shape
        for name in ("density", "alpha", "thermal_conductivity", "specific_heat", "perfusion"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")
        if np.any(self.sound_speed <= 0) or np.any(self.density <= 0):
            raise ValueError("sound speed and density must be positive everywhere")
        if np.any(self.alpha < 0):
            raise ValueError("attenuation must be non-negative")

    @property
    def c_max(self) -> float:
        return float(np.max(self.sound_speed))

    @property
    def c_min(self) -> float:
        return float(np.min(self.sound_speed))

    @property
    def impedance(self) -> FloatArray:
        return self.density * self.sound_speed

    @classmethod
    def homogeneous(
        cls, grid: Grid, material: str | Material = "water", frequency: float = 1e6
    ) -> Medium:
        return cls.from_regions(grid, material, (), frequency)

    @classmethod
    def uniform(
        cls,
        grid: Grid,
        sound_speed: float,
        density: float,
        alpha: float = 0.0,
    ) -> Medium:
        """Uniform acoustic medium with attenuation ``alpha`` given directly in Np/m."""
        full = np.full(grid.shape, 1.0)
        return cls(
            sound_speed=full * sound_speed,
            density=full * density,
            alpha=full * alpha,
            thermal_conductivity=full * 0.6,
            specific_heat=full * 4180.0,
            perfusion=full * 0.0,
            labels=np.zeros(grid.shape, dtype=np.int16),
            material_names=["custom"],
        )

    @classmethod
    def from_regions(
        cls,
        grid: Grid,
        background: str | Material,
        regions: Iterable[Region],
        frequency: float,
    ) -> Medium:
        """Paint ``regions`` (later ones on top) over a ``background`` material."""
        mats = [get_material(background)]
        labels = np.zeros(grid.shape, dtype=np.int16)
        for region in regions:
            mat = get_material(region.material)
            if mat not in mats:
                mats.append(mat)
            labels[region.mask(grid)] = mats.index(mat)

        def table(attr: str) -> FloatArray:
            values = np.array([getattr(m, attr) for m in mats], dtype=np.float64)
            return values[labels]

        alpha_values = np.array([m.alpha_np_per_m(frequency) for m in mats])
        return cls(
            sound_speed=table("sound_speed"),
            density=table("density"),
            alpha=alpha_values[labels],
            thermal_conductivity=table("thermal_conductivity"),
            specific_heat=table("specific_heat"),
            perfusion=table("perfusion"),
            labels=labels,
            material_names=[m.name for m in mats],
        )
