"""Post-processing helpers: focal metrics and beam widths."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .grid import FloatArray

if TYPE_CHECKING:
    from .solver import CWResult
    from .source import ArcTransducer


def width_at_level(profile: FloatArray, spacing: float, level: float = 0.5) -> float:
    """Full width of the lobe containing the maximum of ``profile`` at ``level * max``.

    Crossings are located by linear interpolation. For a pressure profile,
    ``level=0.5`` gives the -6 dB width. Returns ``nan`` if the lobe touches an end.
    """
    prof = np.asarray(profile, dtype=np.float64)
    k = int(np.argmax(prof))
    thr = level * prof[k]
    left = k
    while left > 0 and prof[left - 1] >= thr:
        left -= 1
    right = k
    while right < prof.size - 1 and prof[right + 1] >= thr:
        right += 1
    if left == 0 or right == prof.size - 1:
        return float("nan")
    # interpolate the crossing between (left-1, left) and (right, right+1)
    xl = (left - 1) + (thr - prof[left - 1]) / (prof[left] - prof[left - 1])
    xr = right + (prof[right] - thr) / (prof[right] - prof[right + 1])
    return float((xr - xl) * spacing)


def focal_metrics(result: CWResult, transducer: ArcTransducer) -> dict[str, float]:
    """Summary numbers for a focused CW run (SI units unless the key says otherwise)."""
    g = result.grid
    fx, fy = result.peak_location()
    i, j = g.index_of(fx, fy)
    peak = float(result.amplitude[i, j])
    tx, ty = transducer.target
    return {
        "peak_x_m": fx,
        "peak_y_m": fy,
        "target_x_m": tx,
        "target_y_m": ty,
        "axial_offset_m": fx - tx,
        "lateral_offset_m": fy - ty,
        "peak_amplitude_Pa": peak,
        "focal_gain": peak / transducer.amplitude,
        "peak_intensity_W_m2": float(result.intensity[i, j]),
        "peak_positive_Pa": float(np.max(result.peak_positive[result.interior])),
        "peak_negative_Pa": float(np.min(result.peak_negative[result.interior])),
        "fwhm_axial_m": width_at_level(result.amplitude[:, j], g.dx),
        "fwhm_lateral_m": width_at_level(result.amplitude[i, :], g.spacing_y),
        "dt_s": result.dt,
        "n_steps": float(result.n_steps),
        "elapsed_s": result.elapsed,
    }
