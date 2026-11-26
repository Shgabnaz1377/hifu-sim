"""YAML configs and the command-line interface."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from hifu_sim.cli import main
from hifu_sim.config import RunConfig, build, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

TINY = {
    "name": "tiny",
    "grid": {"size": [0.017, 0.016], "spacing": 2.0e-4},
    "pml": {"thickness": 12},
    "medium": {
        "background": "water",
        "regions": [{"kind": "layer", "material": "soft_tissue", "x_min": 0.007, "x_max": 1.0}],
    },
    "source": {
        "frequency": 5.0e5, "amplitude": 3.0e5, "radius": 0.010, "aperture_deg": 60,
        "n_elements": 8, "focus": [0.0135, 0.008],
    },
    "simulation": {"cfl": 0.3, "record_cycles": 2, "backend": "numpy"},
    "thermal": {"enabled": True, "heating_time": 0.5, "cooling_time": 0.2},
}


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_example_configs_build(path):
    cfg = load_config(path)
    setup = build(cfg, backend="numpy")
    assert setup.grid.nx > 100
    assert setup.simulation.sources


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown"):
        RunConfig.from_dict({"grid": {"size": [0.01, 0.01], "spacnig": 1e-4}})
    with pytest.raises(ValueError, match="unknown top-level"):
        RunConfig.from_dict({"sauce": {}})


def test_cli_run_writes_outputs(tmp_path, capsys):
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(TINY), encoding="utf-8")
    out = tmp_path / "out"
    assert main(["run", str(cfg_path), "-o", str(out), "-q"]) == 0
    for name in ("tiny.npz", "tiny_metrics.json", "tiny_acoustic.png", "tiny_profiles.png",
                 "tiny_thermal.png"):
        assert (out / name).is_file(), name
    data = np.load(out / "tiny.npz")
    assert data["pressure_amplitude"].shape == data["sound_speed"].shape
    assert data["temperature_max"].max() > 37.0
    metrics = json.loads((out / "tiny_metrics.json").read_text())["metrics"]
    assert metrics["peak_amplitude_Pa"] > 3.0e5
    assert "peak |p|" in capsys.readouterr().out


def test_cli_info(capsys):
    assert main(["info"]) == 0
    assert "soft_tissue" in capsys.readouterr().out


def test_cli_reports_bad_config(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("grid: {size: [0.01, 0.01], spacing: -1}\n", encoding="utf-8")
    assert main(["run", str(bad), "-o", str(tmp_path)]) == 2
    assert "error" in capsys.readouterr().err
