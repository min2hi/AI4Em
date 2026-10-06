from pathlib import Path

import pytest

from src.config import load_config, validate_config


def test_paths_resolve_against_project_not_working_directory(tmp_path, monkeypatch):
    config_path = tmp_path / "preprocessing.yaml"
    config_path.write_text("landmark_target_fps: 20\nsequence_fps: 10\nasset_path: models/face.task\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = load_config(config_path, project_root=tmp_path)
    assert config["asset_path"] == tmp_path / "models" / "face.task"


@pytest.mark.parametrize("value", [0, -1, float("nan"), True])
def test_invalid_sampling_rate_is_rejected(value):
    with pytest.raises(ValueError):
        validate_config({"sequence_fps": value})


@pytest.mark.parametrize("config", [
    {"blink_enter": 0.8, "blink_exit": 0.6},
    {"yawn_enter": 0.2, "yawn_exit": 0.3},
    {"low_enter": 0.4, "low_exit": 0.7},
    {"drowsy_enter": 0.4, "drowsy_exit": 0.7},
    {"sequence_window_s": 0.25, "sequence_fps": 10},
    {"sequence_window_s": 5, "stride_s": 6},
    {"min_valid_seconds": 40, "calibration_seconds": 30, "timeout_seconds": 60},
    {"calibration_mode": "P2"},
])
def test_inconsistent_temporal_or_threshold_config_is_rejected(config):
    with pytest.raises(ValueError):
        validate_config(config)


def test_model_mode_mismatch_is_rejected():
    with pytest.raises(ValueError):
        validate_config({"calibration_mode": "P1"}, model_metadata={"calibration_mode": "P0"})


def test_yaml_rejects_unsafe_constructors(tmp_path):
    config_path = tmp_path / "unsafe.yaml"
    config_path.write_text("!!python/object/apply:os.system ['echo unsafe']", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(config_path)
