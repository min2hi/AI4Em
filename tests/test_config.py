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


def calibrated_camera():
    return {"mode": "calibrated", "reference_size": [640, 480],
            "matrix": [[800, 0, 320], [0, 800, 240], [0, 0, 1]],
            "distortion": [0, 0, 0, 0, 0]}


@pytest.mark.parametrize("camera", [
    None, [], {}, {"mode": "unknown"}, {"mode": True},
    {"mode": "approximate", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
    {"mode": "approximate", "distortion": []},
    {"mode": "approximate", "reference_size": [640, 480]},
    {"mode": "calibrated"},
])
def test_invalid_camera_modes_and_contradictions(camera):
    with pytest.raises(ValueError):
        validate_config({"camera_model": camera})


@pytest.mark.parametrize("field,value", [
    ("reference_size", [True, 480]), ("reference_size", [640., 480]),
    ("reference_size", [640, 0]), ("reference_size", [640]),
    ("matrix", [[800, 0, 320], [0, 800, 240]]),
    ("matrix", [[True, 0, 320], [0, 800, 240], [0, 0, 1]]),
    ("matrix", [[0, 0, 320], [0, 800, 240], [0, 0, 1]]),
    ("matrix", [[800, 0, 320], [0, -800, 240], [0, 0, 1]]),
    ("matrix", [[800, 0, 320], [0, 800, 240], [0, 1, 1]]),
    ("matrix", [[800, 0, float("nan")], [0, 800, 240], [0, 0, 1]]),
    ("distortion", [0, 0, 0]), ("distortion", [0, 0, True, 0]),
    ("distortion", [0, 0, float("inf"), 0]),
    ("distortion", [[0, 0, 0, 0]]),
])
def test_calibrated_camera_fields_rejected(field, value):
    camera = calibrated_camera()
    camera[field] = value
    with pytest.raises(ValueError):
        validate_config({"camera_model": camera})


@pytest.mark.parametrize("length", [4, 5, 8, 12, 14])
def test_supported_calibrated_distortion_lengths(length):
    camera = calibrated_camera()
    camera["distortion"] = [0.] * length
    validate_config({"camera_model": camera, "max_reprojection_error_norm": 1})


@pytest.mark.parametrize("value", [1.01, True, float("nan"), 0, -1])
def test_reprojection_limit_is_positive_at_most_one(value):
    with pytest.raises(ValueError):
        validate_config({"max_reprojection_error_norm": value})


def test_nested_profile_resolves_from_project_root_not_shell_cwd(feature_backend, tmp_path, monkeypatch):
    import yaml
    from src.config import load_config
    quality = feature_backend.config["quality"].copy()
    quality["frozen_report_path"] = quality["frozen_report_path"].name
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"quality": quality}), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    loaded = load_config(path, project_root=tmp_path)
    assert loaded["quality"]["frozen_report_path"] == feature_backend.config["quality"]["frozen_report_path"]


@pytest.mark.parametrize("field,value", [("brightness_min", True), ("brightness_max", 256),
    ("min_eye_width_ratio", 0), ("reference_max_side", 512), ("eyes_occluded", 1)])
def test_nested_quality_rejects_invalid_policy(feature_backend, field, value):
    quality = feature_backend.config["quality"].copy()
    quality[field] = value
    with pytest.raises(ValueError):
        validate_config({"quality": quality})
