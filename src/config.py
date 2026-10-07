"""Safe YAML loading with project-relative paths and temporal invariants."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from src.features.head_pose import validate_camera_model
from src.features.quality import validate_quality_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = {"asset_path", "canonical_model_path", "dataset_root", "output_dir", "manifest_path", "model_path"}
POSITIVE_KEYS = {
    "landmark_target_fps", "sequence_fps", "sequence_window_s", "stride_s", "max_gap_s",
    "max_sample_age_ms", "statistics_window_s", "epsilon", "blink_min_s", "blink_max_s",
    "perclos_min_history_s", "yawn_min_s", "camera_width", "camera_height", "prediction_interval_s",
    "stale_ms", "smoothing_samples", "smoothing_expiry_s", "calibration_seconds", "min_valid_seconds",
    "timeout_seconds", "low_enter_duration_s", "low_exit_duration_s", "drowsy_enter_duration_s",
    "drowsy_exit_duration_s", "strong_duration_s", "audio_cooldown_s", "ui_refresh_hz",
    "batch_size", "max_epochs", "learning_rate", "early_stopping_patience", "gradient_clip_norm",
    "hidden_size", "num_layers", "rf_n_estimators", "rf_min_samples_leaf",
    "yawn_enter", "max_reprojection_error_norm", "max_yaw_delta_deg", "max_pitch_delta_deg",
}
RATIO_KEYS = {
    "min_face_detection_confidence", "min_face_presence_confidence", "min_tracking_confidence",
    "blink_enter", "blink_exit", "perclos_threshold", "perclos_min_coverage", "max_missing_ratio",
    "low_enter", "low_exit", "drowsy_enter", "drowsy_exit", "head_dropout",
    "yawn_exit",
}
INTEGER_KEYS = {
    "num_faces", "camera_width", "camera_height", "smoothing_samples", "batch_size",
    "max_epochs", "early_stopping_patience", "hidden_size", "num_layers",
    "rf_n_estimators", "rf_min_samples_leaf",
}


def validate_config(config: dict[str, Any], *, model_metadata: dict[str, Any] | None = None) -> None:
    for key in INTEGER_KEYS:
        if key in config and (isinstance(config[key], bool) or not isinstance(config[key], int)):
            raise ValueError(f"{key} must be an integer")
    for key in POSITIVE_KEYS | RATIO_KEYS:
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        if key in POSITIVE_KEYS and value <= 0:
            raise ValueError(f"{key} must be positive")
        if key in RATIO_KEYS and not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if "max_reprojection_error_norm" in config and config["max_reprojection_error_norm"] > 1:
        raise ValueError("max_reprojection_error_norm cannot exceed one")
    if "camera_model" in config:
        validate_camera_model(config["camera_model"])
    if "quality" in config:
        validate_quality_config(config["quality"])
    for enter, exit_, direction in [
        ("blink_enter", "blink_exit", -1), ("yawn_enter", "yawn_exit", 1),
        ("low_enter", "low_exit", 1), ("drowsy_enter", "drowsy_exit", 1),
    ]:
        if enter in config and exit_ in config and direction * (config[enter] - config[exit_]) <= 0:
            raise ValueError(f"Invalid hysteresis: {enter}, {exit_}")
    for smaller, larger in [("stride_s", "sequence_window_s"), ("blink_min_s", "blink_max_s"),
                            ("min_valid_seconds", "calibration_seconds"), ("calibration_seconds", "timeout_seconds"),
                            ("perclos_min_history_s", "statistics_window_s")]:
        if smaller in config and larger in config and config[smaller] > config[larger]:
            raise ValueError(f"{smaller} cannot exceed {larger}")
    if "sequence_window_s" in config and "sequence_fps" in config:
        steps = config["sequence_window_s"] * config["sequence_fps"]
        if not math.isclose(steps, round(steps)):
            raise ValueError("window * FPS must produce an integer number of steps")
    if "calibration_mode" in config and config["calibration_mode"] not in {"P0", "P1"}:
        raise ValueError("calibration_mode must be P0 or P1")
    if "feature_names" in config:
        names = config["feature_names"]
        if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
            raise ValueError("feature_names must be non-empty unique strings")
    if "num_faces" in config and config["num_faces"] != 1:
        raise ValueError("This project supports one driver per session")
    if "landmarker_mode" in config and config["landmarker_mode"] != "VIDEO":
        raise ValueError("Use the VIDEO Tasks API for the planned pipeline")
    if model_metadata is not None:
        for key in ("calibration_mode", "schema_version", "feature_names", "sequence_fps", "sequence_window_s"):
            if key in config and key in model_metadata and config[key] != model_metadata[key]:
                raise ValueError(f"Model/config mismatch: {key}")


def load_config(path: str | Path, *, project_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root or PROJECT_ROOT).resolve()
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    for key in PATH_KEYS & config.keys():
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a path string")
        value = Path(config[key])
        config[key] = value.resolve() if value.is_absolute() else (root / value).resolve()
    quality = config.get("quality")
    if isinstance(quality, dict) and "frozen_report_path" in quality:
        value = quality["frozen_report_path"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("quality.frozen_report_path must be a path string")
        value = Path(value)
        quality["frozen_report_path"] = value.resolve() if value.is_absolute() else (root / value).resolve()
    validate_config(config)
    return config
