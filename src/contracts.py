"""Shared records. Missing measurements are NaN + validity flags, not false zeros."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
ImageArray = NDArray[np.uint8]


class DriverState(IntEnum):
    ALERT = 0
    LOW_VIGILANCE = 1
    DROWSY = 2


class SystemStatus(StrEnum):
    IDLE = "IDLE"
    CALIBRATING = "CALIBRATING"
    WARMING_UP = "WARMING_UP"
    READY = "READY"
    UNRELIABLE = "UNRELIABLE"
    NO_FACE = "NO_FACE"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class FramePacket:
    image_bgr: ImageArray
    timestamp_ms: int
    frame_index: int
    source_id: str


@dataclass(frozen=True, slots=True)
class LandmarkResult:
    points: FloatArray | None
    timestamp_ms: int
    face_detected: bool
    image_size: tuple[int, int]


@dataclass(frozen=True, slots=True)
class EyeFeatures:
    ear_left: float
    ear_right: float
    ear_mean: float
    left_eye_valid: bool
    right_eye_valid: bool


@dataclass(frozen=True, slots=True)
class MouthFeatures:
    mar: float
    mouth_valid: bool


@dataclass(frozen=True, slots=True)
class PoseFeatures:
    pitch: float
    yaw: float
    roll: float
    pose_valid: bool
    reprojection_error_norm: float


@dataclass(frozen=True, slots=True)
class FeatureSample:
    timestamp_ms: int
    frame_index: int
    source_id: str
    ear_left: float
    ear_right: float
    ear_mean: float
    mar: float
    pitch: float
    yaw: float
    roll: float
    face_detected: bool
    left_eye_valid: bool
    right_eye_valid: bool
    mouth_valid: bool
    pose_valid: bool
    reprojection_error_norm: float


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    mode: str
    valid: bool
    ear_left_baseline: float
    ear_right_baseline: float
    mar_baseline: float
    pitch_baseline: float
    yaw_baseline: float
    roll_baseline: float
    schema_version: str
    asset_sha256: str
    image_size: tuple[int, int]
    quality_stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TemporalSample:
    timestamp_ms: int
    values: FloatArray
    validity: NDArray[np.bool_]
    segment_id: int
    event_summaries: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SequenceWindow:
    x: FloatArray
    start_timestamp_ms: int
    end_timestamp_ms: int
    quality: dict[str, Any]
    feature_names: tuple[str, ...]
    schema_version: str


@dataclass(frozen=True, slots=True)
class Prediction:
    timestamp_ms: int
    probabilities: FloatArray
    class_id: DriverState | None
    valid: bool
    reason: str
    model_id: str


@dataclass(frozen=True, slots=True)
class DetectionResult:
    raw_prediction: Prediction | None
    smoothed_prediction: Prediction | None
    system_status: SystemStatus
    quality: dict[str, Any]
    calibration_status: str


@dataclass(frozen=True, slots=True)
class AlertDecision:
    level: int
    strong: bool
    audio_command: str | None
    message: str
    timestamp_ms: int


@dataclass(frozen=True, slots=True)
class UiSnapshot:
    preview_rgb: bytes | None
    preview_size: tuple[int, int]
    feature_sample: FeatureSample | None
    temporal_display: dict[str, Any]
    detection: DetectionResult
    alert: AlertDecision
    fps: float
    latency_ms: float
    prediction_age_ms: float | None
