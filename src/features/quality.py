"""Reproducible face-ROI optics and conservative raw-channel quality gates."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType

import cv2
import numpy as np

from src.contracts import EyeFeatures, FramePacket, LandmarkResult, MouthFeatures, PoseFeatures

METRIC_NAMES = ("brightness", "blur_variance", "left_eye_width_ratio", "right_eye_width_ratio")
POLICY_FIELDS = (
    "policy_version", "reference_max_side", "min_blur_variance", "brightness_min",
    "brightness_max", "min_eye_width_ratio", "max_abs_yaw_deg", "max_abs_pitch_deg",
    "eyes_occluded",
)


def validate_quality_config(quality_config: dict, *, require_frozen: bool = False) -> None:
    """Validate numeric policy and, when supplied, exact frozen evidence bytes.

    Standalone numerical consumers may omit evidence; the integrated pipeline
    explicitly requires it. Paths supplied here must already be resolved by the
    configuration loader when project-relative resolution is needed.
    """
    if not isinstance(quality_config, dict):
        raise ValueError("quality must be a mapping")
    for field in POLICY_FIELDS:
        if field not in quality_config:
            raise ValueError(f"quality.{field} is required")
    if quality_config["policy_version"] != "raw_quality_v1":
        raise ValueError("quality.policy_version must be raw_quality_v1")
    side = quality_config["reference_max_side"]
    if isinstance(side, bool) or not isinstance(side, Integral) or side != 256:
        raise ValueError("quality.reference_max_side must be integer 256")
    for field in POLICY_FIELDS[2:-1]:
        value = quality_config[field]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"quality.{field} must be a finite number")
    if quality_config["min_blur_variance"] < 0:
        raise ValueError("quality.min_blur_variance must be nonnegative")
    if not 0 <= quality_config["brightness_min"] < quality_config["brightness_max"] <= 255:
        raise ValueError("quality brightness bounds must satisfy 0 <= min < max <= 255")
    if not 0 < quality_config["min_eye_width_ratio"] < 1:
        raise ValueError("quality.min_eye_width_ratio must be in (0, 1)")
    for field in ("max_abs_yaw_deg", "max_abs_pitch_deg"):
        if quality_config[field] <= 0:
            raise ValueError(f"quality.{field} must be positive")
    if not isinstance(quality_config["eyes_occluded"], bool):
        raise ValueError("quality.eyes_occluded must be boolean")
    supplied = "frozen_report_path" in quality_config or "frozen_report_sha256" in quality_config
    if not supplied and not require_frozen:
        return
    path = quality_config.get("frozen_report_path")
    sha = quality_config.get("frozen_report_sha256")
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("quality.frozen_report_path is required for frozen evidence")
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("quality.frozen_report_sha256 must be a lowercase SHA256 digest")
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read frozen quality report: {path}") from exc
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("Frozen quality report SHA256 does not match")
    try:
        report = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Frozen quality report must contain valid JSON") from exc
    frozen = report.get("quality") if isinstance(report, dict) else None
    if not isinstance(frozen, dict):
        raise ValueError("Frozen quality report requires a quality mapping")
    expected = {field: quality_config[field] for field in POLICY_FIELDS}
    if set(frozen) != set(POLICY_FIELDS):
        raise ValueError("Frozen quality report must contain only policy fields in quality")
    validate_quality_config(frozen)
    if frozen != expected:
        raise ValueError("Frozen quality report policy does not match quality configuration")


@dataclass(frozen=True, slots=True)
class QualityResult:
    left_eye_valid: bool
    right_eye_valid: bool
    mouth_valid: bool
    pose_valid: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float | None]

    def __post_init__(self) -> None:
        # Freeze the nested container too, without retaining a caller-owned dict.
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "reasons", tuple(self.reasons))


def measure_quality(image_bgr: np.ndarray, result: LandmarkResult) -> dict[str, float | None]:
    """Measure half-open face ROI at a fixed 256px reference maximum side.

    Invalid anatomical bounds yield null metrics. A malformed caller interface
    raises instead. Eye endpoint errors affect that eye alone; width measures
    horizontal geometry, not lid aperture, so closed eyes are not missing eyes.
    """
    metrics = dict.fromkeys(METRIC_NAMES)
    size = result.image_size
    if not isinstance(size, tuple) or len(size) != 2 or any(
        isinstance(v, bool) or not isinstance(v, Integral) or v <= 0 for v in size
    ):
        raise ValueError("Expected positive integer image_size (W,H)")
    width, height = size
    if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8 or image_bgr.shape != (height, width, 3):
        raise ValueError("Expected BGR uint8 image matching image_size")
    if not result.face_detected or result.points is None:
        return metrics
    points = result.points
    if not isinstance(points, np.ndarray) or points.shape != (478, 3) or not np.issubdtype(points.dtype, np.floating):
        raise ValueError("Expected floating landmark array [478,3]")
    bounds = np.asarray(points[[234, 454, 10, 152], :2], dtype=np.float64)
    if not np.isfinite(bounds).all() or np.any(bounds < 0) or np.any(bounds > 1):
        return metrics
    bounds *= (width, height)
    x1, y1 = np.floor(bounds.min(axis=0)).astype(int)
    x2, y2 = np.ceil(bounds.max(axis=0)).astype(int)
    if x2 <= x1 or y2 <= y1:
        return metrics
    gray = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    scale = 256 / max(gray.shape)
    reference = cv2.resize(gray, (max(1, round(gray.shape[1] * scale)),
                                max(1, round(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    metrics["brightness"] = float(reference.mean())
    metrics["blur_variance"] = float(cv2.Laplacian(reference, cv2.CV_64F).var())
    for key, ids in (("left_eye_width_ratio", (362, 263)), ("right_eye_width_ratio", (33, 133))):
        xy = np.asarray(points[list(ids), :2], dtype=np.float64)
        if np.isfinite(xy).all() and np.all(xy >= 0) and np.all(xy <= 1):
            xy *= (width, height)
            metrics[key] = float(np.linalg.norm(xy[0] - xy[1])) / (x2 - x1)
    return metrics


class QualityGate:
    """Validate and snapshot a policy once; evaluate without temporal history."""

    def __init__(self, quality_config: dict):
        validate_quality_config(quality_config)
        self._policy = MappingProxyType({field: quality_config[field] for field in POLICY_FIELDS})

    def evaluate(self, packet: FramePacket, result: LandmarkResult,
                 eyes: EyeFeatures, mouth: MouthFeatures, pose: PoseFeatures) -> QualityResult:
        metrics = measure_quality(packet.image_bgr, result)
        policy = self._policy
        reasons = []
        optical_valid = True
        brightness, blur = metrics["brightness"], metrics["blur_variance"]
        if brightness is None or blur is None:
            optical_valid = False
            reasons.append("no_valid_face_roi")
        else:
            if not policy["brightness_min"] <= brightness <= policy["brightness_max"]:
                optical_valid = False
                reasons.append("brightness_out_of_bounds")
            if blur < policy["min_blur_variance"]:
                optical_valid = False
                reasons.append("blur_below_minimum")
        pose_geometry = bool(pose.pose_valid and all(math.isfinite(value) for value in (
            pose.pitch, pose.yaw, pose.roll, pose.reprojection_error_norm,
        )))
        frontal = pose_geometry
        if not pose_geometry:
            reasons.append("invalid_pose")
        elif abs(pose.yaw) > policy["max_abs_yaw_deg"] or abs(pose.pitch) > policy["max_abs_pitch_deg"]:
            frontal = False
            reasons.append("oblique_pose")
        eye_widths = []
        for side in ("left", "right"):
            width = metrics[f"{side}_eye_width_ratio"]
            valid = width is not None and width >= policy["min_eye_width_ratio"]
            eye_widths.append(valid)
            if not valid:
                reasons.append(f"{side}_eye_width_invalid")
        if policy["eyes_occluded"]:
            reasons.append("eyes_occluded")
        eyes_allowed = optical_valid and frontal and not policy["eyes_occluded"]
        left = bool(eyes_allowed and eye_widths[0] and eyes.left_eye_valid and math.isfinite(eyes.ear_left))
        right = bool(eyes_allowed and eye_widths[1] and eyes.right_eye_valid and math.isfinite(eyes.ear_right))
        mouth_valid = bool(optical_valid and frontal and mouth.mouth_valid and math.isfinite(mouth.mar))
        return QualityResult(left, right, mouth_valid, bool(optical_valid and pose_geometry), tuple(reasons), metrics)
