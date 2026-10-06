"""One source-bound MediaPipe Tasks VIDEO inference session."""
from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path
from time import perf_counter

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from src.contracts import FramePacket, LandmarkResult


class FaceLandmarkDetector:
    """Extract real normalized landmarks; create a fresh instance per source.

    Configuration is already resolved by the caller. A missing face or malformed
    landmark set is an invalid result, whereas native inference errors propagate.
    ``inference_frames`` counts completed native calls; ``inference_seconds``
    also includes time spent in failed native calls.
    """

    def __init__(self, config: dict):
        if config.get("landmarker_mode") != "VIDEO":
            raise ValueError("FaceLandmarkDetector requires landmarker_mode=VIDEO")
        num_faces = config.get("num_faces")
        if isinstance(num_faces, bool) or not isinstance(num_faces, Integral) or num_faces != 1:
            raise ValueError("FaceLandmarkDetector requires num_faces=1")
        thresholds = {}
        for name in (
            "min_face_detection_confidence",
            "min_face_presence_confidence",
            "min_tracking_confidence",
        ):
            value = config.get(name)
            if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")
            thresholds[name] = float(value)
        asset_value = config.get("asset_path")
        if not isinstance(asset_value, (str, Path)) or not str(asset_value):
            raise ValueError("asset_path must identify the resolved FaceLandmarker asset")
        asset = Path(asset_value)
        if not asset.is_file():
            raise FileNotFoundError(f"FaceLandmarker asset not found: {asset}")
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(asset)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            **thresholds,
        )
        self._model = vision.FaceLandmarker.create_from_options(options)
        self._closed = False
        self._source_id: str | None = None
        self._timestamp_ms: int | None = None
        self.stats = {
            "inference_frames": 0,
            "face_frames": 0,
            "no_face_frames": 0,
            "invalid_landmark_frames": 0,
            "inference_seconds": 0.0,
            "closed": False,
        }

    def detect(self, packet: FramePacket) -> LandmarkResult:
        if self._closed:
            raise RuntimeError("FaceLandmarkDetector is closed")
        timestamp = packet.timestamp_ms
        if isinstance(timestamp, bool) or not isinstance(timestamp, Integral) or timestamp < 0:
            raise ValueError("timestamp_ms must be a nonnegative integer")
        timestamp = int(timestamp)
        if self._timestamp_ms is not None and timestamp <= self._timestamp_ms:
            raise ValueError("timestamp_ms must strictly increase within a detector session")
        if not isinstance(packet.source_id, str) or not packet.source_id:
            raise ValueError("source_id must be a nonempty string")
        if self._source_id is not None and packet.source_id != self._source_id:
            raise ValueError("A new source requires a fresh FaceLandmarkDetector")
        image = packet.image_bgr
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] == 0
            or image.shape[1] == 0
        ):
            raise ValueError("image_bgr must be a nonempty uint8 HxWx3 array")
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # A native call may consume VIDEO state even if it raises. Never reuse
        # that timestamp or silently switch the source after such a failure.
        self._source_id = packet.source_id
        self._timestamp_ms = timestamp
        started = perf_counter()
        try:
            result = self._model.detect_for_video(mp_image, timestamp)
        finally:
            self.stats["inference_seconds"] += perf_counter() - started
        self.stats["inference_frames"] += 1
        faces = result.face_landmarks
        if len(faces) == 0:
            self.stats["no_face_frames"] += 1
            return LandmarkResult(None, timestamp, False, (width, height))
        points = None
        try:
            if len(faces) == 1 and len(faces[0]) == 478:
                with np.errstate(over="ignore", invalid="ignore"):
                    points = np.fromiter(
                        (coordinate for point in faces[0] for coordinate in (point.x, point.y, point.z)),
                        dtype=np.float32,
                        count=478 * 3,
                    ).reshape(478, 3)
        except (AttributeError, TypeError, ValueError, OverflowError):
            points = None
        if points is None or not np.isfinite(points).all():
            self.stats["invalid_landmark_frames"] += 1
            return LandmarkResult(None, timestamp, False, (width, height))
        self.stats["face_frames"] += 1
        return LandmarkResult(points, timestamp, True, (width, height))

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._model.close()
            self.stats["closed"] = True

    def __enter__(self) -> FaceLandmarkDetector:
        if self._closed:
            raise RuntimeError("FaceLandmarkDetector is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
