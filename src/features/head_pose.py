"""Signed six-point pose in a proper, image-aligned object basis."""
from __future__ import annotations

import math
from numbers import Integral, Real
from pathlib import Path

import cv2
import numpy as np

from src.contracts import LandmarkResult, PoseFeatures

POSE_IDS = (1, 152, 33, 263, 61, 291)


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _image_size(size: object) -> tuple[int, int]:
    if not isinstance(size, tuple) or len(size) != 2 or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v <= 0 for v in size
    ):
        raise ValueError("image_size must be positive integer (W,H)")
    return int(size[0]), int(size[1])


def _numeric_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    # Check the uncoerced elements: float conversion would otherwise hide bools.
    raw = np.asarray(value, dtype=object)
    if raw.shape != shape or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) or not math.isfinite(v)
        for v in raw.flat
    ):
        raise ValueError(f"{name} must have shape {shape} and finite nonboolean numbers")
    return np.array(raw, dtype=np.float64)


def validate_camera_model(camera_model: dict) -> dict:
    """Validate explicit intrinsics and return independent, serializable inputs."""
    if not isinstance(camera_model, dict):
        raise ValueError("camera_model must be a mapping")
    mode = camera_model.get("mode")
    if mode == "approximate":
        if set(camera_model) != {"mode"}:
            raise ValueError("approximate camera_model accepts only mode")
        return {"mode": "approximate"}
    if mode != "calibrated":
        raise ValueError("camera_model mode must be approximate or calibrated")
    if set(camera_model) != {"mode", "reference_size", "matrix", "distortion"}:
        raise ValueError("calibrated camera_model requires reference_size, matrix and distortion")
    size = camera_model["reference_size"]
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("reference_size must be positive integer [W,H]")
    reference_size = _image_size(tuple(size))
    matrix = _numeric_array(camera_model["matrix"], (3, 3), "camera matrix")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.array_equal(matrix[2], (0., 0., 1.)):
        raise ValueError("camera matrix requires fx/fy > 0 and last row (0,0,1)")
    distortion = camera_model["distortion"]
    if not isinstance(distortion, (list, tuple, np.ndarray)) or np.ndim(distortion) != 1 or len(distortion) not in (4, 5, 8, 12, 14):
        raise ValueError("distortion must be a vector of length 4,5,8,12 or 14")
    coefficients = _numeric_array(distortion, (len(distortion),), "distortion")
    return {"mode": "calibrated", "reference_size": list(reference_size),
            "matrix": matrix.tolist(), "distortion": coefficients.tolist()}


def _object_points(path: Path) -> np.ndarray:
    vertices = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("v "):
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError("canonical OBJ vertex must contain three coordinates")
        try:
            vertices.append(tuple(float(v) for v in fields[1:]))
        except ValueError as exc:
            raise ValueError("malformed canonical OBJ vertex") from exc
    if len(vertices) <= max(POSE_IDS):
        raise ValueError("canonical OBJ has insufficient vertices")
    points = np.array([vertices[index] for index in POSE_IDS], dtype=np.float64)
    if not np.isfinite(points).all() or np.linalg.matrix_rank(points - points.mean(axis=0)) != 3:
        raise ValueError("selected canonical vertices must be finite and have rank three")
    # A proper rotation, not the y-only reflection used by the old example.
    points *= (1., -1., -1.)
    points.setflags(write=False)
    return points


def _displayed_angles(rotation: np.ndarray, epsilon: float) -> tuple[float, float, float] | None:
    cy = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    if cy <= epsilon:
        return None
    pitch = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    raw_yaw = math.atan2(-float(rotation[2, 0]), cy)
    roll = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return tuple((float(a) + 180.) % 360. - 180. for a in np.rad2deg((pitch, -raw_yaw, roll)))


def _invalid(error: float = math.nan) -> PoseFeatures:
    return PoseFeatures(math.nan, math.nan, math.nan, False, error)


class HeadPoseEstimator:
    """Stateless ITERATIVE PnP with selected-XY and reprojection validity guards."""

    def __init__(self, canonical_path: Path, camera_model: dict, *,
                 max_reprojection_error_norm: float, epsilon: float):
        self.epsilon = _positive_number(epsilon, "epsilon")
        self.max_reprojection_error_norm = _positive_number(max_reprojection_error_norm, "max_reprojection_error_norm")
        if self.max_reprojection_error_norm > 1:
            raise ValueError("max_reprojection_error_norm cannot exceed one")
        self._camera_model = validate_camera_model(camera_model)
        self.object_points = _object_points(canonical_path)
        self._reference_matrix = None
        self._distortion = np.zeros(5, dtype=np.float64)
        if self._camera_model["mode"] == "calibrated":
            self._reference_matrix = np.array(self._camera_model["matrix"], dtype=np.float64)
            self._reference_matrix.setflags(write=False)
            self._distortion = np.array(self._camera_model["distortion"], dtype=np.float64)
        self._distortion.setflags(write=False)

    def _intrinsics(self, size: tuple[int, int]) -> np.ndarray:
        width, height = size
        if self._reference_matrix is None:
            focal = float(max(width, height))
            return np.array([[focal, 0., width / 2.], [0., focal, height / 2.], [0., 0., 1.]])
        ref_width, ref_height = self._camera_model["reference_size"]
        matrix = self._reference_matrix.copy()
        matrix[0] *= width / ref_width
        matrix[1] *= height / ref_height
        return matrix

    def camera_metadata(self, image_size: tuple[int, int]) -> dict:
        size = _image_size(image_size)
        metadata = {"mode": self._camera_model["mode"],
                    "approximate": self._reference_matrix is None,
                    "image_size": list(size), "matrix": self._intrinsics(size).tolist(),
                    "distortion": self._distortion.tolist()}
        if self._reference_matrix is not None:
            metadata["reference_size"] = list(self._camera_model["reference_size"])
        return metadata

    def estimate(self, result: LandmarkResult) -> PoseFeatures:
        if not result.face_detected or result.points is None:
            return _invalid()
        points = result.points
        if not isinstance(points, np.ndarray) or points.shape != (478, 3) or not np.issubdtype(points.dtype, np.floating):
            raise ValueError("Expected floating landmark array [478,3]")
        size = _image_size(result.image_size)
        xy = np.asarray(points[list(POSE_IDS), :2], dtype=np.float64)
        if not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy > 1):
            return _invalid()
        hull = cv2.convexHull(xy.astype(np.float32))
        if cv2.contourArea(hull) <= self.epsilon:
            return _invalid()
        xy *= size
        matrix = self._intrinsics(size)
        try:
            success, rvec, tvec = cv2.solvePnP(self.object_points, xy, matrix, self._distortion,
                                            flags=cv2.SOLVEPNP_ITERATIVE)
            if not success or rvec is None or tvec is None:
                return _invalid()
            if np.size(rvec) != 3 or np.size(tvec) != 3 or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                return _invalid()
            rotation, _ = cv2.Rodrigues(rvec)
            if not np.isfinite(rotation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0) or not math.isclose(float(np.linalg.det(rotation)), 1., abs_tol=1e-7):
                return _invalid()
            camera_points = self.object_points @ rotation.T + np.asarray(tvec).reshape(1, 3)
            if not np.all(camera_points[:, 2] > 0):
                return _invalid()
            angles = _displayed_angles(rotation, self.epsilon)
            if angles is None:
                return _invalid()
            projected, _ = cv2.projectPoints(self.object_points, rvec, tvec, matrix, self._distortion)
            residual = projected.reshape(6, 2) - xy
            error = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))) / math.hypot(*size))
            if not math.isfinite(error):
                return _invalid()
            if error > self.max_reprojection_error_norm:
                return _invalid(error)
            return PoseFeatures(*angles, True, error)
        except cv2.error:
            return _invalid()
