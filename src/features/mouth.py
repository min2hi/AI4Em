"""Project-defined inner-lip MAR; an open mouth is not a yawn decision."""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

from src.contracts import LandmarkResult, MouthFeatures

MOUTH_HORIZONTAL = (78, 308)
MOUTH_VERTICAL = ((82, 87), (13, 14), (312, 317))
_MOUTH_IDS = MOUTH_HORIZONTAL + tuple(index for pair in MOUTH_VERTICAL for index in pair)


class MouthFeatureExtractor:
    def __init__(self, epsilon: float):
        """Epsilon applies to the complete pixel denominator, three mouth widths."""
        if isinstance(epsilon, bool) or not isinstance(epsilon, Real) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.epsilon = float(epsilon)

    def extract(self, result: LandmarkResult) -> MouthFeatures:
        if not result.face_detected or result.points is None:
            return MouthFeatures(math.nan, False)
        points, size = result.points, result.image_size
        if not isinstance(points, np.ndarray) or points.shape != (478, 3) or not np.issubdtype(points.dtype, np.floating):
            raise ValueError("Expected floating landmark array [478,3]")
        if not isinstance(size, tuple) or len(size) != 2 or any(
            isinstance(v, bool) or not isinstance(v, Integral) or v <= 0 for v in size
        ):
            raise ValueError("image_size must be positive integer (W,H)")
        xy = np.asarray(points[list(_MOUTH_IDS), :2], dtype=np.float64)
        if not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy > 1):
            return MouthFeatures(math.nan, False)
        xy *= size
        denominator = 3.0 * np.linalg.norm(xy[0] - xy[1])
        numerator = sum(np.linalg.norm(xy[a] - xy[b]) for a, b in ((2, 3), (4, 5), (6, 7)))
        if not math.isfinite(denominator) or not math.isfinite(numerator) or denominator <= self.epsilon:
            return MouthFeatures(math.nan, False)
        ratio = numerator / denominator
        valid = math.isfinite(ratio)
        return MouthFeatures(float(ratio) if valid else math.nan, valid)
