"""Pixel-space EAR for anatomical eyes; geometry validity is not pose quality."""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

from src.contracts import EyeFeatures, LandmarkResult

LEFT_EYE = (362, 385, 387, 263, 373, 380)
RIGHT_EYE = (33, 160, 158, 133, 153, 144)


class EyeFeatureExtractor:
    def __init__(self, epsilon: float):
        """Epsilon applies to the complete pixel denominator, twice eye width."""
        if isinstance(epsilon, bool) or not isinstance(epsilon, Real) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.epsilon = float(epsilon)

    def extract(self, result: LandmarkResult) -> EyeFeatures:
        if not result.face_detected or result.points is None:
            return EyeFeatures(math.nan, math.nan, math.nan, False, False)
        points, size = result.points, result.image_size
        if not isinstance(points, np.ndarray) or points.shape != (478, 3) or not np.issubdtype(points.dtype, np.floating):
            raise ValueError("Expected floating landmark array [478,3]")
        if not isinstance(size, tuple) or len(size) != 2 or any(
            isinstance(v, bool) or not isinstance(v, Integral) or v <= 0 for v in size
        ):
            raise ValueError("image_size must be positive integer (W,H)")
        values = []
        for ids in (LEFT_EYE, RIGHT_EYE):
            # Advanced indexing owns this small array; never scale source points.
            xy = np.asarray(points[list(ids), :2], dtype=np.float64)
            if not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy > 1):
                values.append(math.nan)
                continue
            xy *= size
            denominator = 2.0 * np.linalg.norm(xy[0] - xy[3])
            numerator = np.linalg.norm(xy[1] - xy[5]) + np.linalg.norm(xy[2] - xy[4])
            if not math.isfinite(denominator) or not math.isfinite(numerator) or denominator <= self.epsilon:
                values.append(math.nan)
                continue
            ratio = numerator / denominator
            values.append(float(ratio) if math.isfinite(ratio) else math.nan)
        left, right = values
        left_valid, right_valid = math.isfinite(left), math.isfinite(right)
        mean = (left + right) / 2 if left_valid and right_valid else math.nan
        return EyeFeatures(left, right, mean, left_valid, right_valid)
