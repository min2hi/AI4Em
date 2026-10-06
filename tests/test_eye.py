"""Hand-derived geometry catches normalized-distance and anatomical-side bugs."""
import math
from dataclasses import replace

import numpy as np
import pytest

from src.contracts import LandmarkResult
from src.features.eye import EyeFeatureExtractor

LEFT_IDS = (362, 385, 387, 263, 373, 380)
RIGHT_IDS = (33, 160, 158, 133, 153, 144)
LEFT_PIXELS = np.array([(100, 100), (110, 95), (130, 95), (140, 100), (130, 105), (110, 105)], dtype=float)
RIGHT_PIXELS = np.array([(200, 100), (210, 97), (230, 97), (240, 100), (230, 103), (210, 103)], dtype=float)


def eye_fixture(size=(640, 320), scale=1.0, shift=(0, 0)):
    points = np.zeros((478, 3), dtype=np.float32)
    for ids, pixels in ((LEFT_IDS, LEFT_PIXELS), (RIGHT_IDS, RIGHT_PIXELS)):
        points[list(ids), :2] = (pixels * scale + shift) / size
    return LandmarkResult(points, 0, True, size)


@pytest.mark.parametrize("size,scale,shift", [
    ((640, 320), 1, (0, 0)), ((640, 640), 1, (0, 0)),
    ((1280, 640), 2, (0, 0)), ((640, 320), 1, (10, 20)),
])
def test_pixel_geometry_preserves_anatomical_sides_under_image_transforms(size, scale, shift):
    result = eye_fixture(size, scale, shift)
    original = result.points.copy()
    value = EyeFeatureExtractor(1e-6).extract(result)
    assert value.ear_left == pytest.approx(0.25)
    assert value.ear_right == pytest.approx(0.15)
    assert value.ear_mean == pytest.approx(0.20)
    assert value.left_eye_valid and value.right_eye_valid
    np.testing.assert_array_equal(result.points, original)


def test_depth_and_unselected_points_do_not_change_2d_ear():
    result = eye_fixture()
    result.points[:, 2] = np.nan
    result.points[0, :2] = np.inf
    value = EyeFeatureExtractor(1e-6).extract(result)
    assert value.ear_left == pytest.approx(0.25)
    assert value.ear_right == pytest.approx(0.15)
    assert value.ear_mean == pytest.approx(0.20)


@pytest.mark.parametrize("ids,other", [(LEFT_IDS, 0.15), (RIGHT_IDS, 0.25)])
@pytest.mark.parametrize("mutation", ["width_zero", "nan", "inf", "outside_low", "outside_high"])
def test_one_invalid_eye_keeps_other_measurement_but_invalidates_mean(ids, other, mutation):
    result = eye_fixture()
    if mutation == "width_zero":
        result.points[ids[3], :2] = result.points[ids[0], :2]
    else:
        result.points[ids[1], 0] = {"nan": np.nan, "inf": np.inf, "outside_low": -0.01, "outside_high": 1.01}[mutation]
    value = EyeFeatureExtractor(1e-6).extract(result)
    if ids == LEFT_IDS:
        assert math.isnan(value.ear_left) and not value.left_eye_valid
        assert value.ear_right == pytest.approx(other) and value.right_eye_valid
    else:
        assert math.isnan(value.ear_right) and not value.right_eye_valid
        assert value.ear_left == pytest.approx(other) and value.left_eye_valid
    assert math.isnan(value.ear_mean)


def test_closed_eye_zero_is_valid_not_missing():
    result = eye_fixture()
    for ids in (LEFT_IDS, RIGHT_IDS):
        result.points[ids[1], :2] = result.points[ids[5], :2]
        result.points[ids[2], :2] = result.points[ids[4], :2]
    value = EyeFeatureExtractor(1e-6).extract(result)
    assert (value.ear_left, value.ear_right, value.ear_mean) == (0, 0, 0)
    assert value.left_eye_valid and value.right_eye_valid


@pytest.mark.parametrize("face,points", [(False, None), (False, "present"), (True, None)])
def test_missing_face_or_points_returns_unknown(face, points):
    result = eye_fixture()
    result = replace(result, face_detected=face, points=result.points if points == "present" else None)
    value = EyeFeatureExtractor(1e-6).extract(result)
    assert all(math.isnan(v) for v in (value.ear_left, value.ear_right, value.ear_mean))
    assert not value.left_eye_valid and not value.right_eye_valid


@pytest.mark.parametrize("points", [
    np.zeros((477, 3), dtype=np.float32), np.zeros((479, 3), dtype=np.float32),
    np.zeros((478, 2), dtype=np.float32), np.zeros((478, 3), dtype=int),
    np.zeros((478, 3), dtype=object), [[0.0, 0.0, 0.0]] * 478,
])
def test_present_landmarks_with_wrong_schema_fail_clearly(points):
    with pytest.raises(ValueError):
        EyeFeatureExtractor(1e-6).extract(replace(eye_fixture(), points=points))


@pytest.mark.parametrize("size", [(0, 320), (640, -1), (640.5, 320), (True, 320), (640,), [640, 320]])
def test_present_landmarks_require_positive_integer_width_height(size):
    with pytest.raises(ValueError):
        EyeFeatureExtractor(1e-6).extract(replace(eye_fixture(), image_size=size))


@pytest.mark.parametrize("epsilon,valid", [(80.0, False), (79.9, True)])
def test_epsilon_checks_complete_pixel_denominator_including_equality(epsilon, valid):
    value = EyeFeatureExtractor(epsilon).extract(eye_fixture())
    assert value.left_eye_valid is valid and value.right_eye_valid is valid
    if valid:
        assert value.ear_left == pytest.approx(0.25)
    else:
        assert math.isnan(value.ear_left) and math.isnan(value.ear_mean)


@pytest.mark.parametrize("epsilon", [0, -1, math.nan, math.inf, True, False, "1e-6", None])
def test_invalid_epsilon_cannot_create_extractor(epsilon):
    with pytest.raises(ValueError):
        EyeFeatureExtractor(epsilon)
