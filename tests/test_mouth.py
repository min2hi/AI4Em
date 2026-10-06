"""Project inner-lip MAR, independent of eye geometry and class labels."""
import math
from dataclasses import replace

import numpy as np
import pytest

from src.contracts import LandmarkResult
from src.features.mouth import MouthFeatureExtractor

MOUTH_IDS = (78, 308, 82, 87, 13, 14, 312, 317)
MOUTH_PIXELS = np.array([(120, 180), (160, 180), (130, 170), (130, 190),
                         (140, 170), (140, 190), (150, 170), (150, 190)], dtype=float)


def mouth_fixture(size=(640, 320), scale=1.0, shift=(0, 0)):
    points = np.zeros((478, 3), dtype=np.float32)
    points[list(MOUTH_IDS), :2] = (MOUTH_PIXELS * scale + shift) / size
    return LandmarkResult(points, 0, True, size)


@pytest.mark.parametrize("size,scale,shift", [
    ((640, 320), 1, (0, 0)), ((640, 640), 1, (0, 0)),
    ((1280, 640), 2, (0, 0)), ((640, 320), 1, (10, 20)),
])
def test_inner_lip_pixel_mar_is_invariant_to_image_transforms(size, scale, shift):
    result = mouth_fixture(size, scale, shift)
    original = result.points.copy()
    value = MouthFeatureExtractor(1e-6).extract(result)
    assert value.mouth_valid
    assert value.mar == pytest.approx(0.5)
    np.testing.assert_array_equal(result.points, original)


def test_depth_and_unselected_points_do_not_affect_inner_lip_mar():
    result = mouth_fixture()
    result.points[:, 2] = np.nan
    result.points[0, :2] = np.inf
    value = MouthFeatureExtractor(1e-6).extract(result)
    assert value.mouth_valid and value.mar == pytest.approx(0.5)


def test_vertical_mean_uses_all_three_pairs_not_only_the_center():
    result = mouth_fixture()
    # Left/middle/right spans 0/10/50 pixels; MAR = 60/(3*40).
    result.points[82, 1] = result.points[87, 1]
    result.points[13, 1] = 175 / 320
    result.points[14, 1] = 185 / 320
    result.points[312, 1] = 155 / 320
    result.points[317, 1] = 205 / 320
    value = MouthFeatureExtractor(1e-6).extract(result)
    assert value.mouth_valid and value.mar == pytest.approx(0.5)


def test_closed_mouth_zero_is_valid_not_missing():
    result = mouth_fixture()
    for upper, lower in ((82, 87), (13, 14), (312, 317)):
        result.points[upper, :2] = result.points[lower, :2]
    value = MouthFeatureExtractor(1e-6).extract(result)
    assert value.mouth_valid and value.mar == 0


def test_open_mouth_ratio_above_one_is_not_clamped():
    result = mouth_fixture()
    for upper, lower in ((82, 87), (13, 14), (312, 317)):
        result.points[upper, 1] = 150 / 320
        result.points[lower, 1] = 210 / 320
    value = MouthFeatureExtractor(1e-6).extract(result)
    assert value.mouth_valid and value.mar == pytest.approx(1.5)


@pytest.mark.parametrize("mutation", ["width_zero", "nan", "inf", "outside_low", "outside_high"])
def test_invalid_selected_geometry_returns_unknown_not_zero(mutation):
    result = mouth_fixture()
    if mutation == "width_zero":
        result.points[308, :2] = result.points[78, :2]
    else:
        result.points[13, 0] = {"nan": np.nan, "inf": np.inf, "outside_low": -0.01, "outside_high": 1.01}[mutation]
    value = MouthFeatureExtractor(1e-6).extract(result)
    assert not value.mouth_valid and math.isnan(value.mar)


@pytest.mark.parametrize("face,points", [(False, None), (False, "present"), (True, None)])
def test_missing_face_or_points_returns_unknown(face, points):
    result = mouth_fixture()
    result = replace(result, face_detected=face, points=result.points if points == "present" else None)
    value = MouthFeatureExtractor(1e-6).extract(result)
    assert not value.mouth_valid and math.isnan(value.mar)


@pytest.mark.parametrize("points", [
    np.zeros((477, 3), dtype=np.float32), np.zeros((479, 3), dtype=np.float32),
    np.zeros((478, 2), dtype=np.float32), np.zeros((478, 3), dtype=int),
    np.zeros((478, 3), dtype=object), [[0.0, 0.0, 0.0]] * 478,
])
def test_present_landmarks_with_wrong_schema_fail_clearly(points):
    with pytest.raises(ValueError):
        MouthFeatureExtractor(1e-6).extract(replace(mouth_fixture(), points=points))


@pytest.mark.parametrize("size", [(0, 320), (640, -1), (640.5, 320), (True, 320), (640,), [640, 320]])
def test_present_landmarks_require_positive_integer_width_height(size):
    with pytest.raises(ValueError):
        MouthFeatureExtractor(1e-6).extract(replace(mouth_fixture(), image_size=size))


@pytest.mark.parametrize("epsilon,valid", [(120.0, False), (119.9, True)])
def test_epsilon_checks_three_times_pixel_width_including_equality(epsilon, valid):
    value = MouthFeatureExtractor(epsilon).extract(mouth_fixture())
    assert value.mouth_valid is valid
    if valid:
        assert value.mar == pytest.approx(0.5)
    else:
        assert math.isnan(value.mar)


@pytest.mark.parametrize("epsilon", [0, -1, math.nan, math.inf, True, False, "1e-6", None])
def test_invalid_epsilon_cannot_create_extractor(epsilon):
    with pytest.raises(ValueError):
        MouthFeatureExtractor(epsilon)
