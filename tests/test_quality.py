import numpy as np
import pytest
import hashlib
import json
from dataclasses import FrozenInstanceError

from src.contracts import EyeFeatures, FramePacket, LandmarkResult, MouthFeatures, PoseFeatures
from src.features.quality import QualityGate, QualityResult, measure_quality, validate_quality_config

def test_uniform_light_and_independent_eye_widths():
    points = np.zeros((478,3),np.float32)
    for index,xy in ((234,(.25,.5)),(454,(.75,.5)),(10,(.5,.25)),(152,(.5,.75)),
                     (362,(.5,.375)),(263,(.625,.375)),(33,(.25,.375)),(133,(.375,.375))):
        points[index,:2] = xy
    image = np.full((480,640,3),80,np.uint8)
    result = LandmarkResult(points,0,True,(640,480))
    value = measure_quality(image,result)
    assert value['brightness'] == 80
    assert value['blur_variance'] == 0
    assert value['left_eye_width_ratio'] == pytest.approx(.25)
    assert value['right_eye_width_ratio'] == pytest.approx(.25)
    points[362,0] = np.nan
    value = measure_quality(image,result)
    assert value['left_eye_width_ratio'] is None
    assert value['right_eye_width_ratio'] == pytest.approx(.25)


def face_points():
    points = np.zeros((478, 3), np.float32)
    for index, xy in (
        (234, (.25, .5)), (454, (.75, .5)), (10, (.5, .25)), (152, (.5, .75)),
        (362, (.5, .375)), (263, (.625, .375)), (33, (.25, .375)), (133, (.375, .375)),
    ):
        points[index, :2] = xy
    return points


def policy(**changes):
    value = {
        "policy_version": "raw_quality_v1", "reference_max_side": 256,
        "min_blur_variance": 10., "brightness_min": 40., "brightness_max": 200.,
        "min_eye_width_ratio": .05, "max_abs_yaw_deg": 35.,
        "max_abs_pitch_deg": 25., "eyes_occluded": False,
    }
    value.update(changes)
    return value


def textured_image():
    # Two-pixel stripes survive the metric's downsampling, unlike a 1px pattern.
    row = np.where((np.arange(640) // 2) % 2, 160, 80).astype(np.uint8)
    return np.broadcast_to(row[None, :, None], (480, 640, 3)).copy()


def evaluate(config=None, *, image=None, points=None, eyes=None, mouth=None, pose=None):
    image = textured_image() if image is None else image
    result = LandmarkResult(face_points() if points is None else points, 123, True, (640, 480))
    packet = FramePacket(image, 123, 7, "quality-fixture")
    return QualityGate(config or policy()).evaluate(
        packet, result, eyes or EyeFeatures(0., 0., 0., True, True),
        mouth or MouthFeatures(0., True), pose or PoseFeatures(0., 0., 0., True, .001),
    )


@pytest.mark.parametrize("brightness", [0, 80, 255])
def test_uniform_optical_metrics(brightness):
    value = measure_quality(
        np.full((480, 640, 3), brightness, np.uint8),
        LandmarkResult(face_points(), 0, True, (640, 480)),
    )
    assert value["brightness"] == brightness
    assert value["blur_variance"] == 0.


@pytest.mark.parametrize("bounds", [np.nan, -0.01, 1.01])
def test_invalid_face_roi_yields_null_metrics(bounds):
    points = face_points()
    points[234, 0] = bounds
    value = measure_quality(textured_image(), LandmarkResult(points, 0, True, (640, 480)))
    assert all(item is None for item in value.values())
    assert json.loads(json.dumps(value, allow_nan=False)) == value


def test_empty_and_missing_roi_yield_null_metrics():
    for result in (
        LandmarkResult(None, 0, False, (640, 480)),
        LandmarkResult(np.zeros((478, 3), np.float32), 0, True, (640, 480)),
    ):
        assert all(value is None for value in measure_quality(textured_image(), result).values())


@pytest.mark.parametrize("image,size", [
    (np.zeros((480, 640), np.uint8), (640, 480)),
    (np.zeros((480, 640, 3), np.float32), (640, 480)),
    (np.zeros((480, 640, 3), np.uint8), (480, 640)),
    (np.zeros((480, 640, 3), np.uint8), (True, 480)),
])
def test_metric_interface_mismatches_raise(image, size):
    with pytest.raises(ValueError):
        measure_quality(image, LandmarkResult(face_points(), 0, True, size))


def test_metric_and_gate_do_not_modify_inputs():
    image, points = textured_image(), face_points()
    original_image, original_points = image.copy(), points.copy()
    evaluate(image=image, points=points)
    np.testing.assert_array_equal(image, original_image)
    np.testing.assert_array_equal(points, original_points)


def test_zero_aperture_measurements_are_valid():
    value = evaluate()
    assert (value.left_eye_valid, value.right_eye_valid, value.mouth_valid, value.pose_valid) == (
        True, True, True, True,
    )
    assert not value.reasons


def test_quality_boundaries_accept_equality():
    image = textured_image()
    measured = measure_quality(image, LandmarkResult(face_points(), 0, True, (640, 480)))
    value = evaluate(policy(
        min_blur_variance=measured["blur_variance"],
        brightness_min=measured["brightness"], min_eye_width_ratio=.25,
    ), pose=PoseFeatures(25., -35., 0., True, .001))
    assert value.left_eye_valid and value.right_eye_valid and value.mouth_valid and value.pose_valid
    value = evaluate(policy(brightness_max=measured["brightness"]))
    assert value.pose_valid


@pytest.mark.parametrize("field,amount", [
    ("min_blur_variance", .001), ("brightness_min", .001),
    ("brightness_max", -.001),
])
def test_optical_rejection_invalidates_every_channel(field, amount):
    measured = measure_quality(textured_image(), LandmarkResult(face_points(), 0, True, (640, 480)))
    key = "blur_variance" if field == "min_blur_variance" else "brightness"
    value = evaluate(policy(**{field: measured[key] + amount}))
    assert not any((value.left_eye_valid, value.right_eye_valid, value.mouth_valid, value.pose_valid))
    assert value.reasons
    assert value.metrics["left_eye_width_ratio"] == .25


def test_left_width_failure_keeps_right_mouth_and_pose():
    points = face_points()
    points[362, :2] = points[263, :2]
    value = evaluate(points=points)
    assert not value.left_eye_valid
    assert value.right_eye_valid and value.mouth_valid and value.pose_valid


def test_unknown_width_is_independent():
    points = face_points()
    points[362, 0] = np.nan
    value = evaluate(points=points)
    assert not value.left_eye_valid
    assert value.right_eye_valid and value.mouth_valid and value.pose_valid


def test_manual_eye_occlusion_preserves_other_channels():
    value = evaluate(policy(eyes_occluded=True))
    assert not value.left_eye_valid and not value.right_eye_valid
    assert value.mouth_valid and value.pose_valid


@pytest.mark.parametrize("pitch,yaw", [(0., 36.), (0., -36.), (26., 0.), (-26., 0.)])
def test_oblique_pose_preserves_pose_but_rejects_eye_and_mouth(pitch, yaw):
    value = evaluate(pose=PoseFeatures(pitch, yaw, 0., True, .001))
    assert not value.left_eye_valid and not value.right_eye_valid and not value.mouth_valid
    assert value.pose_valid


@pytest.mark.parametrize("pose", [
    PoseFeatures(np.nan, np.nan, np.nan, False, .5),
    PoseFeatures(0., np.nan, 0., True, .001),
])
def test_invalid_pose_invalidates_every_channel(pose):
    value = evaluate(pose=pose)
    assert not any((value.left_eye_valid, value.right_eye_valid, value.mouth_valid, value.pose_valid))


def test_invalid_geometry_cannot_be_restored_by_gate():
    value = evaluate(eyes=EyeFeatures(np.nan, 0., np.nan, False, True),
                     mouth=MouthFeatures(np.nan, False))
    assert not value.left_eye_valid and value.right_eye_valid
    assert not value.mouth_valid and value.pose_valid


def test_no_face_gate_returns_no_valid_channels():
    packet = FramePacket(textured_image(), 0, 0, "blank")
    value = QualityGate(policy()).evaluate(
        packet, LandmarkResult(None, 0, False, (640, 480)),
        EyeFeatures(np.nan, np.nan, np.nan, False, False),
        MouthFeatures(np.nan, False), PoseFeatures(np.nan, np.nan, np.nan, False, np.nan),
    )
    assert not any((value.left_eye_valid, value.right_eye_valid, value.mouth_valid, value.pose_valid))
    assert all(metric is None for metric in value.metrics.values())


def test_result_and_policy_are_immutable_snapshots():
    config = policy()
    gate = QualityGate(config)
    config["eyes_occluded"] = True
    result = LandmarkResult(face_points(), 0, True, (640, 480))
    value = gate.evaluate(FramePacket(textured_image(), 0, 0, "test"), result,
                          EyeFeatures(0., 0., 0., True, True), MouthFeatures(0., True),
                          PoseFeatures(0., 0., 0., True, .001))
    assert value.left_eye_valid
    with pytest.raises(FrozenInstanceError):
        value.pose_valid = False
    with pytest.raises(TypeError):
        value.metrics["brightness"] = 0.
    metrics = {"brightness": 80.}
    copied = QualityResult(True, True, True, True, (), metrics)
    metrics["brightness"] = 0.
    assert copied.metrics["brightness"] == 80.


@pytest.mark.parametrize("name,value", [
    ("min_blur_variance", True), ("min_blur_variance", -1),
    ("brightness_min", np.nan), ("brightness_max", 256),
    ("brightness_min", 200), ("min_eye_width_ratio", 0),
    ("min_eye_width_ratio", 1), ("max_abs_yaw_deg", 0),
    ("max_abs_pitch_deg", np.inf), ("reference_max_side", 128),
    ("reference_max_side", True), ("eyes_occluded", 0),
    ("policy_version", "unknown"),
])
def test_invalid_quality_policy_fails_at_startup(name, value):
    with pytest.raises(ValueError):
        QualityGate(policy(**{name: value}))


def test_frozen_report_requires_matching_hash_and_exact_thresholds(tmp_path):
    path = tmp_path / "accepted.json"
    path.write_text(json.dumps({"quality": policy(), "annotations": []}), encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    config = policy(frozen_report_path=path, frozen_report_sha256=sha)
    validate_quality_config(config, require_frozen=True)
    QualityGate(config)
    with pytest.raises(ValueError):
        QualityGate({**config, "min_blur_variance": 11.})
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        QualityGate(config)


def test_frozen_report_presence_is_required_when_requested():
    with pytest.raises(ValueError):
        validate_quality_config(policy(), require_frozen=True)
    with pytest.raises(ValueError):
        QualityGate(policy(frozen_report_sha256="0" * 64))


def measurement_report():
    return {
        "program": "scripts.measure_quality", "report_version": "quality_measurements_v1",
        "sources": [{
            "video_id": "04_0", "source_sha256": "a" * 64, "status": "bounded_window",
            "samples": [
                {"video_id": "04_0", "timestamp_ms": timestamp, "frame_index": index,
                 "metrics": {"brightness": brightness, "blur_variance": blur,
                             "left_eye_width_ratio": .2, "right_eye_width_ratio": .25},
                 "controls": [
                     {"kind": "dark", "metrics": {"brightness": 0., "blur_variance": 0.,
                                                 "left_eye_width_ratio": .2, "right_eye_width_ratio": .25}},
                     {"kind": "overexposed", "metrics": {"brightness": 255., "blur_variance": 0.,
                                                        "left_eye_width_ratio": .2, "right_eye_width_ratio": .25}},
                     {"kind": "gaussian_blur", "metrics": {"brightness": brightness, "blur_variance": 1.,
                                                          "left_eye_width_ratio": .2, "right_eye_width_ratio": .25}},
                 ]}
                for index, (timestamp, brightness, blur) in enumerate(((0, 80., 50.), (50, 100., 60.)))
            ],
        }],
    }


def clean_annotations():
    return [
        {"video_id": "04_0", "timestamp_ms": timestamp, "quality": "clean",
         "note": "Visually reviewed: frontal, eyes unobstructed, adequately lit."}
        for timestamp in (0, 50)
    ]


def test_proposal_uses_clean_observed_envelope_and_rejects_controls():
    from scripts.measure_quality import propose_policy
    result = propose_policy(measurement_report(), clean_annotations())
    assert result["quality"] == policy(
        min_blur_variance=50., brightness_min=80., brightness_max=100., min_eye_width_ratio=.2,
    )
    assert all(item["retained"] for item in result["evidence"]["clean"])
    assert not any(item["retained"] for item in result["evidence"]["controls"])
    assert result["measurements"]["sources"][0]["source_sha256"] == "a" * 64


@pytest.mark.parametrize("change", ["unknown", "duplicate", "blank_note", "non_integer"])
def test_proposal_rejects_annotations_that_do_not_resolve_to_reviewed_samples(change):
    from scripts.measure_quality import propose_policy
    annotations = clean_annotations()
    if change == "unknown":
        annotations[0]["timestamp_ms"] = 1
    elif change == "duplicate":
        annotations.append(annotations[0].copy())
    elif change == "blank_note":
        annotations[0]["note"] = " "
    else:
        annotations[0]["timestamp_ms"] = True
    with pytest.raises(ValueError):
        propose_policy(measurement_report(), annotations)


@pytest.mark.parametrize("change", ["flat_brightness", "zero_blur", "missing_width", "negative_overlap"])
def test_proposal_refuses_degenerate_or_overlapping_evidence(change):
    from scripts.measure_quality import propose_policy
    report = measurement_report()
    samples = report["sources"][0]["samples"]
    if change == "flat_brightness":
        samples[1]["metrics"]["brightness"] = 80.
    elif change == "zero_blur":
        samples[0]["metrics"]["blur_variance"] = 0.
    elif change == "missing_width":
        samples[0]["metrics"]["left_eye_width_ratio"] = None
    else:
        samples[0]["controls"][2]["metrics"] = samples[0]["metrics"].copy()
    with pytest.raises(ValueError):
        propose_policy(report, clean_annotations())


def test_measurement_cli_uses_source_window_and_emits_only_json(tmp_path, feature_backend):
    import cv2
    import yaml
    from scripts.measure_quality import main

    source_dir = tmp_path / "04"
    source_dir.mkdir()
    source = source_dir / "0.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 15, (64, 48))
    assert writer.isOpened()
    for intensity in (80, 100, 120, 140):
        writer.write(np.full((48, 64, 3), intensity, np.uint8))
    writer.release()
    config = {key: value for key, value in feature_backend.config.items() if key != "quality"}
    config["asset_path"] = str(config["asset_path"])
    config["canonical_model_path"] = str(config["canonical_model_path"])
    config_path = tmp_path / "measure.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    destination = tmp_path / "reports"
    assert main(["--config", str(config_path), "--video", str(source), "--start-ms", "50",
                 "--end-ms", "150", "--report-dir", str(destination)]) == 0
    report = json.loads((destination / "quality_measurements.json").read_text(encoding="utf-8"))
    samples = report["sources"][0]["samples"]
    assert [sample["timestamp_ms"] for sample in samples] == [67, 133]
    assert [sample["frame_index"] for sample in samples] == [1, 2]
    assert [sample["metrics"]["brightness"] for sample in samples] == [100., 120.]
    assert all(sample["video_id"] == "04_0" for sample in samples)
    assert all(len(sample["controls"]) == 3 for sample in samples)
    assert report["sources"][0]["status"] == "bounded_window"
    assert not report["sources"][0]["complete_source_validation"]
    assert report["sources"][0]["reader"]["capture_released"]
    assert feature_backend.closed == 1
    assert [path.name for path in destination.iterdir()] == ["quality_measurements.json"]


def test_half_open_floor_ceil_roi_excludes_surrounding_darkness():
    points = face_points()
    for index, xy in ((234, (.2, .5)), (454, (.8, .5)), (10, (.5, .2)), (152, (.5, .8))):
        points[index, :2] = xy
    image = np.zeros((8, 8, 3), np.uint8)
    image[1:7, 1:7] = 80
    measured = measure_quality(image, LandmarkResult(points, 0, True, (8, 8)))
    assert measured["brightness"] == 80.
    assert measured["blur_variance"] == 0.
    assert measured["left_eye_width_ratio"] == pytest.approx(1 / 6)


def test_reference_sized_checkerboard_has_literal_laplacian_variance():
    points = face_points()
    for index, xy in ((234, (0, .5)), (454, (1, .5)), (10, (.5, 0)), (152, (.5, 1))):
        points[index, :2] = xy
    gray = ((np.indices((256, 256)).sum(axis=0) % 2) * 255).astype(np.uint8)
    image = np.repeat(gray[:, :, None], 3, axis=2)
    measured = measure_quality(image, LandmarkResult(points, 0, True, (256, 256)))
    assert measured["brightness"] == 127.5
    assert measured["blur_variance"] == 1040400.
