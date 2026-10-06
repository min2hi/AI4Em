"""Consumer boundary tests; only native model loading/inference is replaced."""
from types import SimpleNamespace

import numpy as np
import pytest
from mediapipe.tasks.python import vision

from src.contracts import FramePacket
from src.features.landmarks import FaceLandmarkDetector


def packet(timestamp=0, source="clip-a", image=None):
    if image is None:
        image = np.zeros((2, 3, 3), dtype=np.uint8)
    return FramePacket(image, timestamp, 0, source)


def model_result(faces):
    return vision.FaceLandmarkerResult(
        face_landmarks=faces, face_blendshapes=[], facial_transformation_matrixes=[]
    )


@pytest.fixture
def backend(tmp_path, monkeypatch):
    # Replacing the external model avoids downloaded assets, not detector logic.
    asset = tmp_path / "face.task"
    asset.write_bytes(b"test-only external model placeholder")
    state = SimpleNamespace(
        result=model_result([]), error=None, calls=[], close_count=0, options=None
    )

    class NativeModel:
        def detect_for_video(self, image, timestamp_ms):
            state.calls.append((image.numpy_view().copy(), timestamp_ms))
            if state.error is not None:
                raise state.error
            return state.result

        def close(self):
            state.close_count += 1

    def create(options):
        state.options = options
        return NativeModel()

    monkeypatch.setattr(vision.FaceLandmarker, "create_from_options", create)
    state.config = {
        "asset_path": asset,
        "landmarker_mode": "VIDEO",
        "num_faces": 1,
        "min_face_detection_confidence": 0.41,
        "min_face_presence_confidence": 0.52,
        "min_tracking_confidence": 0.63,
    }
    return state


@pytest.mark.parametrize("timestamp", [-1, 1.5, True, "1", None])
def test_invalid_timestamp_never_reaches_inference(backend, timestamp):
    with FaceLandmarkDetector(backend.config) as detector:
        with pytest.raises(ValueError):
            detector.detect(packet(timestamp))
        assert detector.stats["inference_frames"] == 0
        assert detector.detect(packet(0)).timestamp_ms == 0


@pytest.mark.parametrize("timestamp", [0, 9])
def test_duplicate_and_regressing_time_rejected_without_consuming_next_frame(backend, timestamp):
    with FaceLandmarkDetector(backend.config) as detector:
        detector.detect(packet(10))
        with pytest.raises(ValueError):
            detector.detect(packet(timestamp))
        assert detector.stats["inference_frames"] == 1
        assert detector.detect(packet(11)).timestamp_ms == 11


def test_source_binding_requires_new_detector_for_new_clip(backend):
    with FaceLandmarkDetector(backend.config) as detector:
        detector.detect(packet(0))
        with pytest.raises(ValueError):
            detector.detect(packet(1, "clip-b"))
        assert detector.stats["inference_frames"] == 1
        assert detector.detect(packet(1)).timestamp_ms == 1
    with FaceLandmarkDetector(backend.config) as fresh:
        assert fresh.detect(packet(0, "clip-b")).timestamp_ms == 0


@pytest.mark.parametrize("image", [
    np.zeros((2, 3, 3), dtype=np.float32),
    np.zeros((2, 3), dtype=np.uint8),
    np.zeros((2, 3, 4), dtype=np.uint8),
    np.zeros((0, 3, 3), dtype=np.uint8),
    np.zeros((2, 0, 3), dtype=np.uint8),
    [[[0, 0, 0]]],
])
def test_invalid_images_rejected_before_inference_or_session_binding(backend, image):
    with FaceLandmarkDetector(backend.config) as detector:
        with pytest.raises(ValueError):
            detector.detect(packet(0, "bad-source", image))
        assert detector.stats["inference_frames"] == 0
        assert detector.detect(packet(0)).image_size == (3, 2)


def test_bgr_to_rgb_conversion_does_not_mutate_source_image(backend):
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[:, :] = [10, 20, 30]
    original = image.copy()
    with FaceLandmarkDetector(backend.config) as detector:
        detector.detect(packet(23, image=image))
        np.testing.assert_array_equal(backend.calls[0][0][0, 0], [30, 20, 10])
        np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("case", ["short", "long", "nan", "infinite", "missing-coordinate", "multiple-faces"])
def test_malformed_landmarks_cannot_become_valid_measurements(backend, case):
    points = [SimpleNamespace(x=0.2, y=0.4, z=-0.1) for _ in range(478)]
    if case == "short":
        points.pop()
    elif case == "long":
        points.append(points[0])
    elif case == "nan":
        points[1].y = float("nan")
    elif case == "infinite":
        points[1].z = float("inf")
    elif case == "missing-coordinate":
        points[1] = SimpleNamespace(x=0.2, y=0.4)
    backend.result = model_result([points, points] if case == "multiple-faces" else [points])
    with FaceLandmarkDetector(backend.config) as detector:
        result = detector.detect(packet())
        assert result.points is None
        assert not result.face_detected
        assert result.image_size == (3, 2)
        assert detector.stats["inference_frames"] == 1
        assert detector.stats["invalid_landmark_frames"] == 1
        assert detector.stats["face_frames"] == 0
        assert detector.stats["no_face_frames"] == 0


def test_runtime_failure_propagates_and_close_is_idempotent(backend):
    backend.error = RuntimeError("native inference failed")
    detector = FaceLandmarkDetector(backend.config)
    assert not detector.stats["closed"]
    with pytest.raises(RuntimeError):
        with detector:
            detector.detect(packet())
    detector.close()
    assert backend.close_count == 1
    assert detector.stats["closed"]
    with pytest.raises(RuntimeError):
        detector.detect(packet(1))
    assert detector.stats["face_frames"] == 0


@pytest.mark.parametrize("change", [
    {"landmarker_mode": "IMAGE"}, {"num_faces": 2},
    {"min_face_detection_confidence": float("nan")},
    {"min_face_presence_confidence": -0.1},
    {"min_tracking_confidence": 1.1},
])
def test_invalid_detector_configuration_rejected_at_startup(backend, change):
    with pytest.raises(ValueError):
        FaceLandmarkDetector({**backend.config, **change})
    assert backend.options is None


def test_missing_asset_fails_at_startup(backend, tmp_path):
    with pytest.raises(FileNotFoundError):
        FaceLandmarkDetector({**backend.config, "asset_path": tmp_path / "missing.task"})
    assert backend.options is None
