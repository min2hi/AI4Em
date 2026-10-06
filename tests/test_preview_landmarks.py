from argparse import Namespace

from mediapipe.tasks.python import vision

from scripts.preview_landmarks import run_source


def test_native_model_initialization_failure_retains_source_error_report(tmp_path, monkeypatch):
    asset = tmp_path / "model.task"
    asset.write_bytes(b"external model boundary replaced by failing loader")
    config = {"asset_path": asset, "landmarker_mode": "VIDEO", "num_faces": 1,
              "min_face_detection_confidence": 0.5, "min_face_presence_confidence": 0.5,
              "min_tracking_confidence": 0.5, "landmark_target_fps": 20, "epsilon": 1e-6}

    def fail_to_load(options):
        raise RuntimeError("native model cannot initialize")

    monkeypatch.setattr(vision.FaceLandmarker, "create_from_options", fail_to_load)
    args = Namespace(headless=True, save_overlay=False, report_dir=tmp_path,
                     camera_width=640, camera_height=480, constant_fps_verified=False, seconds=1)
    report = run_source(tmp_path / "input.mp4", config, args, 0)
    assert report["status"] == "error"
    assert report["complete_source_validation"] is False
    assert report["detector"] is None
    assert report["reader"]["decoded_frames"] == 0
