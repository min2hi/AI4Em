from argparse import Namespace

from mediapipe.tasks.python import vision

from scripts.preview_landmarks import run_source


def test_native_model_initialization_failure_retains_source_error_report(tmp_path, monkeypatch, feature_backend):
    config = feature_backend.config

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


def test_preview_keeps_face_presence_but_masks_dark_measurements(feature_backend, tmp_path):
    import cv2
    import numpy as np
    path = tmp_path / "dark.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (640,480))
    assert writer.isOpened()
    for _ in range(2):
        writer.write(np.zeros((480,640,3),np.uint8))
    writer.release()
    args = Namespace(headless=True,save_overlay=False,report_dir=tmp_path,
        camera_width=640,camera_height=480,constant_fps_verified=False,seconds=None)
    report = run_source(path, feature_backend.config, args, 0)
    assert report["status"] == "eof" and report["complete_source_validation"]
    assert report["reader"]["emitted_frames"] == 2 and report["reader"]["capture_released"]
    assert report["detector"]["face_frames"] == 2 and report["detector"]["closed"]
    for name in ("ear_left","ear_right","ear_mean","mar","pitch","yaw","roll"):
        assert report["features"][name]["valid_frames"] == 0
        assert report["features"][name]["mean"] is None


def test_native_close_failure_reports_source_and_continues_next_clip(feature_backend, tmp_path, monkeypatch):
    import cv2
    import json
    import sys
    import yaml
    from scripts.preview_landmarks import main
    b = feature_backend
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),15,(640,480))
    assert writer.isOpened()
    for _ in range(2):
        writer.write(b.image)
    writer.release()
    original = vision.FaceLandmarker.create_from_options
    def create(options):
        native = original(options)
        class WithCloseFailure:
            def detect_for_video(self, image, timestamp):
                return native.detect_for_video(image, timestamp)
            def close(self):
                native.close()
                if b.closed == 1:
                    raise RuntimeError("native close failed")
        return WithCloseFailure()
    monkeypatch.setattr(vision.FaceLandmarker,"create_from_options",create)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(json.loads(json.dumps(b.config,default=str))),encoding="utf-8")
    report_dir = tmp_path / "reports"
    monkeypatch.setattr(sys,"argv",["preview","--config",str(config_path),
        "--video",str(path),"--video",str(path),"--headless","--report-dir",str(report_dir)])
    assert main() == 1
    report = json.loads((report_dir/"preview_report.json").read_text(encoding="utf-8"))
    first,second = report["sources"]
    assert first["status"] == "error" and not first["complete_source_validation"]
    assert second["status"] == "eof" and second["complete_source_validation"]
    assert all(item["reader"]["capture_released"] for item in (first,second))
    assert b.created == b.closed == 2
