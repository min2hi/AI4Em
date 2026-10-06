"""Phase 0: fetch official assets and exercise actual Tasks/PyTorch/Qt APIs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import requests

from src.config import PROJECT_ROOT, load_config, validate_config

ASSETS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "canonical_face_model.obj": "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj",
}


def download_assets() -> list[dict]:
    directory = PROJECT_ROOT / "models/assets"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for name, url in ASSETS.items():
        destination = directory / name
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".part")
            with requests.get(url, stream=True, timeout=(15,60)) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(1024*1024):
                        handle.write(chunk)
            if temporary.stat().st_size == 0:
                raise ValueError(f"Empty model asset: {name}")
            temporary.replace(destination)
        sha = hashlib.sha256(destination.read_bytes()).hexdigest()
        record = {"filename":name, "source_url":url, "sha256":sha, "size_bytes":destination.stat().st_size,
                  "checked_at_utc":datetime.now(timezone.utc).isoformat(),
                  "license_reference":"https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE",
                  "note":"Source/hash recorded; dataset rights are separate from model repository license."}
        (directory / (name + ".metadata.json")).write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-assets", action="store_true")
    parser.add_argument("--assets-only", action="store_true")
    parser.add_argument("--sample-video", type=Path, help="Local research video, not uploaded or shown in public artifacts")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "runs/phase0")
    args = parser.parse_args()
    if platform.python_version_tuple()[:2] != ("3","12") or platform.architecture()[0] != "64bit":
        raise RuntimeError("Use the project Python 3.12 x64 virtual environment")
    if args.download_assets:
        assets = download_assets()
        print(json.dumps({"assets":assets}, indent=2))
    if args.assets_only:
        return 0
    if args.sample_video is None:
        parser.error("Full smoke requires --sample-video; --assets-only does not complete Phase 0")
    import cv2
    import mediapipe as mp
    import numpy as np
    import pandas as pd
    import pyarrow
    import sklearn
    import seaborn
    import psutil
    import torch
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from PySide6 import QtCore, QtWidgets

    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    configs = {name:load_config(PROJECT_ROOT / f"configs/{name}.yaml") for name in ("preprocessing","training","realtime")}
    validate_config(configs["realtime"], model_metadata=configs["training"])
    asset_path = configs["preprocessing"]["asset_path"]
    canonical_path = configs["preprocessing"]["canonical_model_path"]
    if not asset_path.is_file() or not canonical_path.is_file():
        raise FileNotFoundError("Download official assets before smoke")
    capture = cv2.VideoCapture(str(args.sample_video.resolve()))
    try:
        ok, image = capture.read()
        if not ok:
            raise ValueError("Sample video cannot be decoded")
        capture.set(cv2.CAP_PROP_POS_MSEC, 1000)
        second_ok, second_image = capture.read()
    finally:
        capture.release()
    options = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(asset_path)),
                                          running_mode=vision.RunningMode.VIDEO, num_faces=1)
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        images = [(0,image)] + ([(1000,second_image)] if second_ok else [])
        results = [landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)), timestamp)
                   for timestamp,frame in images]
    face_counts = [len(result.face_landmarks) for result in results]
    landmarks_per_face = [len(face) for result in results for face in result.face_landmarks]
    if not landmarks_per_face or any(count != 478 for count in landmarks_per_face):
        raise RuntimeError(f"Positive face landmark smoke failed: {face_counts}, landmarks={landmarks_per_face}")
    torch.manual_seed(configs["training"]["seed"])
    input_size = len(configs["training"]["feature_names"])
    steps = int(configs["training"]["sequence_fps"] * configs["training"]["sequence_window_s"])
    lstm = torch.nn.LSTM(input_size,64,num_layers=1,batch_first=True).eval()
    head = torch.nn.Sequential(torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Dropout(0.3),torch.nn.Linear(32,3)).eval()
    with torch.inference_mode():
        output, _ = lstm(torch.zeros((1,steps,input_size)))
        probabilities = head(output[:,-1]).softmax(dim=-1)
    if tuple(probabilities.shape) != (1,3) or not torch.isfinite(probabilities).all() or not torch.allclose(probabilities.sum(dim=-1),torch.ones(1)):
        raise RuntimeError("PyTorch forward smoke failed")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = QtWidgets.QWidget()
    window.setWindowTitle("Driver Drowsiness — Phase 0 environment smoke")
    window.resize(640,300)
    layout = QtWidgets.QVBoxLayout(window)
    layout.addWidget(QtWidgets.QLabel("Phase 0: MediaPipe 478 landmarks, CPU PyTorch and PySide6 are running."))
    layout.addWidget(QtWidgets.QLabel("Environment smoke only — this is NOT the final monitoring UI."))
    window.show()
    screenshot = report_dir / "qt_smoke.png"
    qt_result = {"screenshot_saved": False}
    def finish():
        qt_result["screenshot_saved"] = window.grab().save(str(screenshot))
        window.close()
        application.quit()
    QtCore.QTimer.singleShot(1500, finish)
    exit_code = application.exec()
    if exit_code != 0 or not qt_result["screenshot_saved"]:
        raise RuntimeError("Qt desktop smoke failed")
    package_names = ["torch","mediapipe","opencv-contrib-python","PySide6","numpy","pandas","pyarrow","scikit-learn","matplotlib","seaborn","PyYAML","requests","pytest","psutil","ipykernel"]
    pip_check = subprocess.run([sys.executable,"-m","pip","check"],capture_output=True,text=True)
    if pip_check.returncode:
        raise RuntimeError(pip_check.stdout + pip_check.stderr)
    report = {
        "checked_at_utc":datetime.now(timezone.utc).isoformat(), "python":platform.python_version(),
        "executable":sys.executable, "platform":platform.platform(),
        "versions":{name:metadata.version(name) for name in package_names},
        "opencv_distributions":[d.metadata["Name"] for d in metadata.distributions() if d.metadata["Name"].lower().startswith("opencv-")],
        "landmark_face_counts":face_counts, "landmarks_per_face":landmarks_per_face,
        "torch_input_shape":[1,steps,input_size], "torch_output_shape":list(probabilities.shape),
        "torch_cuda_available":torch.cuda.is_available(), "torch_build_cuda":torch.version.cuda,
        "torch_mode":"CPU deliberately selected for Phase 0 and disk economy; CUDA training is a later installation choice",
        "qt_window_exit_code":exit_code,
        "qt_screenshot":str(screenshot.relative_to(PROJECT_ROOT)) if screenshot.is_relative_to(PROJECT_ROOT) else str(screenshot),
        "pip_check":pip_check.stdout.strip(), "rss_bytes":psutil.Process().memory_info().rss,
        "note":"Random initialized smoke network; not a trained drowsiness classifier or evaluation result.",
    }
    (report_dir / "environment_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (report_dir / "config_resolved.json").write_text(json.dumps(configs,indent=2,default=str),encoding="utf-8")
    freeze = subprocess.run([sys.executable,"-m","pip","freeze"],check=True,capture_output=True,text=True).stdout
    (PROJECT_ROOT / "requirements-lock.txt").write_text("# Tested on Windows x64 / Python 3.12. Install CPU torch from the official index first.\n"+freeze,encoding="utf-8")
    print(json.dumps(report,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
