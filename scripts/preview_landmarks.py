"""Unmirrored landmarks and facial geometry preview, bounded capture smoke and stats."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from src.config import PROJECT_ROOT, load_config
from src.contracts import EyeFeatures, LandmarkResult, MouthFeatures
from src.features.landmarks import FaceLandmarkDetector
from src.features.eye import LEFT_EYE, RIGHT_EYE, EyeFeatureExtractor
from src.features.mouth import MOUTH_HORIZONTAL, MOUTH_VERTICAL, MouthFeatureExtractor
from src.preprocessing.video_reader import VideoReader


WINDOW_TITLE = "Facial geometry - unmirrored (q/Esc: stop)"


def draw_overlay(image_bgr: np.ndarray, result: LandmarkResult, *,
                 eyes: EyeFeatures | None = None, mouth: MouthFeatures | None = None) -> np.ndarray:
    """Own the preview copy; never mirror or modify the inference image."""
    image = image_bgr.copy()
    width, height = result.image_size
    if result.points is not None and result.face_detected:
        pixels = np.rint(result.points[:, :2] * (width, height)).astype(np.int32)
        for x, y in pixels:
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(image, (int(x), int(y)), 1, (80, 220, 80), -1)
        for indices, color in ((LEFT_EYE, (255, 180, 0)), (RIGHT_EYE, (0, 180, 255))):
            cv2.polylines(image, [pixels[list(indices)]], True, color, 2)
        if mouth is not None:
            for start, end in (MOUTH_HORIZONTAL, *MOUTH_VERTICAL):
                cv2.line(image, tuple(pixels[start]), tuple(pixels[end]), (255, 0, 255), 2)
        message = f"478 landmarks | {result.timestamp_ms} ms"
    else:
        message = f"NO VALID FACE | {result.timestamp_ms} ms"
    rows = [message, "L(anatomical): cyan | R: orange | NOT mirrored"]
    if eyes is not None:
        left = format_measurement(eyes.ear_left, eyes.left_eye_valid)
        right = format_measurement(eyes.ear_right, eyes.right_eye_valid)
        mean = format_measurement(eyes.ear_mean, eyes.left_eye_valid and eyes.right_eye_valid)
        rows.append(f"EAR L:{left} R:{right} mean:{mean}")
    if mouth is not None:
        rows.append(f"MAR:{format_measurement(mouth.mar, mouth.mouth_valid)}")
    # HighGUI fits either orientation inside 640 px; keep labels readable there.
    text_scale = max(1.0, max(width, height) / 640)
    for index, text in enumerate(rows):
        origin = (round(8 * text_scale), round((24 + 21 * index) * text_scale))
        font_scale = (0.5 if index == 0 else 0.4) * text_scale
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), max(1, round(text_scale)), cv2.LINE_AA)
    return image


def format_measurement(value: float, valid: bool) -> str:
    return f"{value:.3f}" if valid and math.isfinite(value) else "N/A"


def new_geometry_summary() -> dict:
    return {"scheduled_frames": 0, "valid_frames": 0, "invalid_frames": 0,
            "min": None, "max": None, "sum": 0.0}


def update_geometry_summary(summary: dict, value: float, valid: bool) -> None:
    summary["scheduled_frames"] += 1
    if not valid or not math.isfinite(value):
        summary["invalid_frames"] += 1
        return
    summary["valid_frames"] += 1
    summary["sum"] += float(value)
    summary["min"] = float(value) if summary["min"] is None else min(summary["min"], float(value))
    summary["max"] = float(value) if summary["max"] is None else max(summary["max"], float(value))


def export_geometry_summary(summary: dict) -> dict:
    result = {key: value for key, value in summary.items() if key != "sum"}
    result["mean"] = summary["sum"] / summary["valid_frames"] if summary["valid_frames"] else None
    return result


def run_source(source: Path | int, config: dict, args: argparse.Namespace, index: int) -> dict:
    reader = VideoReader()
    detector = None
    camera = isinstance(source, int)
    report = {"source": str(source), "kind": "camera" if camera else "video",
              "status": "running", "error": None, "overlay": None,
              "mirrored": False, "complete_source_validation": False}
    times = []
    first_timestamp = None
    last_timestamp = None
    summaries = {name: new_geometry_summary() for name in ("ear_left", "ear_right", "ear_mean", "mar")}
    started = time.perf_counter()
    frames = None
    try:
        eye_extractor = EyeFeatureExtractor(config["epsilon"])
        mouth_extractor = MouthFeatureExtractor(config["epsilon"])
        detector = FaceLandmarkDetector(config)
        frames = (reader.iter_camera(source, config["landmark_target_fps"],
                                     width=args.camera_width, height=args.camera_height)
                  if camera else reader.iter_frames(source, config["landmark_target_fps"],
                                                    constant_fps_verified=args.constant_fps_verified))
        if not args.headless:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        for packet in frames:
            if first_timestamp is None:
                first_timestamp = packet.timestamp_ms
                if not args.headless:
                    height, width = packet.image_bgr.shape[:2]
                    scale = min(640 / width, 640 / height)
                    cv2.resizeWindow(WINDOW_TITLE, round(width * scale), round(height * scale))
            before = time.perf_counter()
            result = detector.detect(packet)
            times.append((time.perf_counter() - before) * 1000)
            eyes = eye_extractor.extract(result)
            mouth = mouth_extractor.extract(result)
            update_geometry_summary(summaries["ear_left"], eyes.ear_left, eyes.left_eye_valid)
            update_geometry_summary(summaries["ear_right"], eyes.ear_right, eyes.right_eye_valid)
            update_geometry_summary(summaries["ear_mean"], eyes.ear_mean, eyes.left_eye_valid and eyes.right_eye_valid)
            update_geometry_summary(summaries["mar"], mouth.mar, mouth.mouth_valid)
            last_timestamp = packet.timestamp_ms
            if not args.headless or (args.save_overlay and report["overlay"] is None and result.face_detected):
                overlay = draw_overlay(packet.image_bgr, result, eyes=eyes, mouth=mouth)
                if args.save_overlay and report["overlay"] is None and result.face_detected:
                    destination = args.report_dir / f"source_{index}_overlay.png"
                    if not cv2.imwrite(str(destination), overlay):
                        raise RuntimeError(f"Cannot write overlay: {destination}")
                    report["overlay"] = str(destination)
                if not args.headless:
                    cv2.imshow(WINDOW_TITLE, overlay)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        report["status"] = "user_stopped"
                        break
            elapsed = ((packet.timestamp_ms - first_timestamp) / 1000 if not camera
                       else time.perf_counter() - started)
            if args.seconds is not None and elapsed >= args.seconds:
                report["status"] = "bounded_prefix"
                break
        else:
            report["status"] = "eof"
            report["complete_source_validation"] = not camera
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if frames is not None:
            frames.close()
        reader.close()
        if detector is not None:
            detector.close()
        if not args.headless:
            cv2.destroyAllWindows()
    wall = time.perf_counter() - started
    report.update(reader=dict(reader.stats), detector=dict(detector.stats) if detector is not None else None,
                  wall_seconds=wall, processed_fps=len(times) / wall if wall > 0 else 0,
                  first_timestamp_ms=first_timestamp, last_timestamp_ms=last_timestamp,
                  inference_ms={"p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95))}
                  if times else None)
    report["features"] = {name: export_geometry_summary(summary) for name, summary in summaries.items()}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/preprocessing.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, action="append", help="Repeat for independent sessions")
    source.add_argument("--camera", type=int)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--seconds", type=float, help="Source seconds for files; wall seconds for camera")
    parser.add_argument("--constant-fps-verified", action="store_true",
                        help="Caller asserts independent CFR verification; record explicit fallback if needed")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-overlay", action="store_true", help="Opt-in face image; check publication permissions")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "runs/phase5")
    args = parser.parse_args()
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be finite and positive")
    if args.camera is not None and (args.camera < 0 or args.constant_fps_verified):
        parser.error("Camera index must be nonnegative; CFR fallback applies only to files")
    if args.camera_width <= 0 or args.camera_height <= 0:
        parser.error("Camera dimensions must be positive")
    config = load_config(args.config)
    args.report_dir = args.report_dir.resolve()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    with Path(config["asset_path"]).open("rb") as handle:
        asset_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    sources = [path.resolve() for path in args.video] if args.video else [args.camera]
    records = [run_source(item, config, args, index) for index, item in enumerate(sources)]
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "phase": 5,
              "versions": {name: metadata.version(name) for name in ("mediapipe", "opencv-contrib-python", "numpy")},
              "asset_sha256": asset_sha, "landmark_target_fps": config["landmark_target_fps"],
              "constant_fps_verified_by_caller": args.constant_fps_verified,
              "requested_seconds": args.seconds, "sources": records,
              "notes": ["Bounded prefixes do not validate complete files or the full dataset.",
                        "Face present is not absolute quality; no confidence/visibility is fabricated.",
                        "EAR/MAR are geometry measurements, not blink/yawn or drowsiness labels.",
                        "Camera physical disconnect and human no-face transitions need observed hardware evidence."]}
    destination = args.report_dir / "preview_report.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return int(any(record["status"] == "error" for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
