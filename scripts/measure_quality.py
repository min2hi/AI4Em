"""Measure explicit source windows; propose, never apply, visually reviewed optics."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from numbers import Integral, Real
from pathlib import Path

import cv2
import numpy as np

from src.config import PROJECT_ROOT, load_config
from src.features.landmarks import FaceLandmarkDetector
from src.features.quality import METRIC_NAMES, measure_quality, validate_quality_config
from src.preprocessing.video_reader import VideoReader

REPORT_VERSION = "quality_measurements_v1"
PROGRAM_VERSION = "raw_quality_v1"


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _optical_retained(metrics: dict, policy: dict) -> bool:
    if any(metrics.get(key) is None for key in METRIC_NAMES):
        return False
    return bool(
        policy["brightness_min"] <= metrics["brightness"] <= policy["brightness_max"]
        and metrics["blur_variance"] >= policy["min_blur_variance"]
        and metrics["left_eye_width_ratio"] >= policy["min_eye_width_ratio"]
        and metrics["right_eye_width_ratio"] >= policy["min_eye_width_ratio"]
    )


def propose_policy(measurements: dict, annotations: list[dict]) -> dict:
    """Resolve reviewed annotations and derive a conservative observed envelope.

    Notes are the caller's visual-review attestation, not inferred from weak
    physiological class labels. All accepted sources and annotated frames,
    including controls, remain in the returned evidence rather than just the
    final thresholds. Invalid evidence fails closed without publishing a policy.
    """
    sources = measurements.get("sources", [])
    if not sources or any(source.get("status") == "error" for source in sources):
        raise ValueError("A proposal requires successfully measured source windows")
    samples = {}
    for source in sources:
        for sample in source.get("samples", []):
            key = (sample["video_id"], sample["timestamp_ms"])
            if key in samples:
                raise ValueError("Measured video/timestamp identity is ambiguous")
            samples[key] = sample
    if not isinstance(annotations, list) or not annotations:
        raise ValueError("A proposal requires nonempty visually reviewed annotations")
    seen = set()
    clean, rejected = [], []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError("Each annotation must be an object")
        video_id, timestamp = annotation.get("video_id"), annotation.get("timestamp_ms")
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError("Annotation video_id must be nonempty")
        if isinstance(timestamp, bool) or not isinstance(timestamp, Integral) or timestamp < 0:
            raise ValueError("Annotation timestamp_ms must be a nonnegative integer")
        key = (video_id, timestamp)
        if key in seen or key not in samples:
            raise ValueError(f"Annotation must uniquely resolve to an emitted sample: {key}")
        seen.add(key)
        if annotation.get("quality") not in ("clean", "reject"):
            raise ValueError("Annotation quality must be clean or reject")
        note = annotation.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValueError("Every annotation requires a visually reviewed note")
        target = clean if annotation["quality"] == "clean" else rejected
        target.append((annotation, samples[key]))
    if len(clean) < 2:
        raise ValueError("Review at least two clean frames to establish nondegenerate bounds")
    for _, sample in clean:
        metrics = sample["metrics"]
        if any(isinstance(metrics.get(key), bool) or not isinstance(metrics.get(key), Real)
               or not math.isfinite(metrics[key]) for key in METRIC_NAMES):
            raise ValueError("Clean annotations require finite measurements for every metric")
    policy = {
        "policy_version": PROGRAM_VERSION, "reference_max_side": 256,
        "min_blur_variance": min(sample["metrics"]["blur_variance"] for _, sample in clean),
        "brightness_min": min(sample["metrics"]["brightness"] for _, sample in clean),
        "brightness_max": max(sample["metrics"]["brightness"] for _, sample in clean),
        "min_eye_width_ratio": min(min(sample["metrics"]["left_eye_width_ratio"],
                                       sample["metrics"]["right_eye_width_ratio"]) for _, sample in clean),
        "max_abs_yaw_deg": 35., "max_abs_pitch_deg": 25., "eyes_occluded": False,
    }
    validate_quality_config(policy)
    if policy["min_blur_variance"] <= 0:
        raise ValueError("Clean samples must have nonzero texture; review more frames")
    evidence = {"clean": [], "reject": [], "controls": []}
    for kind, group in (("clean", clean), ("reject", rejected)):
        for annotation, sample in group:
            retained = _optical_retained(sample["metrics"], policy)
            evidence[kind].append({**annotation, "metrics": dict(sample["metrics"]), "retained": retained})
            if retained != (kind == "clean"):
                raise ValueError("Observed optical bounds overlap reviewed rejections; review more lawful windows")
    for annotation, sample in clean:
        controls = sample.get("controls", [])
        if len(controls) != 3 or {item.get("kind") for item in controls} != {"dark", "overexposed", "gaussian_blur"}:
            raise ValueError("Each clean frame requires darkness, overexposure and Gaussian-blur controls")
        for control in controls:
            metrics = control["metrics"]
            if any(isinstance(metrics.get(key), bool) or not isinstance(metrics.get(key), Real)
                   or not math.isfinite(metrics[key]) for key in METRIC_NAMES):
                raise ValueError("Clean-frame degraded controls require finite measured metrics")
            retained = _optical_retained(control["metrics"], policy)
            evidence["controls"].append({
                "video_id": annotation["video_id"], "timestamp_ms": annotation["timestamp_ms"],
                **control, "retained": retained,
            })
            if retained:
                raise ValueError("Proposed bounds overlap degraded controls; review more lawful windows")
    return {
        "report_version": "quality_proposal_v1", "program": "scripts.measure_quality",
        "program_version": PROGRAM_VERSION, "quality": policy,
        "annotations": deepcopy(annotations), "measurements": deepcopy(measurements),
        "evidence": evidence, "automatically_applied": False,
        "notes": ["Annotations attest visual review of optics, never physiological source labels.",
                  "This measured envelope is a conservative development policy, not universal quality calibration.",
                  "Execution owner must inspect this evidence before freezing exact bytes and updating configuration."],
    }


def _controls(image: np.ndarray, result) -> list[dict]:
    # Reuse the real detected geometry to isolate optical degradation. Re-running
    # the detector on negatives would conflate no-face with the optical metrics.
    controls = []
    for kind, intensity in (("dark", 0), ("overexposed", 255)):
        degraded = np.full_like(image, intensity)
        controls.append({"kind": kind, "parameters": {"uniform_intensity": intensity},
                         "metrics": measure_quality(degraded, result)})
    sigma = max(image.shape[:2]) / 256 * 8
    degraded = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    controls.append({"kind": "gaussian_blur", "parameters": {"sigma_source_pixels": sigma},
                     "metrics": measure_quality(degraded, result)})
    return controls


def measure_source(path: Path, video_id: str, config: dict, *, start_ms: int, end_ms: int,
                   constant_fps_verified: bool = False) -> dict:
    """Use one fresh detector/source, decoding from origin without seeking."""
    reader = VideoReader()
    detector = None
    frames = None
    record = {"video_id": video_id, "path": str(path), "status": "running", "error": None,
              "source_sha256": None, "source_size_bytes": None, "samples": []}
    try:
        record["source_sha256"] = _digest(path)
        record["source_size_bytes"] = path.stat().st_size
        detector = FaceLandmarkDetector(config)
        frames = reader.iter_frames(path, config["landmark_target_fps"],
                                    constant_fps_verified=constant_fps_verified)
        for packet in frames:
            if packet.timestamp_ms > end_ms:
                record["status"] = "bounded_window"
                break
            result = detector.detect(packet)
            if packet.timestamp_ms < start_ms:
                continue
            record["samples"].append({
                "video_id": video_id, "source_id": packet.source_id,
                "timestamp_ms": packet.timestamp_ms, "frame_index": packet.frame_index,
                "image_size": list(result.image_size), "face_detected": result.face_detected,
                "metrics": measure_quality(packet.image_bgr, result),
                "controls": _controls(packet.image_bgr, result),
            })
            if packet.timestamp_ms == end_ms:
                record["status"] = "bounded_window"
                break
        else:
            record["status"] = "eof"
        if not record["samples"]:
            raise ValueError("Requested window has no emitted source frames")
        if _digest(path) != record["source_sha256"] or path.stat().st_size != record["source_size_bytes"]:
            raise ValueError("Source changed during measurement")
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if frames is not None:
            frames.close()
        reader.close()
        if detector is not None:
            detector.close()
    record["reader"] = dict(reader.stats)
    record["detector"] = dict(detector.stats) if detector is not None else None
    record["complete_source_validation"] = record["reader"]["status"] == "EOF"
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/preprocessing.yaml")
    parser.add_argument("--video", type=Path, action="append", required=True,
                        help="Repeat for fresh independent source sessions; IDs are parent-name_stem")
    parser.add_argument("--start-ms", type=int, default=0)
    parser.add_argument("--end-ms", type=int, required=True)
    parser.add_argument("--annotations", type=Path,
                        help="JSON annotation list, or an object containing annotations; notes attest visual review")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--constant-fps-verified", action="store_true",
                        help="Caller asserts independently verified CFR; never inferred from reported FPS")
    args = parser.parse_args(argv)
    if args.start_ms < 0 or args.end_ms <= args.start_ms:
        parser.error("Require 0 <= --start-ms < --end-ms")
    sources = [path.resolve() for path in args.video]
    video_ids = [f"{path.parent.name}_{path.stem}" for path in sources]
    if len(set(sources)) != len(sources) or len(set(video_ids)) != len(video_ids):
        parser.error("Repeated sources and ambiguous parent-name_stem video IDs are not permitted")
    config = load_config(args.config)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = args.report_dir / "quality_proposal.json"
    # A failed rerun must not leave an older proposal next to fresh measurements.
    proposal_path.unlink(missing_ok=True)
    report = {
        "report_version": REPORT_VERSION, "program": "scripts.measure_quality",
        "program_version": PROGRAM_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "program_sha256": _digest(Path(__file__)),
        "metric_program_sha256": _digest(PROJECT_ROOT / "src/features/quality.py"),
        "versions": {name: metadata.version(name) for name in ("mediapipe", "opencv-contrib-python", "numpy")},
        "asset_sha256": _digest(Path(config["asset_path"])),
        "landmark_target_fps": config["landmark_target_fps"],
        "detector_config": {key: config[key] for key in (
            "landmarker_mode", "num_faces", "min_face_detection_confidence",
            "min_face_presence_confidence", "min_tracking_confidence")},
        "requested_window_ms": {"start": args.start_ms, "end": args.end_ms, "bounds": "inclusive"},
        "constant_fps_verified_by_caller": args.constant_fps_verified,
        "roi": {"landmark_ids": [234, 454, 10, 152], "pixel_bounds": "floor-min/ceil-max half-open",
                "reference_max_side": 256, "interpolation": "INTER_AREA", "laplacian_depth": "CV_64F"},
        "sources": [measure_source(path, video_id, config, start_ms=args.start_ms, end_ms=args.end_ms,
                                   constant_fps_verified=args.constant_fps_verified)
                    for path, video_id in zip(sources, video_ids)],
        "proposal_status": "not_requested", "proposal_error": None,
        "notes": ["No faces/images or feature Parquet are saved.",
                  "Controls reuse original landmarks and are optical diagnostic negatives, not physiological labels.",
                  "Window diagnostics do not validate a whole file unless reader reached actual EOF.",
                  "Annotation note is a visual-review attestation; source labels are neither read nor used."],
    }
    proposal = None
    if args.annotations is not None:
        try:
            raw = args.annotations.read_bytes()
            annotations = json.loads(raw)
            if isinstance(annotations, dict):
                annotations = annotations.get("annotations")
            report["annotation_provenance"] = {"path": str(args.annotations.resolve()),
                                               "sha256": hashlib.sha256(raw).hexdigest()}
            report["proposal_status"] = "proposed"
            proposal = propose_policy(report, annotations)
        except (ValueError, OSError, UnicodeError) as exc:
            report["proposal_status"] = "rejected"
            report["proposal_error"] = f"{type(exc).__name__}: {exc}"
    (args.report_dir / "quality_measurements.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if proposal is not None:
        proposal_path.write_text(json.dumps(proposal, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"report_dir": str(args.report_dir.resolve()),
                      "samples": sum(len(source["samples"]) for source in report["sources"]),
                      "proposal_status": report["proposal_status"], "proposal_error": report["proposal_error"]},
                     allow_nan=False))
    return int(any(source["status"] == "error" for source in report["sources"])
               or report["proposal_status"] == "rejected")


if __name__ == "__main__":
    raise SystemExit(main())
