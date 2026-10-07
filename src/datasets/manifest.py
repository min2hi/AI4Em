"""Inspect real videos and retain failed inputs instead of silently discarding them."""
from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any

import cv2
import pandas as pd

from src.datasets.acquisition import SOURCE_LABELS, VIDEO_SUFFIXES, digest_file

MANIFEST_COLUMNS = [
    "dataset_name", "subject_id", "video_id", "relative_path", "source_label", "label_id", "label_source",
    "part_id", "fold_id", "fold_source", "split", "fps_reported", "duration_s", "width", "height",
    "frame_count", "size_bytes", "sha256", "image_publishable", "annotation_path",
    "sampled_decode_ok", "sample_timestamps_ms", "status", "error",
]


def probe_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("OpenCV cannot open video")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1 or min(width, height) < 1:
            raise ValueError("Invalid FPS/frame count/resolution metadata")
        frame_count = int(count)
        timestamps = []
        for frame_index in sorted({0, frame_count // 2, frame_count - 1}):
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                raise ValueError(f"Video backend cannot seek to frame {frame_index}")
            ok, image = capture.read()
            if not ok or image is None:
                raise ValueError(f"Cannot decode sampled frame {frame_index}")
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError(f"Invalid decoded timestamp at frame {frame_index}")
            timestamps.append(timestamp)
        if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("Sampled timestamps are not strictly increasing")
        return {"fps_reported": fps, "duration_s": frame_count / fps, "width": width, "height": height,
                "frame_count": frame_count, "sampled_decode_ok": True, "sample_timestamps_ms": timestamps}
    finally:
        capture.release()


def build_manifest(root: str | Path, dataset: str, *, source_records: list[dict] | None = None,
                   permission_map: dict[str, bool] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    if dataset != "uta_rldd":
        raise ValueError("Phase 2 currently supports the acquired UTA-RLDD dataset only")
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    sources = {}
    for record in source_records or []:
        relative = record["relative_path"]
        if relative in sources:
            raise ValueError(f"Duplicate source record: {relative}")
        sources[relative] = record
    rows, errors = [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        row = {name: None for name in MANIFEST_COLUMNS}
        row.update(dataset_name=dataset, relative_path=relative, subject_id=path.parent.name,
                   video_id=f"{path.parent.name}_{path.stem}", label_source="video_weak", split="unassigned",
                   size_bytes=path.stat().st_size, sampled_decode_ok=False, status="error", error="")
        try:
            if not re.fullmatch(r"[0-9]{2}", row["subject_id"]):
                raise ValueError("Subject ID must retain two-digit UTA identity")
            match = re.fullmatch(r"(0|5|10)(?:_([1-9][0-9]*))?", path.stem)
            if not match:
                raise ValueError("Unknown UTA source label; not guessed from filename")
            label = int(match.group(1))
            row.update(source_label=label, label_id=SOURCE_LABELS[label], part_id=match.group(2))
            source = sources.get(relative)
            if source:
                if source["subject_id"] != row["subject_id"] or source["source_label"] != label or source["video_id"] != row["video_id"]:
                    raise ValueError("Source inventory and local identity do not match")
                if source["size_bytes"] != row["size_bytes"]:
                    raise ValueError("Downloaded size differs from source index")
                row.update(fold_id=source["fold_id"], fold_source="official_archive_index")
            if permission_map is not None:
                row["image_publishable"] = permission_map.get(row["subject_id"])
            sha, _ = digest_file(path)
            row["sha256"] = sha
            if source and source.get("sha256") and source["sha256"] != sha:
                raise ValueError("SHA256 differs from verified download receipt")
            row.update(probe_video(path))
            row["status"] = "ok"
        except (ValueError, OSError, cv2.error) as exc:
            row["error"] = str(exc)
            errors.append({"relative_path": relative, "error": str(exc)})
        rows.append(row)
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    for column in ("source_label", "label_id", "fold_id"):
        frame[column] = pd.array(frame[column], dtype="Int8")
    for column in ("width", "height", "frame_count", "size_bytes"):
        frame[column] = pd.array(frame[column], dtype="Int64")
    for column in ("subject_id", "video_id", "relative_path", "part_id"):
        frame[column] = frame[column].astype("string")
    frame["image_publishable"] = pd.array(frame["image_publishable"], dtype="boolean")
    frame["sampled_decode_ok"] = frame["sampled_decode_ok"].astype(bool)
    return frame, errors


def validate_manifest(frame: pd.DataFrame, *, expected_subjects: int | None = None,
                      expected_videos: int | None = None, require_folds: bool = False,
                      expected_records: list[dict] | None = None) -> None:
    missing = set(MANIFEST_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("No videos discovered")
    for key in ("relative_path", "video_id"):
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"Duplicate or missing {key}")
    if (frame["status"] != "ok").any() or not frame["sampled_decode_ok"].all():
        raise ValueError("Manifest retains unreadable/invalid inputs; inspect error report")
    if frame["label_id"].isna().any() or not frame["source_label"].isin(SOURCE_LABELS).all():
        raise ValueError("Invalid class labels")
    for row in frame.itertuples():
        if row.label_id != SOURCE_LABELS[row.source_label]:
            raise ValueError("Source/internal label mismatch")
    for subject, group in frame.groupby("subject_id"):
        if set(group["source_label"]) != {0, 5, 10}:
            raise ValueError(f"Subject {subject} lacks a state; never split by frame to compensate")
        if group["fold_id"].nunique() > 1:
            raise ValueError(f"Subject {subject} appears in multiple official folds")
    known_folds = frame["fold_id"].dropna()
    if not known_folds.isin([1, 2, 3, 4, 5]).all() or require_folds and frame["fold_id"].isna().any():
        raise ValueError("Missing or invalid official fold metadata")
    if expected_subjects is not None and frame["subject_id"].nunique() != expected_subjects:
        raise ValueError("Subject count does not match acquisition plan")
    if expected_videos is not None and len(frame) != expected_videos:
        raise ValueError("Physical video count does not match acquisition plan")
    if expected_records is not None:
        keys = ("subject_id", "video_id", "source_label", "label_id", "fold_id", "size_bytes")
        expected = {row["relative_path"]: tuple(row.get(key) for key in keys) for row in expected_records}
        actual = {row["relative_path"]: tuple(row[key] for key in keys) for row in frame.to_dict("records")}
        if len(expected) != len(expected_records) or actual != expected:
            raise ValueError("Manifest identities/provenance differ from acquisition plan")


def validate_working_manifest(frame: pd.DataFrame) -> None:
    """Validate an identifiable working snapshot, not acquisition completeness.

    Failed source rows are retained for reporting, but cannot relax identities,
    labels or official subject fold consistency.
    """
    missing = set(MANIFEST_COLUMNS) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Missing manifest columns or empty snapshot: {sorted(missing)}")
    for key in ("relative_path", "video_id"):
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"Duplicate or missing {key}")
    for row in frame.to_dict("records"):
        subject, video, relative = row["subject_id"], row["video_id"], row["relative_path"]
        if not isinstance(row["dataset_name"], str) or row["dataset_name"] != "uta_rldd" or not isinstance(subject, str) or not re.fullmatch(r"[0-9]{2}", subject):
            raise ValueError("Invalid dataset or two-digit UTA subject identity")
        match = re.fullmatch(r"([0-9]{2})_(0|5|10)(?:_([1-9][0-9]*))?", video) if isinstance(video, str) else None
        if match is None or match.group(1) != subject:
            raise ValueError("Unsafe or inconsistent UTA video identity")
        label = int(match.group(2))
        if (pd.isna(row["source_label"]) or isinstance(row["source_label"], bool)
                or row["source_label"] != label or pd.isna(row["label_id"])
                or isinstance(row["label_id"], bool) or row["label_id"] != SOURCE_LABELS[label]):
            raise ValueError("Source/internal label mismatch")
        stem = video[len(subject) + 1:]
        if (not isinstance(relative, str) or "\\" in relative
                or not re.fullmatch(re.escape(subject + "/" + stem) + r"\.[a-zA-Z0-9]+", relative)
                or Path(relative).suffix.lower() not in VIDEO_SUFFIXES):
            raise ValueError("Unsafe or inconsistent relative source identity")
        part = row["part_id"]
        if (match.group(3) is None and not pd.isna(part)) or (match.group(3) is not None and str(part) != match.group(3)):
            raise ValueError("Part identity mismatch")
        if not isinstance(row["label_source"], str) or row["label_source"] != "video_weak":
            raise ValueError("Invalid UTA label provenance")
        fold = row["fold_id"]
        if not pd.isna(fold) and (isinstance(fold, bool) or fold not in (1, 2, 3, 4, 5)):
            raise ValueError("Invalid official fold")
        if not pd.isna(fold) and (not isinstance(row["fold_source"], str) or row["fold_source"] != "official_archive_index"):
            raise ValueError("Known fold requires official provenance")
        status = row["status"]
        if not isinstance(status, str) or status not in ("ok", "error"):
            raise ValueError("Unknown manifest status")
        if status == "error":
            if not isinstance(row["error"], str) or not row["error"].strip():
                raise ValueError("Error row must retain a meaningful reason")
            continue
        if pd.isna(fold) or not isinstance(row["fold_source"], str) or row["fold_source"] != "official_archive_index":
            raise ValueError("Missing official fold provenance")
        if pd.isna(row["sampled_decode_ok"]) or row["sampled_decode_ok"] != True:
            raise ValueError("Successful source requires sampled decode")
        if not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-fA-F]{64}", row["sha256"]):
            raise ValueError("Invalid source SHA256")
        for key in ("width", "height", "frame_count", "size_bytes"):
            value = row[key]
            if pd.isna(value) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 or int(value) != value:
                raise ValueError(f"Invalid source metadata: {key}")
        for key in ("fps_reported", "duration_s"):
            value = row[key]
            if pd.isna(value) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid source metadata: {key}")
    for subject, group in frame.groupby("subject_id"):
        if group["fold_id"].nunique() > 1:
            raise ValueError(f"Subject {subject} appears in multiple official folds")
