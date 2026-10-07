"""Stream raw features and publish metadata-last, hash-validated video pairs.

Single-writer only. Consumers must validate the companion commit marker, never
interpret a lone Parquet file or an old metadata file as a completed extraction.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
from typing import Any
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import validate_config
from src.datasets.acquisition import digest_file
from src.datasets.manifest import validate_working_manifest
from src.features.pipeline import FeaturePipeline
from src.features.quality import validate_quality_config
from src.preprocessing.video_reader import VideoReader

PRODUCER_VERSION = "raw_feature_builder_v1"
SCHEMA_VERSION = "facial_features_v1"
FLOAT_FIELDS = ("ear_left", "ear_right", "ear_mean", "mar", "pitch", "yaw", "roll", "reprojection_error_norm")
FLAG_FIELDS = ("face_detected", "left_eye_valid", "right_eye_valid", "mouth_valid", "pose_valid")
STORAGE_SCHEMA = pa.schema([
    pa.field("dataset_name", pa.string()), pa.field("subject_id", pa.string()),
    pa.field("video_id", pa.string()), pa.field("frame_index", pa.int32()),
    pa.field("timestamp_ms", pa.int64()),
    *(pa.field(name, pa.float32()) for name in FLOAT_FIELDS[:-1]),
    *(pa.field(name, pa.bool_()) for name in FLAG_FIELDS),
    pa.field("reprojection_error_norm", pa.float32()), pa.field("label_id", pa.int8()),
    pa.field("label_source", pa.string()),
])


def _json_value(value: Any) -> Any:
    """Convert scalar/array provenance and diagnostics into finite JSON values."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Nonfinite JSON constant: {value}")
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle, parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError("Metadata/profile must be a JSON object")
    return result


def storage_row(sample: Any, manifest_row: dict) -> dict:
    """Join only storage provenance; retain every emitted sample and missing value."""
    row = {name: getattr(sample, name) for name in ("frame_index", "timestamp_ms", *FLOAT_FIELDS, *FLAG_FIELDS)}
    for name in FLOAT_FIELDS:
        value = row[name]
        row[name] = float(value) if math.isfinite(value) else None
    row.update({name: manifest_row[name] for name in
                ("dataset_name", "subject_id", "video_id", "label_id", "label_source")})
    if pd.isna(row["label_id"]):
        row["label_id"] = None
    return row


class FeatureDatasetBuilder:
    """Extract selected complete sources with at most 1024 buffered scalar rows."""

    def __init__(self, config: dict, *, constant_fps_verified: bool = False):
        if not isinstance(constant_fps_verified, bool):
            raise ValueError("constant_fps_verified must be explicit boolean")
        validate_config(config)
        self.config = config
        self.constant_fps_verified = constant_fps_verified

    def _signature(self) -> dict:
        validate_quality_config(self.config["quality"], require_frozen=True)
        hashes = {name: digest_file(Path(self.config[key]))[0] for name, key in
                  (("asset_sha256", "asset_path"), ("canonical_sha256", "canonical_model_path"))}
        quality = self.config["quality"]
        profile_path = Path(quality["frozen_report_path"])
        hashes["quality_sha256"] = digest_file(profile_path)[0]
        if hashes["quality_sha256"] != quality["frozen_report_sha256"]:
            raise ValueError("Frozen quality profile SHA256 mismatch")
        effective = {key: value for key, value in self.config.items()
                     if key not in ("dataset_root", "output_dir", "manifest_path")}
        dependencies = {name: version(name) for name in
                        ("numpy", "opencv-contrib-python", "mediapipe", "pandas", "pyarrow")}
        signature = dict(schema_version=SCHEMA_VERSION, producer_version=PRODUCER_VERSION,
                         effective_config=_json_value(effective), artifact_hashes=hashes,
                         dependency_versions=dependencies,
                         constant_fps_verified=self.constant_fps_verified)
        signature["config_sha256"] = _hash_json(effective)
        signature["extraction_fingerprint"] = _hash_json(signature)
        return signature

    @staticmethod
    def _paths(root: Path, video_id: str) -> tuple[Path, Path]:
        parquet = root / f"{video_id}.parquet"
        metadata = root / f"{video_id}.metadata.json"
        for path in (parquet, metadata):
            if not path.resolve().is_relative_to(root):
                raise ValueError("Destination escapes output directory")
        return parquet, metadata

    @staticmethod
    def _source(root: Path, row: dict) -> Path:
        path = (root / row["relative_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Source escapes dataset_root")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    @staticmethod
    def _cached(parquet: Path, marker: Path, row: dict, source_sha: str,
                source_size: int, signature: dict) -> dict | None:
        if not parquet.is_file() or not marker.is_file():
            return None
        try:
            metadata = _read_json(marker)
            required = ("reader_stats", "pipeline_stats", "validity_counts",
                        "quality_reason_counts", "camera_metadata", "artifact_hashes",
                        "dependency_versions", "effective_config", "manifest_row")
            if any(not isinstance(metadata.get(key), dict) for key in required):
                return None
            if (type(metadata.get("row_count")) is not int
                    or type(metadata.get("first_timestamp_ms")) is not int
                    or type(metadata.get("last_timestamp_ms")) is not int
                    or metadata["first_timestamp_ms"] < 0
                    or metadata["last_timestamp_ms"] < metadata["first_timestamp_ms"]
                    or metadata["reader_stats"].get("emitted_frames") != metadata["row_count"]
                    or metadata["artifact_hashes"] != signature["artifact_hashes"]
                    or metadata["dependency_versions"] != signature["dependency_versions"]
                    or metadata["effective_config"] != signature["effective_config"]
                    or metadata["manifest_row"] != _json_value(row)):
                return None
            if (metadata.get("schema_version") != SCHEMA_VERSION
                    or metadata.get("producer_version") != PRODUCER_VERSION
                    or metadata.get("complete_source_validation") is not True
                    or metadata.get("extraction_fingerprint") != signature["extraction_fingerprint"]
                    or metadata.get("source_sha256") != source_sha
                    or metadata.get("source_size_bytes") != source_size
                    or metadata.get("manifest_row_sha256") != _hash_json(row)
                    or metadata.get("reader_stats", {}).get("status") != "EOF"
                    or metadata.get("reader_stats", {}).get("capture_released") is not True
                    or metadata.get("parquet_sha256") != digest_file(parquet)[0]):
                return None
            with pq.ParquetFile(parquet) as stored:
                if (not stored.schema_arrow.equals(STORAGE_SCHEMA, check_metadata=True)
                        or stored.metadata.num_rows != metadata.get("row_count")
                        or stored.metadata.num_rows <= 0):
                    return None
            return metadata
        except (OSError, ValueError, TypeError, KeyError, pa.ArrowException):
            return None

    def _extract(self, row: dict, source: Path, source_sha: str, source_size: int,
                 parquet: Path, marker: Path, signature: dict, diagnostics: dict) -> dict:
        token = uuid.uuid4().hex
        staged_parquet = parquet.with_name(parquet.name + "." + token + ".tmp")
        staged_marker = marker.with_name(marker.name + "." + token + ".tmp")
        reader = VideoReader()
        pipeline = None
        writer = None
        iterator = None
        count = 0
        first_timestamp = last_timestamp = last_index = None
        validity = Counter({name: 0 for name in FLAG_FIELDS})
        reasons: Counter = Counter()
        image_size = camera = None
        pipeline_stats = None
        try:
            pipeline = FeaturePipeline(self.config)
            iterator = reader.iter_frames(source, target_fps=self.config["landmark_target_fps"],
                                          constant_fps_verified=self.constant_fps_verified)
            writer = pq.ParquetWriter(staged_parquet, STORAGE_SCHEMA)
            buffer = []
            for packet in iterator:
                height, width = packet.image_bgr.shape[:2]
                actual_size = (width, height)
                if image_size is None:
                    image_size = actual_size
                    if image_size != (int(row["width"]), int(row["height"])):
                        raise ValueError("First-frame resolution differs from manifest")
                    camera = pipeline.pose_estimator.camera_metadata(image_size)
                elif actual_size != image_size:
                    raise ValueError("Within-source resolution change cannot be represented")
                if (not 0 <= packet.frame_index <= 2**31 - 1
                        or not 0 <= packet.timestamp_ms <= 2**63 - 1
                        or last_index is not None and packet.frame_index <= last_index
                        or last_timestamp is not None and packet.timestamp_ms <= last_timestamp):
                    raise ValueError("Source indices/timestamps must be strictly increasing and fit storage types")
                sample = pipeline.process(packet)
                if (sample.frame_index != packet.frame_index or sample.timestamp_ms != packet.timestamp_ms
                        or sample.source_id != packet.source_id):
                    raise ValueError("Feature pipeline changed source identity")
                buffer.append(storage_row(sample, row))
                count += 1
                if first_timestamp is None:
                    first_timestamp = packet.timestamp_ms
                last_timestamp, last_index = packet.timestamp_ms, packet.frame_index
                validity.update({name: int(getattr(sample, name)) for name in FLAG_FIELDS})
                if pipeline.last_quality is not None:
                    reasons.update(pipeline.last_quality.reasons)
                if len(buffer) == 1024:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=STORAGE_SCHEMA))
                    buffer.clear()
            if reader.stats["status"] != "EOF" or not reader.stats["capture_released"] or count == 0:
                raise RuntimeError("Only released full-EOF sources may be published")
            if buffer:
                writer.write_table(pa.Table.from_pylist(buffer, schema=STORAGE_SCHEMA))
                buffer.clear()
            writer.close()
            writer = None
            pipeline.close()
            pipeline_stats = pipeline.stats
            pipeline = None
            after_sha = digest_file(source)[0]
            after_size = source.stat().st_size
            if after_sha != source_sha or after_size != source_size:
                raise ValueError("Source changed during extraction")
            # Sync data before replacing either final path; the JSON marker is last.
            # Windows FlushFileBuffers requires a writable handle, even after close.
            with staged_parquet.open("r+b") as handle:
                os.fsync(handle.fileno())
            metadata = dict(signature, source_id=str(source), source_sha256=source_sha,
                source_size_bytes=source_size, manifest_row=_json_value(row),
                manifest_row_sha256=_hash_json(row), image_size=image_size, camera_metadata=camera,
                reader_stats=reader.stats, pipeline_stats=pipeline_stats, row_count=count,
                first_timestamp_ms=first_timestamp, last_timestamp_ms=last_timestamp,
                last_frame_index=last_index, validity_counts=dict(validity),
                quality_reason_counts=dict(reasons),
                created_at_utc=datetime.now(timezone.utc).isoformat(),
                complete_source_validation=True, parquet_sha256=digest_file(staged_parquet)[0])
            metadata = _json_value(metadata)
            with staged_marker.open("x", encoding="utf-8") as handle:
                json.dump(metadata, handle, sort_keys=True, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(staged_parquet, parquet)
            os.replace(staged_marker, marker)
            return metadata
        finally:
            try:
                if iterator is not None:
                    iterator.close()
                reader.close()
            finally:
                try:
                    if writer is not None:
                        writer.close()
                finally:
                    try:
                        if pipeline is not None:
                            pipeline.close()
                    finally:
                        diagnostics.update(reader_stats=reader.stats, row_count=count,
                            first_timestamp_ms=first_timestamp, last_timestamp_ms=last_timestamp,
                            validity_counts=dict(validity), quality_reason_counts=dict(reasons))
                        if pipeline is not None:
                            diagnostics["pipeline_stats"] = pipeline.stats
                        elif pipeline_stats is not None:
                            diagnostics["pipeline_stats"] = pipeline_stats
                        staged_parquet.unlink(missing_ok=True)
                        staged_marker.unlink(missing_ok=True)

    def build(self, manifest: pd.DataFrame, output_dir: Path, *,
              video_ids: tuple[str, ...] | None = None) -> dict:
        validate_working_manifest(manifest)
        rows = manifest.to_dict("records")
        by_id = {row["video_id"]: row for row in rows}
        if video_ids is not None:
            if not video_ids or len(set(video_ids)) != len(video_ids) or any(item not in by_id for item in video_ids):
                raise ValueError("Selection must contain known unique video IDs")
            selected = [by_id[item] for item in video_ids]
        else:
            selected = rows
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        dataset_root = Path(self.config["dataset_root"]).resolve()
        report = dict(schema_version=SCHEMA_VERSION, producer_version=PRODUCER_VERSION,
            created_at_utc=datetime.now(timezone.utc).isoformat(), selected_count=len(selected),
            working_snapshot_count=len(rows), acquisition_planned_count=45,
            working_manifest_sha256=_hash_json(sorted(rows, key=lambda row: row["video_id"])),
            selected_manifest_sha256=_hash_json(selected),
            unselected_video_ids=[row["video_id"] for row in rows if row["video_id"] not in {item["video_id"] for item in selected}],
            class_counts=dict(Counter(str(row["label_id"]) for row in selected)),
            source_label_counts=dict(Counter(str(row["source_label"]) for row in selected)),
            videos=[])
        for row in selected:
            item = dict(video_id=row["video_id"], subject_id=row["subject_id"], status="failed", output_current=False)
            try:
                if row["status"] == "error":
                    raise ValueError(f"Manifest source error: {row['error']}")
                parquet, marker = self._paths(output, row["video_id"])
                item.update(parquet_path=str(parquet), metadata_path=str(marker),
                            previous_output_present=parquet.exists() or marker.exists())
                source = self._source(dataset_root, row)
                source_sha = digest_file(source)[0]
                source_size = source.stat().st_size
                if source_sha.lower() != row["sha256"].lower() or source_size != int(row["size_bytes"]):
                    raise ValueError("Actual source hash/size differs from manifest")
                signature = self._signature()
                metadata = self._cached(parquet, marker, row, source_sha, source_size, signature)
                if metadata is None:
                    metadata = self._extract(row, source, source_sha, source_size, parquet, marker, signature, item)
                    item["status"] = "completed"
                else:
                    item["status"] = "cached"
                item.update(output_current=True, row_count=metadata["row_count"],
                            first_timestamp_ms=metadata["first_timestamp_ms"],
                            last_timestamp_ms=metadata["last_timestamp_ms"],
                            extraction_fingerprint=metadata["extraction_fingerprint"],
                            reader_stats=metadata["reader_stats"], pipeline_stats=metadata["pipeline_stats"],
                            validity_counts=metadata["validity_counts"],
                            quality_reason_counts=metadata["quality_reason_counts"])
            except Exception as exc:
                item.update(error=str(exc), stale_output=bool(item.get("previous_output_present")))
            report["videos"].append(item)
        report["totals"] = {status: sum(item["status"] == status for item in report["videos"])
                            for status in ("completed", "cached", "failed")}
        report["status"] = "failed" if report["totals"]["failed"] else "completed"
        return _json_value(report)
