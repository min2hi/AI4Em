"""Chronological source-time replay and parity comparison."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Protocol, Sequence

import numpy as np

from src.contracts import DetectionResult, FramePacket, Prediction


class ReplayDetector(Protocol):
    def process(self, packet: FramePacket, *, current_time_ms: int | None = None) -> DetectionResult: ...


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    source_id: str
    frame_index: int
    timestamp_ms: int
    system_status: str
    calibration_status: str
    raw_probabilities: tuple[float, float, float] | None
    raw_class_id: int | None
    smoothed_probabilities: tuple[float, float, float] | None
    smoothed_class_id: int | None


def _probabilities(prediction: Prediction | None) -> tuple[float, float, float] | None:
    if prediction is None:
        return None
    values = np.asarray(prediction.probabilities)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("replay received malformed prediction probabilities")
    return tuple(float(value) for value in values)


def replay_session(packets: Iterable[FramePacket], detector: ReplayDetector) -> tuple[ReplayRecord, ...]:
    """Replay every packet against its source timestamp, never wall-clock speed."""
    records: list[ReplayRecord] = []
    source_id: str | None = None
    previous_timestamp: int | None = None
    previous_frame_index: int | None = None
    for packet in packets:
        if not isinstance(packet, FramePacket):
            raise TypeError("packets must contain FramePacket values")
        if source_id is None:
            source_id = packet.source_id
        elif packet.source_id != source_id:
            raise ValueError("one replay_session cannot join different sources")
        if previous_timestamp is not None and packet.timestamp_ms <= previous_timestamp:
            raise ValueError("replay packet timestamps must strictly increase")
        if previous_frame_index is not None and packet.frame_index <= previous_frame_index:
            raise ValueError("replay frame indices must strictly increase")
        detection = detector.process(packet, current_time_ms=packet.timestamp_ms)
        raw_class = None if detection.raw_prediction is None or detection.raw_prediction.class_id is None else int(detection.raw_prediction.class_id)
        smooth_class = None if detection.smoothed_prediction is None or detection.smoothed_prediction.class_id is None else int(detection.smoothed_prediction.class_id)
        records.append(ReplayRecord(
            source_id=packet.source_id,
            frame_index=packet.frame_index,
            timestamp_ms=packet.timestamp_ms,
            system_status=detection.system_status.value,
            calibration_status=detection.calibration_status,
            raw_probabilities=_probabilities(detection.raw_prediction),
            raw_class_id=raw_class,
            smoothed_probabilities=_probabilities(detection.smoothed_prediction),
            smoothed_class_id=smooth_class,
        ))
        previous_timestamp = packet.timestamp_ms
        previous_frame_index = packet.frame_index
    return tuple(records)


def compare_replays(
    expected: Sequence[ReplayRecord],
    actual: Sequence[ReplayRecord],
    *,
    atol: float = 1e-6,
) -> dict[str, object]:
    """Compare all scheduled timestamps and report, rather than drop, mismatches."""
    if isinstance(atol, bool) or not isinstance(atol, (int, float)) or not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be a finite nonnegative number")
    mismatches: list[dict[str, object]] = []
    if len(expected) != len(actual):
        mismatches.append({"field": "record_count", "expected": len(expected), "actual": len(actual)})
    for index in range(max(len(expected), len(actual))):
        if index >= len(expected):
            mismatches.append({"index": index, "field": "unexpected_record", "actual": actual[index].timestamp_ms})
            continue
        if index >= len(actual):
            mismatches.append({"index": index, "field": "missing_record", "expected": expected[index].timestamp_ms})
            continue
        left, right = expected[index], actual[index]
        for field in (
            "source_id", "frame_index", "timestamp_ms", "system_status", "calibration_status",
            "raw_class_id", "smoothed_class_id",
        ):
            if getattr(left, field) != getattr(right, field):
                mismatches.append({
                    "index": index,
                    "timestamp_ms": left.timestamp_ms,
                    "field": field,
                    "expected": getattr(left, field),
                    "actual": getattr(right, field),
                })
        for field in ("raw_probabilities", "smoothed_probabilities"):
            expected_values = getattr(left, field)
            actual_values = getattr(right, field)
            if (expected_values is None) != (actual_values is None):
                mismatches.append({
                    "index": index,
                    "timestamp_ms": left.timestamp_ms,
                    "field": field,
                    "expected": expected_values,
                    "actual": actual_values,
                })
            elif expected_values is not None and actual_values is not None and not np.allclose(
                expected_values, actual_values, atol=atol, rtol=0, equal_nan=False
            ):
                mismatches.append({
                    "index": index,
                    "timestamp_ms": left.timestamp_ms,
                    "field": field,
                    "expected": expected_values,
                    "actual": actual_values,
                })
    return {
        "match": not mismatches,
        "expected_records": len(expected),
        "actual_records": len(actual),
        "atol": float(atol),
        "mismatches": mismatches,
    }
