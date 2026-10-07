from dataclasses import replace

import numpy as np
import pytest

from src.contracts import DetectionResult, DriverState, FramePacket, Prediction, SystemStatus
from src.evaluation.replay import ReplayRecord, compare_replays, replay_session


class Detector:
    def __init__(self):
        self.times = []

    def process(self, packet, *, current_time_ms=None):
        self.times.append(current_time_ms)
        if packet.frame_index == 1:
            return DetectionResult(None, None, SystemStatus.NO_FACE, {}, "COMPLETE")
        probabilities = np.array([.2, .3, .5], dtype=np.float32)
        raw = Prediction(packet.timestamp_ms, probabilities, DriverState.DROWSY,
                         True, "", "model")
        return DetectionResult(raw, None, SystemStatus.READY, {}, "COMPLETE")


def packets():
    return [
        FramePacket(np.zeros((2, 2, 3), dtype=np.uint8), timestamp, index, "video")
        for index, timestamp in enumerate((0, 67, 133))
    ]


def test_replay_uses_source_time_and_preserves_unavailable_timestamps():
    detector = Detector()
    records = replay_session(packets(), detector)
    assert detector.times == [0, 67, 133]
    assert [record.timestamp_ms for record in records] == [0, 67, 133]
    assert records[1].system_status == "NO_FACE"
    assert records[1].raw_probabilities is None
    assert records[2].raw_class_id == 2


def test_replay_rejects_source_join_and_nonmonotonic_input():
    source_change = packets()
    source_change[1] = replace(source_change[1], source_id="other")
    with pytest.raises(ValueError, match="different sources"):
        replay_session(source_change, Detector())
    backwards = packets()
    backwards[2] = replace(backwards[2], timestamp_ms=50)
    with pytest.raises(ValueError, match="strictly increase"):
        replay_session(backwards, Detector())


def record(timestamp, probabilities=(.2, .3, .5)):
    return ReplayRecord("video", timestamp, timestamp, "READY", "COMPLETE",
                        probabilities, 2, None, None)


def test_parity_accepts_declared_numeric_tolerance():
    expected = [record(0)]
    actual = [record(0, (.2000004, .2999998, .4999998))]
    report = compare_replays(expected, actual, atol=1e-6)
    assert report["match"]
    assert report["mismatches"] == []


def test_parity_reports_missing_timestamp_and_probability_difference():
    expected = [record(0), record(100)]
    actual = [record(0, (.1, .4, .5))]
    report = compare_replays(expected, actual, atol=1e-6)
    assert not report["match"]
    fields = [item["field"] for item in report["mismatches"]]
    assert "record_count" in fields
    assert "raw_probabilities" in fields
    assert "missing_record" in fields
