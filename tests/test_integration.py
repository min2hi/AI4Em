from dataclasses import replace

import numpy as np
import pytest

from src.contracts import (
    DetectionResult,
    DriverState,
    FeatureSample,
    FramePacket,
    Prediction,
    SystemStatus,
)
from src.evaluation.replay import ReplayRecord, compare_replays, replay_session


class Detector:
    def __init__(self):
        self.times = []
        self.resets = 0
        self.last_feature_sample = None

    def reset_session(self):
        self.resets += 1
        self.last_feature_sample = None

    def process(self, packet, *, current_time_ms=None):
        self.times.append(current_time_ms)
        valid = packet.frame_index != 1
        value = 0.3 if valid else np.nan
        self.last_feature_sample = FeatureSample(
            packet.timestamp_ms,
            packet.frame_index,
            packet.source_id,
            value,
            value,
            value,
            0.1 if valid else np.nan,
            0.0 if valid else np.nan,
            0.0 if valid else np.nan,
            0.0 if valid else np.nan,
            valid,
            valid,
            valid,
            valid,
            valid,
            0.0 if valid else np.nan,
        )
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
    assert detector.resets == 1
    assert detector.times == [0, 67, 133]
    assert [record.timestamp_ms for record in records] == [0, 67, 133]
    assert records[1].system_status == "NO_FACE"
    assert records[1].feature_values == (None,) * 8
    assert records[1].feature_validity == (False,) * 5
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


@pytest.mark.parametrize("bad_prediction", [
    Prediction(134, np.array([.2, .3, .5], dtype=np.float32), DriverState.DROWSY, True, "", "model"),
    Prediction(133, np.array([.2, .2, .2], dtype=np.float32), DriverState.ALERT, True, "", "model"),
])
def test_replay_rejects_future_or_malformed_predictions(bad_prediction):
    class BadDetector(Detector):
        def process(self, packet, *, current_time_ms=None):
            return DetectionResult(bad_prediction, None, SystemStatus.READY, {}, "COMPLETE")

    with pytest.raises(ValueError):
        replay_session([packets()[-1]], BadDetector())


def record(timestamp, probabilities=(.2, .3, .5)):
    return ReplayRecord(
        source_id="video",
        frame_index=timestamp,
        timestamp_ms=timestamp,
        system_status="READY",
        calibration_status="COMPLETE",
        feature_values=(0.3, 0.3, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0),
        feature_validity=(True, True, True, True, True),
        raw_timestamp_ms=timestamp,
        raw_model_id="model",
        raw_probabilities=probabilities,
        raw_class_id=2,
        smoothed_timestamp_ms=None,
        smoothed_model_id=None,
        smoothed_probabilities=None,
        smoothed_class_id=None,
    )


def test_parity_accepts_declared_numeric_tolerance():
    expected = [record(0)]
    actual = [replace(
        record(0, (.2000004, .2999998, .4999998)),
        feature_values=(.3000004, .3, .3, .1, 0., 0., 0., 0.),
    )]
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


def test_parity_rejects_prediction_cadence_or_model_identity_drift():
    expected = [record(100)]
    actual = [replace(record(100), raw_timestamp_ms=99, raw_model_id="other")]
    report = compare_replays(expected, actual)
    assert not report["match"]
    fields = [item["field"] for item in report["mismatches"]]
    assert "raw_timestamp_ms" in fields
    assert "raw_model_id" in fields


def test_parity_reports_feature_value_and_validity_drift():
    expected = [record(100)]
    actual = [replace(
        record(100),
        feature_values=(.2, .3, .3, .1, 0., 0., 0., 0.),
        feature_validity=(True, False, True, True, True),
    )]
    report = compare_replays(expected, actual)
    fields = [item["field"] for item in report["mismatches"]]
    assert "feature_values" in fields
    assert "feature_validity" in fields


def test_replay_rejects_feature_trace_from_another_packet():
    class MisalignedDetector(Detector):
        def process(self, packet, *, current_time_ms=None):
            result = super().process(packet, current_time_ms=current_time_ms)
            self.last_feature_sample = replace(self.last_feature_sample, frame_index=99)
            return result

    with pytest.raises(ValueError, match="packet identity"):
        replay_session([packets()[0]], MisalignedDetector())
