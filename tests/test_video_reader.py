import json

import cv2
import numpy as np
import pytest

from src.preprocessing.video_reader import VideoReader


@pytest.mark.parametrize("fps,indices", [
    (15, list(range(15))),
    (24, [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23]),
    (30, [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29]),
])
def test_real_avi_samples_source_grid_without_upsampling(tmp_path, fps, indices):
    path = tmp_path / f"{fps}.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    assert writer.isOpened()
    try:
        for index in range(fps):
            writer.write(np.full((48, 64, 3), index * 6, dtype=np.uint8))
    finally:
        writer.release()
    with VideoReader() as reader:
        packets = list(reader.iter_frames(path))
    assert [packet.frame_index for packet in packets] == indices
    assert [packet.timestamp_ms for packet in packets] == pytest.approx(
        [index * 1000 / fps for index in indices], abs=1
    )
    assert all(a.timestamp_ms < b.timestamp_ms for a, b in zip(packets, packets[1:]))
    assert all(abs(float(packet.image_bgr.mean()) - packet.frame_index * 6) < 3 for packet in packets)
    assert len({packet.source_id for packet in packets}) == 1
    assert reader.stats["decoded_frames"] == fps
    assert reader.stats["emitted_frames"] == len(indices)
    assert reader.stats["dropped_frames"] == fps - len(indices)
    assert reader.stats["status"] == "EOF"
    assert reader.stats["error"] is None
    assert reader.stats["duration_s"] == pytest.approx(1)
    assert reader.stats["fps_reported"] == fps
    assert reader.stats["timestamp_method"] == "CAP_PROP_POS_MSEC"
    assert reader.stats["wall_seconds"] >= 0
    assert reader.stats["capture_released"] is True
    json.dumps(reader.stats, allow_nan=False)


class Capture:
    """Backend double for corrupt timing and capture failures AVI cannot express."""

    def __init__(self, timestamps, *, fps=30, count=None, opened=True):
        self.timestamps = timestamps
        self.fps = fps
        self.count = len(timestamps) if count is None else count
        self.opened = opened
        self.position = 0
        self.released = False

    def isOpened(self):
        return self.opened and not self.released

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.count
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.timestamps[self.position - 1]
        return 0

    def set(self, prop, value):
        return True

    def read(self):
        if self.released or self.position == len(self.timestamps):
            return False, None
        image = np.full((4, 6, 3), self.position, dtype=np.uint8)
        self.position += 1
        return True, image

    def release(self):
        self.released = True


def backend(monkeypatch, capture):
    monkeypatch.setattr("src.preprocessing.video_reader.cv2.VideoCapture", lambda source: capture)
    return capture


@pytest.mark.parametrize("timestamps", [[0, 0], [0, -1], [0, float("nan")], [0, float("inf")], [0, 0.1], [0, 50, 49]])
def test_bad_timestamps_fail_even_on_frames_sampling_would_drop(monkeypatch, timestamps):
    capture = backend(monkeypatch, Capture(timestamps))
    reader = VideoReader()
    with pytest.raises(RuntimeError):
        list(reader.iter_frames("unreliable.avi", target_fps=1))
    assert reader.stats["status"] == "ERROR"
    assert reader.stats["error"]
    assert capture.released
    json.dumps(reader.stats, allow_nan=False)


def test_explicit_constant_fps_fallback_records_real_indices(monkeypatch):
    capture = backend(monkeypatch, Capture([0, 0, 0, 0], fps=30))
    reader = VideoReader()
    packets = list(reader.iter_frames("verified.avi", constant_fps_verified=True))
    assert [packet.frame_index for packet in packets] == [0, 2, 3]
    assert [packet.timestamp_ms for packet in packets] == [0, 67, 100]
    assert reader.stats["timestamp_method"] == "frame_index/fps"
    assert reader.stats["status"] == "EOF"
    assert capture.released


@pytest.mark.parametrize("field,value", [("fps", 0), ("fps", -2), ("fps", float("nan")), ("fps", float("inf")), ("count", 0), ("count", -1), ("count", float("nan")), ("count", float("inf")), ("fps", 1e-320)])
def test_unusable_metadata_fails_and_releases(monkeypatch, field, value):
    capture = backend(monkeypatch, Capture([0, 30], **{field: value}))
    reader = VideoReader()
    with pytest.raises((ValueError, RuntimeError)):
        list(reader.iter_frames("bad.avi", constant_fps_verified=True))
    assert reader.stats["status"] == "ERROR"
    assert reader.stats["error"]
    assert capture.released
    json.dumps(reader.stats, allow_nan=False)


def test_source_timestamp_beyond_duration_tolerance_is_rejected(monkeypatch):
    capture = backend(monkeypatch, Capture([0, 500], fps=30))
    reader = VideoReader()
    with pytest.raises(RuntimeError):
        list(reader.iter_frames("bad-duration.avi", constant_fps_verified=True))
    assert reader.stats["status"] == "ERROR"
    assert capture.released


def test_failed_decode_before_declared_end_is_not_eof(monkeypatch):
    capture = backend(monkeypatch, Capture([0, 33], count=10))
    reader = VideoReader()
    with pytest.raises(RuntimeError):
        list(reader.iter_frames("truncated.avi"))
    assert reader.stats["decoded_frames"] == 2
    assert reader.stats["status"] == "ERROR"
    assert reader.stats["error"]
    assert capture.released


def test_unopened_source_is_an_error(monkeypatch):
    capture = backend(monkeypatch, Capture([], opened=False))
    reader = VideoReader()
    with pytest.raises(RuntimeError):
        list(reader.iter_frames("missing.avi"))
    assert reader.stats["status"] == "ERROR"
    assert capture.released


@pytest.mark.parametrize("target", [0, -1, 21, float("nan"), float("inf")])
def test_invalid_target_is_rejected_for_both_sources(target):
    for iterator in (VideoReader().iter_frames("unused.avi", target), VideoReader().iter_camera(target_fps=target)):
        with pytest.raises(ValueError):
            next(iterator)


def test_close_stops_prefix_without_claiming_complete_and_reader_is_reusable(monkeypatch):
    capture = backend(monkeypatch, Capture([0, 33, 67]))
    reader = VideoReader()
    iterator = reader.iter_frames("prefix.avi")
    assert next(iterator).frame_index == 0
    reader.close()
    reader.close()
    assert capture.released
    assert list(iterator) == []
    assert reader.stats["status"] == "STOPPED"
    assert reader.stats["decoded_frames"] == 1
    replacement = backend(monkeypatch, Capture([0, 33, 67]))
    assert [p.frame_index for p in reader.iter_frames("new.avi")] == [0, 2]
    assert reader.stats["status"] == "EOF"
    assert replacement.released


def test_generator_close_releases_and_rejects_simultaneous_iterator(monkeypatch):
    capture = backend(monkeypatch, Capture([0, 33, 67]))
    with VideoReader() as reader:
        first = reader.iter_frames("a.avi")
        next(first)
        with pytest.raises(RuntimeError):
            next(reader.iter_frames("b.avi"))
        assert reader.stats["status"] == "RUNNING"
        first.close()
        assert capture.released
        assert reader.stats["status"] == "STOPPED"


def test_context_exit_stops_live_iterator(monkeypatch):
    capture = backend(monkeypatch, Capture([0, 33, 67]))
    with VideoReader() as reader:
        iterator = reader.iter_frames("a.avi")
        next(iterator)
    assert capture.released
    assert reader.stats["status"] == "STOPPED"
    assert list(iterator) == []


def test_camera_uses_read_completion_time_and_disconnect_is_error(monkeypatch):
    capture = backend(monkeypatch, Capture([0] * 5))
    times = iter([1_000_000_000, 1_010_000_000, 1_050_000_000, 1_099_000_000, 1_100_000_000])
    monkeypatch.setattr("src.preprocessing.video_reader.time.monotonic_ns", lambda: next(times))
    high_resolution_times = iter([1_000_000_000, 1_010_000_000, 1_050_000_000, 1_099_000_000, 1_100_000_000])
    monkeypatch.setattr("src.preprocessing.video_reader.time.perf_counter_ns", lambda: next(high_resolution_times))
    reader = VideoReader()
    iterator = reader.iter_camera()
    packets = [next(iterator), next(iterator), next(iterator)]
    assert [p.frame_index for p in packets] == [0, 2, 4]
    assert [p.timestamp_ms for p in packets] == [1000, 1050, 1100]
    with pytest.raises(RuntimeError):
        next(iterator)
    assert reader.stats["status"] == "ERROR"
    assert reader.stats["error"]
    assert reader.stats["decoded_frames"] == 5
    assert reader.stats["emitted_frames"] == 3
    assert reader.stats["dropped_frames"] == 2
    assert capture.released


def test_camera_uses_high_resolution_monotonic_clock_when_windows_tick_repeats(monkeypatch):
    capture = backend(monkeypatch, Capture([0] * 3))
    coarse = iter([1_000_000_000, 1_000_000_000, 1_062_500_000])
    precise = iter([2_000_000_000, 2_005_000_000, 2_050_000_000])
    monkeypatch.setattr("src.preprocessing.video_reader.time.monotonic_ns", lambda: next(coarse))
    monkeypatch.setattr("src.preprocessing.video_reader.time.perf_counter_ns", lambda: next(precise))
    reader = VideoReader()
    stream = reader.iter_camera()
    try:
        frames = [next(stream), next(stream)]
        assert [frame.timestamp_ms for frame in frames] == [2000, 2050]
        assert [frame.frame_index for frame in frames] == [0, 2]
    finally:
        stream.close()
    assert capture.released


def test_valid_variable_timestamps_use_source_origin_not_frame_rate(monkeypatch):
    backend(monkeypatch, Capture([1000, 1020, 1060, 1080, 1110], fps=20))
    reader = VideoReader()
    packets = list(reader.iter_frames("variable.avi"))
    assert [p.frame_index for p in packets] == [0, 2, 4]
    assert [p.timestamp_ms for p in packets] == [1000, 1060, 1110]
    assert reader.stats["timestamp_method"] == "CAP_PROP_POS_MSEC"


def test_old_stopped_iterator_cannot_consume_or_close_reused_reader(monkeypatch):
    old_capture = backend(monkeypatch, Capture([0, 33, 67]))
    reader = VideoReader()
    old = reader.iter_frames("old.avi")
    next(old)
    reader.close()
    new_capture = backend(monkeypatch, Capture([0, 33, 67]))
    new = reader.iter_frames("new.avi")
    assert next(new).frame_index == 0
    assert list(old) == []
    assert reader.stats["status"] == "RUNNING"
    assert [p.frame_index for p in new] == [2]
    assert reader.stats["status"] == "EOF"
    assert old_capture.released and new_capture.released
