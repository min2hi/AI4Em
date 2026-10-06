import hashlib
import io
import os
from types import SimpleNamespace
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.datasets.acquisition import inventory_archive, select_subset, download_member


def test_selects_complete_subjects_balanced_across_folds():
    records = []
    for fold in range(1, 6):
        for subject in range(4):
            for label in [0, 5, 10]:
                records.append({"subject_id": f"{fold}{subject}", "fold_id": fold, "source_label": label,
                                "size_bytes": 100 * (subject + 1), "video_id": f"{fold}{subject}_{label}"})
    selected = select_subset(records, subjects_per_fold=3)
    assert len(selected) == 45
    assert {r["subject_id"] for r in selected} == {f"{fold}{subject}" for fold in range(1,6) for subject in range(3)}
    assert all(sum(r["fold_id"] == fold for r in selected) == 9 for fold in range(1,6))


def test_subject_with_missing_or_duplicate_state_is_not_silently_selected():
    records = [
        {"subject_id": "01", "fold_id": 1, "source_label": label, "size_bytes": 1, "video_id": str(i)}
        for i, label in enumerate([0, 5, 10, 10])
    ]
    with pytest.raises(ValueError):
        select_subset(records, subjects_per_fold=1)


@pytest.fixture
def range_server():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Fold1_part1/01/0.mp4", b"video payload" * 1000)
    payload = buffer.getvalue()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requested = self.headers.get("Range", "")
            if self.path == "/ignore":
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            bounds = requested.removeprefix("bytes=").split("-")
            start, end = map(int, bounds)
            data = payload[start:end+1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", len(payload)
    server.shutdown()
    server.server_close()
    thread.join()


def test_range_download_extracts_only_member_and_verifies_receipt(range_server, tmp_path):
    url, size = range_server
    rows = inventory_archive({"name": "Fold1_part1.zip", "file_id": "fixture", "size_bytes": size}, url=url)
    assert [(r["subject_id"], r["source_label"], r["fold_id"]) for r in rows] == [("01", 0, 1)]
    receipt = download_member(rows[0], tmp_path, url=url, reserve_bytes=0)
    output = tmp_path / "01" / "0.mp4"
    assert output.read_bytes() == b"video payload" * 1000
    assert receipt["size_bytes"] == 13000
    assert receipt["sha256"] == hashlib.sha256(b"video payload" * 1000).hexdigest()
    assert not output.with_suffix(".mp4.part").exists()


def test_server_ignoring_range_is_refused(range_server):
    url, size = range_server
    with pytest.raises(RuntimeError):
        inventory_archive({"name":"Fold1_part1.zip", "file_id":"fixture", "size_bytes":size}, url=url+"/ignore")


def test_corrupt_member_is_not_published(range_server, tmp_path):
    url, size = range_server
    row = inventory_archive({"name":"Fold1_part1.zip", "file_id":"fixture", "size_bytes":size}, url=url)[0]
    row["crc32"] ^= 1
    with pytest.raises(ValueError):
        download_member(row, tmp_path, url=url, reserve_bytes=0)
    assert not (tmp_path / "01" / "0.mp4").exists()


@pytest.mark.parametrize("suffix", [".part", ".receipt.json"])
def test_temporary_and_receipt_hardlinks_do_not_modify_outside_file(range_server, tmp_path, suffix):
    url, size = range_server
    row = inventory_archive({"name":"Fold1_part1.zip", "file_id":"fixture", "size_bytes":size}, url=url)[0]
    raw = tmp_path / "raw"
    (raw / "01").mkdir(parents=True)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"private payload")
    os.link(outside, raw / "01" / ("0.mp4" + suffix))
    download_member(row, raw, url=url, reserve_bytes=0)
    assert outside.read_bytes() == b"private payload"
    assert (raw / "01" / "0.mp4").read_bytes() == b"video payload" * 1000


def test_stale_partial_is_reclaimed_before_retry_disk_admission(range_server, tmp_path, monkeypatch):
    from src.datasets import acquisition
    url, size = range_server
    row = inventory_archive({"name":"Fold1_part1.zip", "file_id":"fixture", "size_bytes":size}, url=url)[0]
    (tmp_path / "01").mkdir()
    (tmp_path / "01" / "0.mp4.part").write_bytes(b"x" * 6000)
    capacity = row["size_bytes"] + 4096
    def file_backed_disk_usage(root):
        used = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        return SimpleNamespace(total=capacity, used=used, free=capacity-used)
    monkeypatch.setattr(acquisition.shutil, "disk_usage", file_backed_disk_usage)
    download_member(row, tmp_path, url=url, reserve_bytes=4096)
    assert (tmp_path / "01" / "0.mp4").read_bytes() == b"video payload" * 1000
    assert not (tmp_path / "01" / "0.mp4.part").exists()
