"""Download selected UTA members using strict HTTP ranges, never entire ZIPs."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import zipfile
import zlib

import requests

OFFICIAL_FOLDER = "https://drive.google.com/drive/folders/1d_QwgpMXnLY_FmLYXDn7TLcw0D-svXEl"
OFFICIAL_PAGE = "https://sites.google.com/view/utarldd/home"
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
SOURCE_LABELS = {0: 0, 5: 1, 10: 2}


def record_source(dataset: str, url: str, terms: str) -> dict:
    return {"dataset": dataset, "url": url, "accessed_at_utc": datetime.now(timezone.utc).isoformat(), "terms": terms}


def list_official_archives() -> list[dict]:
    response = requests.get(OFFICIAL_FOLDER, timeout=(15, 60))
    response.raise_for_status()
    match = re.search(r"window\['_DRIVE_ivd'\]\s*=\s*'([^']+)'", response.text)
    if not match:
        raise RuntimeError("Official Drive index format changed; cannot discover archive IDs")
    encoded = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), match.group(1))
    data = json.loads(encoded.replace("\\/", "/"))
    result = []
    for item in data[0]:
        if re.fullmatch(r"Fold[1-5]_part[12]\.zip", item[2]):
            result.append({"file_id": item[0], "name": item[2], "size_bytes": int(item[13])})
    if len(result) != 10:
        raise RuntimeError(f"Expected ten official archives, discovered {len(result)}")
    return sorted(result, key=lambda row: row["name"])


def download_url(file_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
        raise ValueError("Invalid Drive file ID")
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def open_range(session: requests.Session, url: str, start: int, size: int, total: int):
    if start < 0 or size <= 0 or start + size > total:
        raise ValueError("Byte range lies outside the archive")
    end = start + size - 1
    response = session.get(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                           stream=True, timeout=(15, 120))
    expected = f"bytes {start}-{end}/{total}"
    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
        status, actual = response.status_code, response.headers.get("Content-Range")
        response.close()
        raise RuntimeError(f"Range request refused: HTTP {status}, Content-Range={actual}; will NOT download the full archive")
    return response


class RangeFile(io.RawIOBase):
    """Seekable metadata-only view used by zipfile to parse ZIP64 indexes."""
    def __init__(self, url: str, size: int):
        super().__init__()
        self.url, self.size, self.position = url, size, 0
        self.session = requests.Session()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset: int, whence: int = 0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if not 0 <= position <= self.size:
            raise ValueError("Seek outside archive")
        self.position = position
        return position

    def read(self, size: int = -1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size == 0:
            return b""
        if size > 4 * 1024 * 1024:
            raise RuntimeError("ZIP metadata request exceeds 4 MiB; refusing full download")
        with open_range(self.session, self.url, self.position, size, self.size) as response:
            data = response.raw.read(size + 1)
        if len(data) != size:
            raise RuntimeError("Truncated metadata range")
        self.position += size
        return data

    def close(self):
        self.session.close()
        super().close()


def inventory_archive(archive: dict, *, url: str | None = None) -> list[dict]:
    name_match = re.fullmatch(r"Fold([1-5])_part([12])\.zip", archive["name"])
    if not name_match:
        raise ValueError("Archive is not an official FoldN_partM ZIP")
    fold = int(name_match.group(1))
    records = []
    with RangeFile(url or download_url(archive["file_id"]), archive["size_bytes"]) as remote:
        with zipfile.ZipFile(remote) as zipped:
            for item in zipped.infolist():
                path = PurePosixPath(item.filename)
                if item.is_dir() or path.suffix.lower() not in VIDEO_SUFFIXES:
                    continue
                match = re.fullmatch(r"(0|5|10)(?:_([1-9][0-9]*))?", path.stem)
                if len(path.parts) != 3 or path.parts[0] != archive["name"][:-4] or not re.fullmatch(r"[0-9]{2}", path.parent.name) or not match:
                    raise ValueError(f"Unrecognized video member: {item.filename}")
                label = int(match.group(1))
                records.append({
                    "subject_id": path.parent.name, "video_id": f"{path.parent.name}_{path.stem}",
                    "source_label": label, "label_id": SOURCE_LABELS[label], "part_id": match.group(2),
                    "fold_id": fold, "source_archive": archive["name"], "source_file_id": archive["file_id"],
                    "archive_size": archive["size_bytes"], "member_path": item.filename,
                    "size_bytes": item.file_size, "compressed_bytes": item.compress_size,
                    "header_offset": item.header_offset, "compression": item.compress_type, "crc32": item.CRC,
                    "relative_path": f"{path.parent.name}/{path.name}",
                })
    return records


def select_subset(records: list[dict], *, subjects_per_fold: int = 3) -> list[dict]:
    if not isinstance(subjects_per_fold, int) or not 1 <= subjects_per_fold <= 3:
        raise ValueError("Only 1–3 subjects per fold are permitted (at most 45 videos)")
    grouped = defaultdict(list)
    for record in records:
        grouped[record["subject_id"]].append(record)
    selected = []
    for fold in range(1, 6):
        eligible = []
        for subject, entries in grouped.items():
            if len(entries) == 3 and {r["source_label"] for r in entries} == {0, 5, 10} and {r["fold_id"] for r in entries} == {fold}:
                eligible.append((sum(r["size_bytes"] for r in entries), subject, entries))
        eligible.sort(key=lambda row: (row[0], row[1]))
        if len(eligible) < subjects_per_fold:
            raise ValueError(f"Fold {fold} lacks {subjects_per_fold} complete, unsegmented subjects")
        for _, _, entries in eligible[:subjects_per_fold]:
            selected.extend(sorted(entries, key=lambda row: row["source_label"]))
    return selected


def digest_file(path: Path) -> tuple[str, int]:
    digest, crc = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            crc = zlib.crc32(chunk, crc)
    return digest.hexdigest(), crc & 0xFFFFFFFF


def _member_paths(record: dict, root: Path) -> tuple[Path, Path, Path, Path]:
    relative = PurePosixPath(record["relative_path"])
    if not re.fullmatch(r"[0-9]{2}/(?:0|5|10)(?:_[1-9][0-9]*)?\.[a-z0-9]+", str(relative), re.IGNORECASE) or relative.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError("Unsafe output path")
    destination = root.joinpath(*relative.parts)
    paths = (destination, destination.with_suffix(destination.suffix + ".part"),
             destination.with_suffix(destination.suffix + ".receipt.json"),
             destination.with_suffix(destination.suffix + ".receipt.json.partial"))
    for path in paths:
        if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Unsafe output path or link: {path}")
        if path.exists() and not path.is_file():
            raise ValueError(f"Output path is not a regular file: {path}")
    return paths


def discard_partials(records: list[dict], root: str | Path) -> None:
    """Reclaim only planned temporary files before measuring space for a restart."""
    root = Path(root).resolve()
    for record in records:
        _, partial, _, receipt_partial = _member_paths(record, root)
        partial.unlink(missing_ok=True)
        receipt_partial.unlink(missing_ok=True)


def download_member(record: dict, root: str | Path, *, url: str | None = None, reserve_bytes: int = 8 * 1024**3) -> dict:
    root = Path(root).resolve()
    destination, part_path, receipt_path, receipt_partial = _member_paths(record, root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path.unlink(missing_ok=True)
    receipt_partial.unlink(missing_ok=True)
    if destination.exists():
        sha, crc = digest_file(destination)
        if destination.stat().st_size != record["size_bytes"] or crc != record["crc32"]:
            raise ValueError(f"Existing file is different; refusing overwrite: {destination}")
        receipt = {**record, "sha256": sha, "status": "verified", "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    else:
        if shutil.disk_usage(root).free < record["size_bytes"] + reserve_bytes:
            raise RuntimeError("Insufficient free disk space including reserve")
        total = record["archive_size"]
        target_url = url or download_url(record["source_file_id"])
        with requests.Session() as session:
            with open_range(session, target_url, record["header_offset"], 30, total) as response:
                header = response.raw.read(31)
            if len(header) != 30 or header[:4] != b"PK\x03\x04":
                raise ValueError("Invalid ZIP local header")
            fields = struct.unpack("<4s5H3I2H", header)
            if fields[2] & 1 or fields[3] != record["compression"]:
                raise ValueError("Encrypted or inconsistent ZIP member")
            if record["compression"] not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise ValueError("Unsupported ZIP compression")
            start = record["header_offset"] + 30 + fields[-2] + fields[-1]
            decompressor = zlib.decompressobj(-zlib.MAX_WBITS) if record["compression"] == zipfile.ZIP_DEFLATED else None
            digest, crc, written, transferred = hashlib.sha256(), 0, 0, 0
            with open_range(session, target_url, start, record["compressed_bytes"], total) as response, part_path.open("xb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    transferred += len(chunk)
                    data = decompressor.decompress(chunk) if decompressor else chunk
                    written += len(data)
                    if written > record["size_bytes"]:
                        raise ValueError("Decompressed member exceeds advertised size")
                    if shutil.disk_usage(root).free < reserve_bytes + len(data):
                        raise RuntimeError("Disk reserve reached; leaving incomplete .part, not a video")
                    handle.write(data)
                    digest.update(data)
                    crc = zlib.crc32(data, crc)
                if decompressor:
                    tail = decompressor.flush()
                    handle.write(tail)
                    digest.update(tail)
                    crc = zlib.crc32(tail, crc)
                    written += len(tail)
                    if not decompressor.eof or decompressor.unused_data:
                        raise ValueError("Incomplete or overlong deflate stream")
            if transferred != record["compressed_bytes"] or written != record["size_bytes"]:
                raise ValueError("Member length mismatch")
            if crc & 0xFFFFFFFF != record["crc32"]:
                raise ValueError("CRC mismatch; member was not published")
            part_path.replace(destination)
            receipt = {**record, "sha256": digest.hexdigest(), "status": "verified", "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    with receipt_partial.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
    receipt_partial.replace(receipt_path)
    return receipt
