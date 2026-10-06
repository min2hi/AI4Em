"""Acquire at most 45 UTA videos without downloading whole official ZIPs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from src.config import PROJECT_ROOT
from src.datasets.acquisition import (OFFICIAL_FOLDER, OFFICIAL_PAGE, discard_partials, download_member,
    inventory_archive, list_official_archives, record_source, select_subset)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-per-fold", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2, choices=range(1, 4))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/raw/uta_rldd")
    parser.add_argument("--metadata-dir", type=Path, default=PROJECT_ROOT / "data/acquisition")
    parser.add_argument("--inventory", type=Path, help="Reuse previously verified remote inventory")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    metadata_dir = args.metadata_dir.resolve()
    metadata_dir.mkdir(parents=True, exist_ok=True)
    if args.inventory:
        raw = json.loads(args.inventory.read_text(encoding="utf-8"))
        archives = raw["archives"]
        # ZIP indexes are fetched again: offsets must refer to current source bytes.
    else:
        archives = list_official_archives()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        groups = list(pool.map(inventory_archive, archives))
    records = [row for group in groups for row in group]
    chosen = select_subset(records, subjects_per_fold=args.subjects_per_fold)
    plan = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": record_source("uta_rldd", OFFICIAL_PAGE, "Research use; cite authors; public images only with subject permission"),
        "folder_url": OFFICIAL_FOLDER, "strategy": "smallest complete subjects per official fold; storage-biased development subset, not full benchmark",
        "subjects_per_fold": args.subjects_per_fold,
        "subject_ids": sorted({r["subject_id"] for r in chosen}),
        "video_count": len(chosen), "expected_raw_bytes": sum(r["size_bytes"] for r in chosen),
        "expected_transfer_bytes": sum(r["compressed_bytes"] for r in chosen),
        "records": chosen,
    }
    (metadata_dir / "uta_source_inventory.json").write_text(json.dumps({"archives": archives, "records": records}, indent=2), encoding="utf-8")
    plan_path = metadata_dir / "uta_subset_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Planned {len(chosen)} videos / {len(plan['subject_ids'])} subjects; raw {plan['expected_raw_bytes']/1024**3:.2f} GiB", flush=True)
    if args.plan_only:
        return 0
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.rglob("*.zip")):
        raise ValueError("Do not retain full archives in the subset directory")
    existing = {str(p.relative_to(root).as_posix()) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".m4v", ".avi", ".mkv"}}
    planned = {row["relative_path"] for row in chosen}
    if existing - planned:
        raise ValueError("Raw directory contains videos outside the 1/4 plan; refusing to add more")
    discard_partials(chosen, root)
    reserve = 8 * 1024**3
    remaining = sum(row["size_bytes"] for row in chosen if row["relative_path"] not in existing)
    if shutil.disk_usage(root).free < remaining + reserve:
        raise RuntimeError("Not enough space for the planned subset plus 8 GiB reserve")
    receipts, errors = [], []
    def acquire(row):
        print(f"START {row['video_id']} ({row['size_bytes']/1024**2:.1f} MiB)", flush=True)
        return download_member(row, root, reserve_bytes=reserve)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(acquire, row): row for row in chosen}
        for future in as_completed(futures):
            row = futures[future]
            try:
                receipt = future.result()
                receipts.append(receipt)
                print(f"OK {row['video_id']} [{len(receipts)}/{len(chosen)}] SHA256 {receipt['sha256'][:12]}", flush=True)
            except Exception as exc:
                errors.append({"video_id": row["video_id"], "error": str(exc)})
                print(f"ERROR {row['video_id']}: {exc}", flush=True)
            (metadata_dir / "uta_download_report.json").write_text(json.dumps({"receipts": receipts, "errors": errors, "expected_videos":len(chosen)}, indent=2), encoding="utf-8")
    return 1 if errors or len(receipts) != len(chosen) else 0


if __name__ == "__main__":
    raise SystemExit(main())
