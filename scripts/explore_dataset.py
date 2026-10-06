"""Build Parquet manifest and a reproducible report from the acquired UTA subset."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import PROJECT_ROOT, load_config
from src.datasets.manifest import build_manifest, validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/preprocessing.yaml")
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "data/acquisition/uta_subset_plan.json")
    parser.add_argument("--download-report", type=Path, default=PROJECT_ROOT / "data/acquisition/uta_download_report.json")
    parser.add_argument("--permissions", type=Path, default=PROJECT_ROOT / "data/acquisition/uta_image_permissions.json")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "runs/phase2")
    args = parser.parse_args()
    config = load_config(args.config)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    downloaded = json.loads(args.download_report.read_text(encoding="utf-8"))
    permissions = json.loads(args.permissions.read_text(encoding="utf-8"))["permissions"] if args.permissions.exists() else None
    frame, errors = build_manifest(config["dataset_root"], "uta_rldd", source_records=downloaded["receipts"], permission_map=permissions)
    output = config["manifest_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    validation_error = None
    try:
        validate_manifest(frame, expected_subjects=len(plan["subject_ids"]), expected_videos=plan["video_count"],
                          require_folds=True, expected_records=plan["records"])
        if downloaded["errors"] or len(downloaded["receipts"]) != plan["video_count"]:
            raise ValueError("Acquisition is incomplete; do not call Phase 2 complete")
        planned = {row["relative_path"]: row for row in plan["records"]}
        for receipt in downloaded["receipts"]:
            expected = planned.get(receipt["relative_path"])
            if receipt.get("status") != "verified" or expected is None or any(receipt.get(key) != value for key,value in expected.items()):
                raise ValueError("Download receipt differs from verified acquisition plan")
    except ValueError as exc:
        validation_error = str(exc)
    valid = frame[frame["status"] == "ok"]
    discovered_paths = set(frame["relative_path"])
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": "uta_rldd_development_quarter_subset",
        "selection_strategy": plan["strategy"], "physical_video_count": len(frame), "subject_count": int(frame["subject_id"].nunique()),
        "expected_video_count": plan["video_count"], "expected_subject_count": len(plan["subject_ids"]),
        "missing_planned_videos": [row["video_id"] for row in plan["records"] if row["relative_path"] not in discovered_paths],
        "download_errors": downloaded["errors"],
        "class_counts": {str(k): int(v) for k,v in valid.groupby("label_id").size().items()},
        "fold_subject_counts": {str(k): int(v) for k,v in valid.groupby("fold_id")["subject_id"].nunique().items()},
        "total_bytes": int(frame["size_bytes"].sum()), "total_hours": float(valid["duration_s"].sum()/3600),
        "fps_range": [float(valid["fps_reported"].min()), float(valid["fps_reported"].max())] if len(valid) else None,
        "duration_s_range": [float(valid["duration_s"].min()), float(valid["duration_s"].max())] if len(valid) else None,
        "resolution_counts": {f"{int(w)}x{int(h)}":int(n) for (w,h),n in valid.groupby(["width","height"]).size().items()},
        "validation_error": validation_error, "errors": errors,
        "decode_check": "three sampled frames per video (start/middle/end), not full-frame validation",
        "duration_method": "reported frame_count / reported FPS; VFR/timestamp verification continues in Phase 3",
        "protocol": "Official fold membership from ZIP index, restricted to three subjects per fold; NOT the full 60-subject benchmark",
        "split_status": "unassigned; train/val/test assignment belongs to Phase 9",
    }
    (report_dir / "exploration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (report_dir / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    frame.drop(columns=["sample_timestamps_ms"]).to_csv(report_dir / "manifest_summary.csv", index=False, encoding="utf-8-sig")
    if len(valid):
        figure, axes = plt.subplots(1,2, figsize=(10,4))
        valid["label_id"].value_counts().sort_index().plot.bar(ax=axes[0], title="Physical videos per class")
        valid["fps_reported"].plot.hist(ax=axes[1], title="Reported source FPS")
        figure.tight_layout()
        figure.savefig(report_dir / "dataset_summary.png", dpi=120)
        plt.close(figure)
    print(json.dumps(report, indent=2))
    return 1 if validation_error or errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
