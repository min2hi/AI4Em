"""Build full-EOF raw Parquet for explicitly selected working-snapshot videos.

Single writer only: do not run concurrent builders against one output directory.
A companion metadata commit marker and its Parquet hash establish completeness;
file existence alone is not sufficient. No bounded-prefix completion is supported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.preprocessing.builder import FeatureDatasetBuilder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/preprocessing.yaml")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "runs/phase7/preprocess")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--video-id", action="append", help="Explicit video ID; repeat to select several")
    selection.add_argument("--all-working-snapshot", action="store_true",
                           help="Explicitly process the entire working snapshot (not acquisition completeness)")
    parser.add_argument("--constant-fps-verified", action="store_true",
                        help="Allow timestamp fallback ONLY for a caller-independently-verified CFR source")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        manifest = pd.read_parquet(args.manifest or config["manifest_path"])
        output = args.output_dir or config["output_dir"]
        builder = FeatureDatasetBuilder(config, constant_fps_verified=args.constant_fps_verified)
        report = builder.build(manifest, output,
                               video_ids=tuple(args.video_id) if args.video_id else None)
    except Exception as exc:
        report = {"status": "failed", "error": str(exc), "videos": [],
                  "totals": {"completed": 0, "cached": 0, "failed": 0}}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    (args.report_dir / "report.json").write_text(encoded, encoding="utf-8")
    print(encoded)
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
