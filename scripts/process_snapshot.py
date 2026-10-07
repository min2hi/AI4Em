"""Freeze and process every working-snapshot member sequentially.

Single writer only: never run alongside another builder in the output directory.
Existing pairs are reused only after the builder verifies their commit markers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import PROJECT_ROOT, load_config
from src.preprocessing.snapshot import run_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=PROJECT_ROOT / 'runs/phase8')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs/preprocessing.yaml')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        report = run_snapshot(args.config, args.manifest or config['manifest_path'],
                              args.run_dir, args.output_dir or config['output_dir'])
    except Exception as exc:
        # Refused freezes leave the previous freeze and report untouched.
        print(json.dumps({'status': 'failed', 'error': str(exc)}, indent=2,
                         sort_keys=True, allow_nan=False))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return int(report['status'] != 'completed')


if __name__ == '__main__':
    raise SystemExit(main())
