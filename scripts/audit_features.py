"""Independently audit one explicitly selected Phase8 snapshot (never extracts)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import PROJECT_ROOT
from src.preprocessing.audit import audit_snapshot
from src.preprocessing.builder import _read_json, _canonical


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--status-manifest', type=Path,
                        default=PROJECT_ROOT/'data/processed/extraction_status.parquet')
    args = parser.parse_args(argv)
    frozen = args.run_dir/'snapshot'
    try:
        snapshot = _read_json(frozen/'snapshot.json')
        report = audit_snapshot(frozen/'preprocessing.yaml', frozen/'manifest.parquet',
            args.output_dir or Path(snapshot['output_dir']), snapshot, _read_json(args.run_dir/'report.json'))
        if args.status_manifest.resolve() in {
                (frozen/'manifest.parquet').resolve(), Path(snapshot['original_manifest_path']).resolve()}:
            raise ValueError('Extraction status must not overwrite a source manifest')
        statuses = []
        sources = {row['video_id']:row for row in snapshot['sources']}
        for item in report['videos']:
            status = {**sources[item['video_id']],
                'source_status':sources[item['video_id']]['status'],
                'extraction_status':item['extraction_status'], 'audit_status':item['status'],
                'extraction_error':item.get('extraction_error'),
                'audit_error':item.get('error'), 'audited_row_count':item['row_count'],
                'snapshot_sha256':report['snapshot_sha256'],
                'extraction_fingerprint':report['extraction_fingerprint'],
                'manifest_row_sha256':item['manifest_row_sha256']}
            statuses.append(status)
        table = pa.Table.from_pandas(pd.DataFrame(statuses), preserve_index=False)
        table = table.replace_schema_metadata({**(table.schema.metadata or {}),
            b'snapshot_sha256':report['snapshot_sha256'].encode(),
            b'working_manifest_sha256':report['working_manifest_sha256'].encode(),
            b'manifest_file_sha256':report['manifest_file_sha256'].encode(),
            b'extraction_fingerprint':report['extraction_fingerprint'].encode(),
            b'source_root':str(snapshot['source_root']).encode(),
            b'snapshot_sources':_canonical(snapshot['sources']).encode()})
        args.status_manifest.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(table,args.status_manifest)
        pd.DataFrame(report['coverage']).to_csv(args.run_dir/'coverage.csv',index=False)
    except Exception as exc:
        report = dict(schema_version='feature_audit_v1',status='failed',error=str(exc))
    args.run_dir.mkdir(parents=True,exist_ok=True)
    encoded = json.dumps(report,indent=2,sort_keys=True,allow_nan=False)
    (args.run_dir/'audit.json').write_text(encoded,encoding='utf-8')
    print(encoded)
    return int(report['status']=='failed')


if __name__ == '__main__':
    raise SystemExit(main())
