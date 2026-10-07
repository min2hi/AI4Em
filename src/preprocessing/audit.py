"""Read-only, independently reopened working-snapshot feature audit."""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.config import PROJECT_ROOT, load_config
from src.datasets.acquisition import digest_file
from src.datasets.manifest import validate_working_manifest
from src.preprocessing.builder import (FeatureDatasetBuilder, STORAGE_SCHEMA, FLAG_FIELDS,
    FLOAT_FIELDS, _hash_json, _json_value)

COUNTS = (*FLAG_FIELDS, 'both_eyes_valid')
IDENTITY = ('dataset_name', 'subject_id', 'video_id', 'label_id', 'label_source')


def _audit_pair(path: Path, source: dict, metadata: dict, *, batch_size: int = 1024) -> dict:
    counts = Counter({key: 0 for key in COUNTS})
    count = 0
    first = last = last_index = None
    with pq.ParquetFile(path) as stored:
        if not stored.schema_arrow.equals(STORAGE_SCHEMA, check_metadata=True):
            raise ValueError('Exact Arrow schema mismatch')
        for batch in stored.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                if any(row[key] != source[key] for key in IDENTITY):
                    raise ValueError('Stored row identity/label differs from frozen source')
                index, timestamp = row['frame_index'], row['timestamp_ms']
                if (type(index) is not int or type(timestamp) is not int
                        or not 0 <= index <= 2**31-1 or not 0 <= timestamp <= 2**63-1
                        or last_index is not None and index <= last_index
                        or last is not None and timestamp <= last):
                    raise ValueError('Indices/timestamps must be strictly increasing and in bounds')
                if count == 0 and index != 0:
                    raise ValueError('The first emitted frame must have frame_index zero')
                if any(type(row[key]) is not bool for key in FLAG_FIELDS):
                    raise ValueError('Validity masks cannot be null')
                for key in FLOAT_FIELDS:
                    value = row[key]
                    if value is not None and not math.isfinite(value):
                        raise ValueError(f'{key} must be finite or null')
                if not row['face_detected'] and any(row[key] for key in FLAG_FIELDS[1:]):
                    raise ValueError('Valid measurements require a detected face')
                for mask, field in (('left_eye_valid','ear_left'),('right_eye_valid','ear_right'),('mouth_valid','mar')):
                    value = row[field]
                    if row[mask] != (value is not None) or value is not None and value < 0:
                        raise ValueError(f'{field} null/mask or nonnegative invariant failed')
                both = row['left_eye_valid'] and row['right_eye_valid']
                mean = row['ear_mean']
                if both != (mean is not None) or both and not math.isclose(mean,
                        (row['ear_left']+row['ear_right'])/2, rel_tol=2e-6, abs_tol=1e-7):
                    raise ValueError('EAR mean requires and agrees with both eyes')
                for field in ('pitch','yaw','roll'):
                    value = row[field]
                    bound = 90 if field == 'yaw' else 180
                    if row['pose_valid'] != (value is not None) or value is not None and not -bound <= value <= bound:
                        raise ValueError('Pose null/mask or estimator range invariant failed')
                rms = row['reprojection_error_norm']
                if rms is not None and rms < 0 or row['pose_valid'] and rms is None:
                    raise ValueError('Pose RMS must be finite nonnegative when present')
                counts.update({key: int(row[key]) for key in FLAG_FIELDS})
                counts['both_eyes_valid'] += int(both)
                count += 1
                if first is None:
                    first = timestamp
                last, last_index = timestamp, index
    if count == 0 or any(metadata.get(key) != value for key, value in
            (('row_count',count),('first_timestamp_ms',first),('last_timestamp_ms',last),('last_frame_index',last_index))):
        raise ValueError('Emitted row/index/timestamp metadata mismatch')
    if metadata['validity_counts'] != {key: counts[key] for key in FLAG_FIELDS}:
        raise ValueError('Stored validity counts mismatch')
    reader, pipeline = metadata['reader_stats'], metadata['pipeline_stats']
    if (reader.get('status') != 'EOF' or reader.get('capture_released') is not True
            or reader.get('emitted_frames') != count or pipeline.get('closed') is not True
            or pipeline.get('processed_frames') != count or pipeline.get('detector',{}).get('closed') is not True):
        raise ValueError('Reader EOF/release or model closure/count mismatch')
    decoded = reader.get('decoded_frames')
    detector = pipeline.get('detector', {})
    if (type(decoded) is not int or decoded < count or last_index >= decoded
            or reader.get('dropped_frames') != decoded-count
            or detector.get('inference_frames') != count
            or detector.get('face_frames') != counts['face_detected']
            or sum(detector.get(key, -count-1) for key in
                   ('face_frames','no_face_frames','invalid_landmark_frames')) != count):
        raise ValueError('Reader/model decoded, dropped, inference or face counters mismatch')
    return dict(row_count=count, validity_counts=dict(counts), first_timestamp_ms=first,
                last_timestamp_ms=last, last_frame_index=last_index)


def _coverage(videos: list[dict]) -> list[dict]:
    groups = {}
    for video in videos:
        for scope, group in (('video',video['video_id']),('subject',video['subject_id']),('class',str(video['label_id']))):
            item = groups.setdefault((scope,group),dict(scope=scope,group=group,
                planned_videos=0,complete_videos=0,failed_videos=0,row_count=0,
                **{key+'_count':0 for key in COUNTS}))
            item['planned_videos'] += 1
            complete = video['status'] == 'complete'
            item['complete_videos' if complete else 'failed_videos'] += 1
            if complete:
                item['row_count'] += video['row_count']
                for key in COUNTS:
                    item[key+'_count'] += video['validity_counts'][key]
    for item in groups.values():
        for key in COUNTS:
            item[key+'_coverage'] = item[key+'_count']/item['row_count'] if item['row_count'] else None
    return [groups[key] for key in sorted(groups)]


def audit_snapshot(config_path: Path, manifest_path: Path, output_dir: Path,
                   snapshot: dict, extraction_report: dict) -> dict:
    """Reject freeze drift globally; retain every member on pair/extraction failures."""
    required = {'schema_version','created_at_utc','original_manifest_path','manifest_file_sha256',
        'working_manifest_sha256','config_file_sha256','signature','source_root','output_dir',
        'working_snapshot_count','subject_count','acquisition_planned_count',
        'missing_acquisition_video_ids','sources','extraction_program_sha256','snapshot_sha256'}
    if set(snapshot) != required or snapshot['schema_version'] != 'working_snapshot_v1':
        raise ValueError('Invalid strict working snapshot format')
    if _hash_json({key:value for key,value in snapshot.items() if key != 'snapshot_sha256'}) != snapshot['snapshot_sha256']:
        raise ValueError('Snapshot identity hash mismatch')
    for path, key in ((config_path,'config_file_sha256'),(manifest_path,'manifest_file_sha256')):
        if digest_file(Path(path))[0] != snapshot[key]:
            raise ValueError(f'Frozen {key} drift')
    original = Path(snapshot['original_manifest_path'])
    if digest_file(original)[0] != snapshot['manifest_file_sha256']:
        raise ValueError('Original source manifest drift')
    for path, expected in snapshot['extraction_program_sha256'].items():
        program = Path(path)
        if not program.is_absolute():
            program = PROJECT_ROOT / program
        if digest_file(program)[0] != expected:
            raise ValueError(f'Extraction program drift: {path}')
    manifest = pd.read_parquet(manifest_path)
    validate_working_manifest(manifest)
    rows = _json_value(manifest.to_dict('records'))
    if (rows != snapshot['sources'] or _hash_json(sorted(rows,key=lambda row:row['video_id'])) != snapshot['working_manifest_sha256']
            or len(rows) != snapshot['working_snapshot_count']
            or len({row['subject_id'] for row in rows}) != snapshot['subject_count']
            or snapshot['acquisition_planned_count'] != 45
            or len(snapshot['missing_acquisition_video_ids']) != 45-len(rows)
            or len(set(snapshot['missing_acquisition_video_ids'])) != len(snapshot['missing_acquisition_video_ids'])
            or set(snapshot['missing_acquisition_video_ids']) & {row['video_id'] for row in rows}):
        raise ValueError('Frozen source identity/support mismatch')
    config = load_config(config_path)
    builder = FeatureDatasetBuilder(config, constant_fps_verified=snapshot['signature']['constant_fps_verified'])
    signature = builder._signature()
    if (signature != snapshot['signature'] or Path(config['dataset_root']).resolve() != Path(snapshot['source_root']).resolve()
            or Path(output_dir).resolve() != Path(snapshot['output_dir']).resolve()):
        raise ValueError('Snapshot config/signature/source/output drift')
    if extraction_report.get('snapshot_sha256') != snapshot['snapshot_sha256']:
        raise ValueError('Extraction report snapshot identity mismatch')
    extracted = extraction_report.get('videos',[])
    if (len(extracted) != len(rows) or {item['video_id'] for item in extracted} != {row['video_id'] for row in rows}):
        raise ValueError('Extraction report membership mismatch')
    by_id = {item['video_id']:item for item in extracted}
    videos = []
    reasons = Counter()
    root = Path(snapshot['source_root']).resolve()
    for row in rows:
        extraction = by_id[row['video_id']]
        item = dict(video_id=row['video_id'],subject_id=row['subject_id'],label_id=row['label_id'],
            source_label=row['source_label'],status='failed', extraction_status=extraction.get('status'),
            extraction_error=extraction.get('error'),
            row_count=0,validity_counts={key:0 for key in COUNTS},
            snapshot_sha256=snapshot['snapshot_sha256'],source_sha256=row['sha256'],
            source_size_bytes=row['size_bytes'],manifest_row_sha256=_hash_json(row),
            extraction_fingerprint=signature['extraction_fingerprint'])
        try:
            source = builder._source(root,row)
            source_sha, source_size = digest_file(source)[0], source.stat().st_size
            if source_sha.lower() != row['sha256'].lower() or source_size != int(row['size_bytes']):
                raise ValueError('Source SHA/size drift')
            if extraction.get('status') not in ('completed','cached') or extraction.get('output_current') is not True:
                raise ValueError('Extraction failed or not complete: '+str(extraction.get('error',extraction.get('status'))))
            parquet, marker = builder._paths(Path(output_dir).resolve(),row['video_id'])
            metadata = builder._cached(parquet,marker,row,source_sha,source_size,signature)
            if metadata is None:
                raise ValueError('Complete pair/schema/fingerprint validation failed')
            result = _audit_pair(parquet,row,metadata)
            if any(extraction.get(key) != metadata[key] for key in ('row_count','extraction_fingerprint','reader_stats','pipeline_stats','validity_counts','quality_reason_counts')):
                raise ValueError('Extraction report differs from committed metadata')
            quality = metadata['quality_reason_counts']
            if any(type(value) is not int or not 0 <= value <= result['row_count'] for value in quality.values()):
                raise ValueError('Quality reason event count out of bounds')
            pipeline = metadata['pipeline_stats']
            if pipeline.get('quality_reasons') != quality or pipeline.get('valid_frames') != metadata['validity_counts']:
                raise ValueError('Pipeline diagnostic counts mismatch')
            item.update(result,status='complete',quality_reason_counts=quality)
            reasons.update(quality)
        except Exception as exc:
            item['error'] = str(exc)
        videos.append(item)
    complete = sum(item['status']=='complete' for item in videos)
    return dict(schema_version='feature_audit_v1',snapshot_sha256=snapshot['snapshot_sha256'],
        working_manifest_sha256=snapshot['working_manifest_sha256'],manifest_file_sha256=snapshot['manifest_file_sha256'],
        config_file_sha256=snapshot['config_file_sha256'],source_root=snapshot['source_root'],
        output_dir=snapshot['output_dir'],
        extraction_fingerprint=signature['extraction_fingerprint'],working_snapshot_count=len(rows),
        acquisition_planned_count=45,subject_count=snapshot['subject_count'],
        missing_acquisition_video_ids=snapshot['missing_acquisition_video_ids'],
        complete_videos=complete,failed_videos=len(rows)-complete,status='complete' if complete==len(rows) else 'failed',
        quality_reason_counts=dict(reasons),quality_reason_semantics='Overlapping categorical event counts, not rejected-frame totals',
        videos=videos,coverage=_coverage(videos))
