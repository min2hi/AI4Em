"""Real Parquet audit invariants; no extraction or native models are invoked."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from src.preprocessing.audit import _audit_pair, _coverage, audit_snapshot
from src.preprocessing.builder import STORAGE_SCHEMA, FLAG_FIELDS, FeatureDatasetBuilder, _hash_json, _json_value
from src.datasets.manifest import build_manifest
from src.datasets.acquisition import digest_file


def pair(tmp_path, rows):
    path = tmp_path / 'features.parquet'
    pq.write_table(pa.Table.from_pylist(rows, schema=STORAGE_SCHEMA), path, row_group_size=1)
    metadata = dict(row_count=len(rows), first_timestamp_ms=rows[0]['timestamp_ms'],
        last_timestamp_ms=rows[-1]['timestamp_ms'], last_frame_index=rows[-1]['frame_index'],
        validity_counts={key:sum(row[key] for row in rows) for key in FLAG_FIELDS},
        reader_stats=dict(status='EOF', capture_released=True, emitted_frames=len(rows),
                          decoded_frames=len(rows), dropped_frames=0),
        pipeline_stats=dict(closed=True, processed_frames=len(rows), detector=dict(closed=True,
            inference_frames=len(rows), face_frames=sum(row['face_detected'] for row in rows),
            no_face_frames=sum(not row['face_detected'] for row in rows), invalid_landmark_frames=0)))
    return path, metadata


def row(index=0, **updates):
    result = dict(dataset_name='uta_rldd', subject_id='51', video_id='51_5', label_id=1,
        label_source='video', frame_index=index, timestamp_ms=index*50,
        **dict.fromkeys(('ear_left','ear_right','ear_mean','mar','pitch','yaw','roll','reprojection_error_norm')),
        **dict.fromkeys(FLAG_FIELDS, False))
    result.update(updates)
    return result


def test_disorder_across_parquet_batches_is_rejected(tmp_path):
    rows = [row(0), row(1), row(2)]
    rows[-1]['timestamp_ms'] = 40
    path, metadata = pair(tmp_path, rows)
    with pytest.raises(ValueError, match='increasing'):
        _audit_pair(path, rows[0], metadata, batch_size=1)


@pytest.mark.parametrize('updates', [dict(ear_left=.2), dict(left_eye_valid=True),
    dict(pose_valid=True), dict(face_detected=False, mouth_valid=True, mar=.2)])
def test_invalid_null_contradictions_are_rejected(tmp_path, updates):
    item = row(**updates)
    path, metadata = pair(tmp_path, [item])
    with pytest.raises(ValueError):
        _audit_pair(path, row(), metadata)


@pytest.mark.parametrize('field,value', [('subject_id','04'),('video_id','04_5'),('label_id',2),('label_source','frame')])
def test_consumer_visible_identity_drift_is_rejected(tmp_path, field, value):
    item = row()
    item[field] = value
    path, metadata = pair(tmp_path, [item])
    with pytest.raises(ValueError, match='identity'):
        _audit_pair(path, row(), metadata)


def test_invalid_pose_can_retain_finite_rms(tmp_path):
    path, metadata = pair(tmp_path, [row(reprojection_error_norm=.12)])
    assert _audit_pair(path, row(), metadata)['row_count'] == 1


def test_coverage_is_frame_weighted_and_retains_failed_members():
    members = [dict(video_id='a',subject_id='51',label_id=1,status='complete',row_count=1,
                    validity_counts={key:1 for key in (*FLAG_FIELDS,'both_eyes_valid')}),
               dict(video_id='b',subject_id='51',label_id=1,status='complete',row_count=3,
                    validity_counts={key:0 for key in (*FLAG_FIELDS,'both_eyes_valid')}),
               dict(video_id='c',subject_id='04',label_id=0,status='failed',row_count=0,
                    validity_counts={key:0 for key in (*FLAG_FIELDS,'both_eyes_valid')})]
    coverage = _coverage(members)
    subject = next(item for item in coverage if item['scope']=='subject' and item['group']=='51')
    assert subject['face_detected_coverage'] == .25
    failed = next(item for item in coverage if item['scope']=='subject' and item['group']=='04')
    assert failed['planned_videos'] == failed['failed_videos'] == 1
    assert failed['complete_videos'] == 0 and failed['face_detected_coverage'] is None


@pytest.fixture
def frozen(tmp_path, feature_backend):
    root = tmp_path/'sources'
    (root/'51').mkdir(parents=True)
    writer = cv2.VideoWriter(str(root/'51/5.avi'),cv2.VideoWriter_fourcc(*'MJPG'),20,(64,48))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((48,64,3),dtype=np.uint8))
    writer.release()
    manifest, errors = build_manifest(root,'uta_rldd')
    assert not errors
    manifest['fold_id'] = pd.array([1],dtype='Int8')
    manifest['fold_source'] = 'official_archive_index'
    manifest_path = tmp_path/'manifest.parquet'
    manifest.to_parquet(manifest_path,index=False)
    config = {**feature_backend.config,'dataset_root':root,'output_dir':tmp_path/'output'}
    config_path = tmp_path/'preprocessing.yaml'
    config_path.write_text(yaml.safe_dump(_json_value(config)),encoding='utf-8')
    signature = FeatureDatasetBuilder(config)._signature()
    rows = _json_value(manifest.to_dict('records'))
    snapshot = dict(schema_version='working_snapshot_v1',created_at_utc='2026-10-07T00:00:00+00:00',
        original_manifest_path=str(manifest_path),manifest_file_sha256=digest_file(manifest_path)[0],
        working_manifest_sha256=_hash_json(sorted(rows,key=lambda item:item['video_id'])),
        config_file_sha256=digest_file(config_path)[0],signature=signature,source_root=str(root),
        output_dir=str(config['output_dir']),working_snapshot_count=1,subject_count=1,
        acquisition_planned_count=45,missing_acquisition_video_ids=[f'missing_{i}' for i in range(44)],
        sources=rows,extraction_program_sha256={str(Path(__file__).resolve()):digest_file(Path(__file__))[0]})
    snapshot['snapshot_sha256'] = _hash_json(snapshot)
    report = dict(snapshot_sha256=snapshot['snapshot_sha256'],videos=[dict(video_id=rows[0]['video_id'],
        status='failed',error='native failure',output_current=False)])
    return config_path,manifest_path,config['output_dir'],snapshot,report


def test_failed_extraction_is_explicit_with_zero_support(frozen):
    report = audit_snapshot(*frozen)
    assert report['working_snapshot_count']==1 and report['acquisition_planned_count']==45
    assert report['complete_videos']==0 and report['failed_videos']==1
    assert report['videos'][0]['status']=='failed'
    assert 'native failure' in report['videos'][0]['error']
    assert all(item['face_detected_coverage'] is None for item in report['coverage'])


def test_snapshot_hash_identity_drift_is_refused(frozen):
    frozen[3]['subject_count']=2
    with pytest.raises(ValueError,match='identity hash'):
        audit_snapshot(*frozen)


def test_source_drift_cannot_pass_as_complete(frozen):
    source = Path(frozen[3]['source_root'])/frozen[3]['sources'][0]['relative_path']
    with source.open('ab') as handle:
        handle.write(b'drift')
    report = audit_snapshot(*frozen)
    assert report['failed_videos']==1
    assert 'SHA/size drift' in report['videos'][0]['error']

def publish_pair(frozen):
    _, _, output, snapshot, report = frozen
    output.mkdir()
    source = snapshot['sources'][0]
    records = [row(i, **{key:source[key] for key in
               ('dataset_name','subject_id','video_id','label_id','label_source')}) for i in range(3)]
    path, metadata = pair(output, records)
    final = output/(source['video_id']+'.parquet')
    path.rename(final)
    metadata.update(snapshot['signature'], manifest_row=source, manifest_row_sha256=_hash_json(source),
        source_sha256=source['sha256'], source_size_bytes=source['size_bytes'],
        complete_source_validation=True, parquet_sha256=digest_file(final)[0],camera_metadata={},
        quality_reason_counts={})
    metadata['pipeline_stats'].update(valid_frames=metadata['validity_counts'],quality_reasons={})
    (output/(source['video_id']+'.metadata.json')).write_text(json.dumps(metadata),encoding='utf-8')
    report['videos'] = [dict(video_id=source['video_id'],status='completed',output_current=True,
        **{key:metadata[key] for key in ('row_count','extraction_fingerprint','reader_stats',
            'pipeline_stats','validity_counts','quality_reason_counts')})]
    return final


def test_complete_pair_is_independently_reopened(frozen):
    publish_pair(frozen)
    result = audit_snapshot(*frozen)
    assert result['status']=='complete' and result['complete_videos']==1
    assert result['videos'][0]['row_count']==3
    assert result['coverage'][0]['face_detected_coverage']==0


def test_failed_report_cannot_be_relabelled_by_present_valid_pair(frozen):
    publish_pair(frozen)
    frozen[4]['videos'][0].update(status='failed',output_current=False,error='failed retry')
    result = audit_snapshot(*frozen)
    assert result['failed_videos']==1 and result['videos'][0]['row_count']==0

@pytest.mark.parametrize('change', ['reader_release','model_close','row_count','validity_count','last_index','schema'])
def test_committed_diagnostics_and_schema_are_checked(tmp_path, change):
    path, metadata = pair(tmp_path,[row()])
    if change == 'reader_release':
        metadata['reader_stats']['capture_released'] = False
    elif change == 'model_close':
        metadata['pipeline_stats']['detector']['closed'] = False
    elif change == 'row_count':
        metadata['row_count'] = 2
    elif change == 'validity_count':
        metadata['validity_counts']['face_detected'] = 1
    elif change == 'last_index':
        metadata['last_frame_index'] = 1
    else:
        pq.write_table(pa.table({'frame_index':[0]}),path)
    with pytest.raises(ValueError):
        _audit_pair(path,row(),metadata)


def test_ratios_are_not_clamped_or_arbitrarily_bounded(tmp_path):
    item = row(face_detected=True,left_eye_valid=True,right_eye_valid=True,mouth_valid=True,
               ear_left=12.,ear_right=14.,ear_mean=13.,mar=100.)
    path, metadata = pair(tmp_path,[item])
    assert _audit_pair(path,row(),metadata)['validity_counts']['both_eyes_valid']==1


@pytest.mark.parametrize('yaw', [-120., 120.])
def test_valid_pose_yaw_outside_estimator_range_is_rejected(tmp_path, yaw):
    item = row(face_detected=True,pose_valid=True,pitch=0.,yaw=yaw,roll=0.,
               reprojection_error_norm=.01)
    path, metadata = pair(tmp_path,[item])
    with pytest.raises(ValueError,match='estimator range'):
        _audit_pair(path,row(),metadata)


@pytest.mark.parametrize('yaw', [-90., 90.])
def test_float32_yaw_endpoints_are_accepted(tmp_path, yaw):
    item = row(face_detected=True,pose_valid=True,pitch=-180.,yaw=yaw,roll=180.,
               reprojection_error_norm=.01)
    path, metadata = pair(tmp_path,[item])
    assert _audit_pair(path,row(),metadata)['validity_counts']['pose_valid']==1


def test_missing_emitted_prefix_is_rejected_with_consistent_counters(tmp_path):
    path, metadata = pair(tmp_path,[row(5)])
    metadata['reader_stats'].update(decoded_frames=6,dropped_frames=5)
    with pytest.raises(ValueError,match='first emitted frame'):
        _audit_pair(path,row(),metadata)


def test_legitimate_subsampling_after_frame_zero_is_accepted(tmp_path):
    path, metadata = pair(tmp_path,[row(0),row(2)])
    metadata['reader_stats'].update(decoded_frames=3,dropped_frames=1)
    assert _audit_pair(path,row(),metadata,batch_size=1)['row_count']==2


def test_relative_program_hashes_use_project_root_after_chdir(frozen, tmp_path, monkeypatch):
    import src.preprocessing.audit as audit

    publish_pair(frozen)
    project = tmp_path/'project'
    program = project/'scripts/preprocess.py'
    program.parent.mkdir(parents=True)
    program.write_bytes(b'print(\"deterministic extraction program\")\n')
    monkeypatch.setattr(audit,'PROJECT_ROOT',project)
    snapshot = frozen[3]
    snapshot['extraction_program_sha256'] = {'scripts/preprocess.py':digest_file(program)[0]}
    snapshot['snapshot_sha256'] = _hash_json({key:value for key,value in snapshot.items()
                                           if key != 'snapshot_sha256'})
    frozen[4]['snapshot_sha256'] = snapshot['snapshot_sha256']
    elsewhere = tmp_path/'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert audit_snapshot(*frozen)['status']=='complete'
    program.write_bytes(b'print(\"changed extraction program\")\n')
    with pytest.raises(ValueError,match='Extraction program drift'):
        audit_snapshot(*frozen)
