import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from mediapipe.tasks.python import vision

from src.datasets.manifest import build_manifest
from src.preprocessing.builder import FeatureDatasetBuilder, storage_row

def test_missing_measurements_are_null_without_losing_identity():
    sample = SimpleNamespace(frame_index=0,timestamp_ms=0,face_detected=False,
        left_eye_valid=False,right_eye_valid=False,mouth_valid=False,pose_valid=False,
        **dict.fromkeys(('ear_left','ear_right','ear_mean','mar','pitch','yaw','roll','reprojection_error_norm'),math.nan))
    row = storage_row(sample,dict(dataset_name='uta_rldd',subject_id='04',video_id='04_5',label_id=1,label_source='video'))
    assert row['ear_mean'] is None and row['reprojection_error_norm'] is None
    assert row['timestamp_ms']==0 and row['frame_index']==0
    assert row['subject_id']=='04' and row['label_id']==1
    assert not row['face_detected'] and not row['pose_valid']


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    root = tmp_path / "sources"
    (root / "04").mkdir(parents=True)
    for label in (0, 5, 10):
        writer = cv2.VideoWriter(str(root / "04" / f"{label}.avi"),
                                 cv2.VideoWriter_fourcc(*"MJPG"), 15, (64, 48))
        assert writer.isOpened()
        for index in range(15):
            image = np.random.default_rng(index).integers(80, 161, (48, 64, 3), dtype=np.uint8)
            writer.write(image)
        writer.release()
    manifest, errors = build_manifest(root, "uta_rldd")
    assert not errors
    manifest["fold_id"] = pd.array([1, 1, 1], dtype="Int8")
    manifest["fold_source"] = "official_archive_index"
    canonical = tmp_path / "canonical.obj"
    vertices = np.zeros((468, 3))
    vertices[[1, 152, 33, 263, 61, 291]] = [
        (0, -1.126865, 7.475604), (0, -9.403378, 4.264492),
        (-4.445859, 2.663991, 3.173422), (4.445859, 2.663991, 3.173422),
        (-2.456206, -4.342621, 4.283884), (2.456206, -4.342621, 4.283884)]
    canonical.write_text("\n".join(f"v {x} {y} {z}" for x, y, z in vertices), encoding="utf-8")
    quality = dict(policy_version="raw_quality_v1", reference_max_side=256,
        min_blur_variance=10., brightness_min=40., brightness_max=200.,
        min_eye_width_ratio=.01, max_abs_yaw_deg=35., max_abs_pitch_deg=25., eyes_occluded=False)
    frozen = tmp_path / "quality.json"
    frozen.write_text(json.dumps({"quality": quality}), encoding="utf-8")
    quality.update(frozen_report_path=frozen,
        frozen_report_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest())
    asset = tmp_path / "face.task"
    asset.write_bytes(b"native loader boundary only")
    state = SimpleNamespace(created=0, closed=0, calls=0, fail_at=None, callback=None)
    state.points = None
    class Native:
        def detect_for_video(self, image, timestamp):
            state.calls += 1
            if state.callback:
                state.callback()
            if state.fail_at == state.calls:
                raise RuntimeError("native failure after earlier decoded rows")
            faces = [] if state.points is None else [[SimpleNamespace(x=float(x), y=float(y), z=float(z))
                                                      for x, y, z in state.points]]
            return vision.FaceLandmarkerResult(face_landmarks=faces, face_blendshapes=[],
                                               facial_transformation_matrixes=[])
        def close(self):
            state.closed += 1
    def create(options):
        state.created += 1
        return Native()
    monkeypatch.setattr(vision.FaceLandmarker, "create_from_options", create)
    config = dict(schema_version="facial_features_v1", dataset_root=root, asset_path=asset,
        canonical_model_path=canonical, epsilon=1e-6, max_reprojection_error_norm=.03,
        landmarker_mode="VIDEO", num_faces=1, min_face_detection_confidence=.5,
        min_face_presence_confidence=.5, min_tracking_confidence=.5,
        landmark_target_fps=20, quality=quality, camera_model={"mode": "approximate"})
    return SimpleNamespace(root=root, manifest=manifest, config=config, state=state,
                           output=tmp_path / "output")


def build_one(fixture, **kwargs):
    return FeatureDatasetBuilder(fixture.config, **kwargs).build(
        fixture.manifest, fixture.output, video_ids=("04_5",))


def test_real_video_preserves_all_missing_rows_and_verified_pair_resume(source_fixture):
    fixture = source_fixture
    report = build_one(fixture)
    assert report["videos"][0]["status"] == "completed"
    table = pq.read_table(fixture.output / "04_5.parquet")
    assert table["timestamp_ms"].to_pylist() == [0,67,133,200,267,333,400,467,533,600,667,733,800,867,933]
    assert table["frame_index"].to_pylist() == list(range(15))
    assert table["subject_id"].to_pylist() == ["04"] * 15
    assert table["label_id"].to_pylist() == [1] * 15
    for name in ("ear_left","ear_right","ear_mean","mar","pitch","yaw","roll","reprojection_error_norm"):
        assert table.schema.field(name).type == pa.float32()
        assert table[name].null_count == 15
    for name in ("face_detected","left_eye_valid","right_eye_valid","mouth_valid","pose_valid"):
        assert table.schema.field(name).type == pa.bool_()
        assert table[name].to_pylist() == [False] * 15
    assert table.schema.field("timestamp_ms").type == pa.int64()
    assert table.schema.field("frame_index").type == pa.int32()
    assert table.schema.field("label_id").type == pa.int8()
    metadata = json.loads((fixture.output / "04_5.metadata.json").read_text())
    assert metadata["reader_stats"]["status"] == "EOF"
    assert metadata["reader_stats"]["capture_released"]
    assert metadata["row_count"] == 15 and metadata["last_timestamp_ms"] == 933
    assert metadata["complete_source_validation"] is True
    assert metadata["parquet_sha256"] == hashlib.sha256((fixture.output / "04_5.parquet").read_bytes()).hexdigest()
    assert build_one(fixture)["videos"][0]["status"] == "cached"
    assert fixture.state.created == fixture.state.closed == 1


@pytest.mark.parametrize("change", ["config", "camera", "asset", "canonical", "quality", "producer", "dependency", "cfr"])
def test_changed_extraction_inputs_never_reuse_stale_output(source_fixture, monkeypatch, change):
    fixture = source_fixture
    assert build_one(fixture)["videos"][0]["status"] == "completed"
    kwargs = {}
    if change == "config":
        fixture.config["landmark_target_fps"] = 15
    elif change == "camera":
        fixture.config["camera_model"] = dict(mode="calibrated", reference_size=[64,48],
            matrix=[[64,0,32],[0,64,24],[0,0,1]], distortion=[0,0,0,0,0])
    elif change in ("asset", "canonical"):
        key = "asset_path" if change == "asset" else "canonical_model_path"
        with fixture.config[key].open("ab") as handle:
            handle.write(b"\n# changed artifact\n")
    elif change == "quality":
        quality = fixture.config["quality"]
        quality["min_blur_variance"] = 11.
        policy = {k:v for k,v in quality.items() if not k.startswith("frozen_report_")}
        quality["frozen_report_path"].write_text(json.dumps({"quality": policy}))
        quality["frozen_report_sha256"] = hashlib.sha256(quality["frozen_report_path"].read_bytes()).hexdigest()
    elif change == "producer":
        monkeypatch.setattr("src.preprocessing.builder.PRODUCER_VERSION", "test-new-producer")
    elif change == "dependency":
        import src.preprocessing.builder as module
        real_version = module.version
        monkeypatch.setattr(module, "version", lambda name: real_version(name) + ".changed")
    else:
        kwargs["constant_fps_verified"] = True
    assert build_one(fixture, **kwargs)["videos"][0]["status"] == "completed"
    assert fixture.state.created == 2


@pytest.mark.parametrize("corruption", ["metadata_missing", "parquet", "row_count", "eof", "schema"])
def test_resume_rejects_uncommitted_or_corrupt_pairs(source_fixture, corruption):
    fixture = source_fixture
    build_one(fixture)
    metadata_path = fixture.output / "04_5.metadata.json"
    parquet_path = fixture.output / "04_5.parquet"
    metadata = json.loads(metadata_path.read_text())
    if corruption == "metadata_missing":
        metadata_path.unlink()
    elif corruption == "parquet":
        parquet_path.write_bytes(b"not parquet")
    else:
        if corruption == "row_count":
            metadata["row_count"] = 14
        elif corruption == "eof":
            metadata["reader_stats"]["status"] = "STOPPED"
        else:
            pq.write_table(pa.table({"timestamp_ms": [0]}), parquet_path)
            metadata["parquet_sha256"] = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
        metadata_path.write_text(json.dumps(metadata))
    assert build_one(fixture)["videos"][0]["status"] == "completed"


def test_source_hash_is_checked_before_model_load_and_again_before_commit(source_fixture):
    fixture = source_fixture
    source = fixture.root / "04/5.avi"
    original = source.read_bytes()
    source.write_bytes(original + b"mutation")
    assert build_one(fixture)["videos"][0]["status"] == "failed"
    assert fixture.state.created == 0
    source.write_bytes(original)
    fixture.state.callback = lambda: source.write_bytes(original + b"mutation")
    assert build_one(fixture)["videos"][0]["status"] == "failed"
    assert not (fixture.output / "04_5.metadata.json").exists()
    assert fixture.state.closed == 1


def test_native_failure_does_not_publish_prefix_and_other_sources_continue(source_fixture):
    fixture = source_fixture
    fixture.state.fail_at = 2
    report = FeatureDatasetBuilder(fixture.config).build(
        fixture.manifest, fixture.output, video_ids=("04_5", "04_10"))
    assert [row["status"] for row in report["videos"]] == ["failed", "completed"]
    assert not (fixture.output / "04_5.metadata.json").exists()
    assert not (fixture.output / "04_5.parquet").exists()
    assert fixture.state.closed == 2
    assert not list(fixture.output.glob("*.tmp"))


@pytest.mark.parametrize("failure", ["write", "rename_parquet", "rename_metadata"])
def test_publication_failure_never_leaves_a_cacheable_pair(source_fixture, monkeypatch, failure):
    fixture = source_fixture
    import src.preprocessing.builder as module
    if failure == "write":
        monkeypatch.setattr(module.pq.ParquetWriter, "write_table",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("write failure")))
    else:
        real_replace = module.os.replace
        def replace(source, target):
            if (failure == "rename_metadata") == str(target).endswith(".metadata.json"):
                raise OSError("rename failure")
            return real_replace(source, target)
        monkeypatch.setattr(module.os, "replace", replace)
    assert build_one(fixture)["videos"][0]["status"] == "failed"
    assert fixture.state.closed == 1
    assert not list(fixture.output.glob("*.tmp"))
    monkeypatch.undo()
    # Restore only the external native loader boundary after undoing fault injection.
    monkeypatch.setattr(vision.FaceLandmarker, "create_from_options",
        lambda options: SimpleNamespace(detect_for_video=lambda image, timestamp:
            vision.FaceLandmarkerResult(face_landmarks=[], face_blendshapes=[],
                                        facial_transformation_matrixes=[]), close=lambda: None))
    assert build_one(fixture)["videos"][0]["status"] == "completed"


@pytest.mark.parametrize("selection", [("missing",), ("04_5", "04_5"), ()])
def test_bad_selection_fails_before_model_loading(source_fixture, selection):
    fixture = source_fixture
    with pytest.raises(ValueError):
        FeatureDatasetBuilder(fixture.config).build(fixture.manifest, fixture.output, video_ids=selection)
    assert fixture.state.created == 0


def test_partial_subject_51_is_a_real_storage_consumer(source_fixture):
    fixture = source_fixture
    (fixture.root / "51").mkdir()
    (fixture.root / "51/5.avi").write_bytes((fixture.root / "04/5.avi").read_bytes())
    manifest, errors = build_manifest(fixture.root, "uta_rldd")
    assert not errors
    manifest["fold_id"] = pd.array([1 if subject == "04" else 5 for subject in manifest["subject_id"]], dtype="Int8")
    manifest["fold_source"] = "official_archive_index"
    report = FeatureDatasetBuilder(fixture.config).build(manifest, fixture.output, video_ids=("51_5",))
    assert report["selected_count"] == 1 and report["working_snapshot_count"] == 4
    assert report["acquisition_planned_count"] == 45
    table = pq.read_table(fixture.output / "51_5.parquet")
    assert table["subject_id"].to_pylist() == ["51"] * 15
    assert table["label_id"].to_pylist() == [1] * 15


def test_real_truncated_avi_cannot_commit_decoded_prefix(source_fixture):
    fixture = source_fixture
    path = fixture.root / "04/5.avi"
    data = path.read_bytes()
    position = data.index(b"00dc", data.index(b"movi") + 4)
    # Retain two complete MJPG chunks, but the real AVI header still declares15.
    for _ in range(2):
        size = int.from_bytes(data[position + 4:position + 8], "little")
        position += 8 + size + size % 2
    path.write_bytes(data[:position])
    fixture.manifest.loc[fixture.manifest["video_id"] == "04_5", "sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    fixture.manifest.loc[fixture.manifest["video_id"] == "04_5", "size_bytes"] = path.stat().st_size
    report = build_one(fixture)
    assert report["videos"][0]["status"] == "failed"
    assert report["videos"][0]["reader_stats"]["status"] == "ERROR"
    assert report["videos"][0]["row_count"] == 2
    assert not (fixture.output / "04_5.metadata.json").exists()
    assert fixture.state.closed == 1


@pytest.mark.parametrize("escape", ["source", "destination"])
def test_resolved_symlink_escape_is_rejected_before_native_load(source_fixture, tmp_path, escape):
    fixture = source_fixture
    outside = tmp_path / "outside"
    outside.write_bytes((fixture.root / "04/5.avi").read_bytes())
    target = fixture.root / "04/5.avi" if escape == "source" else fixture.output / "04_5.parquet"
    if escape == "source":
        target.unlink()
    else:
        fixture.output.mkdir()
    try:
        target.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Host cannot create symlinks: {exc}")
    assert build_one(fixture)["videos"][0]["status"] == "failed"
    assert fixture.state.created == 0
    assert outside.read_bytes()


def test_retained_error_rows_are_reported_without_native_load(source_fixture):
    fixture = source_fixture
    row = fixture.manifest["video_id"] == "04_5"
    fixture.manifest.loc[row, ["status", "error", "sampled_decode_ok"]] = ["error", "decode unavailable", False]
    assert build_one(fixture)["videos"][0]["status"] == "failed"
    assert fixture.state.created == 0


def test_failed_recompute_preserves_old_bytes_but_reports_stale(source_fixture):
    fixture = source_fixture
    build_one(fixture)
    old_parquet = (fixture.output / "04_5.parquet").read_bytes()
    old_marker = (fixture.output / "04_5.metadata.json").read_bytes()
    fixture.config["landmark_target_fps"] = 15
    fixture.state.fail_at = fixture.state.calls + 2
    item = build_one(fixture)["videos"][0]
    assert item["status"] == "failed" and item["stale_output"] and not item["output_current"]
    assert (fixture.output / "04_5.parquet").read_bytes() == old_parquet
    assert (fixture.output / "04_5.metadata.json").read_bytes() == old_marker


def test_builder_cli_explicit_three_class_selection_and_resume(source_fixture, tmp_path):
    from scripts.preprocess import main
    fixture = source_fixture
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(json.loads(json.dumps(fixture.config, default=str))))
    manifest_path = tmp_path / "manifest.parquet"
    fixture.manifest.to_parquet(manifest_path, index=False)
    report_dir = tmp_path / "reports"
    args = ["--config", str(config_path), "--manifest", str(manifest_path),
            "--output-dir", str(fixture.output), "--report-dir", str(report_dir)]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    args += ["--video-id", "04_0", "--video-id", "04_5", "--video-id", "04_10"]
    assert main(args) == 0
    for label, expected in ((0, 0), (5, 1), (10, 2)):
        table = pq.read_table(fixture.output / f"04_{label}.parquet")
        assert table["label_id"].to_pylist() == [expected] * 15
    assert main(args) == 0
    report = json.loads((report_dir / "report.json").read_text())
    assert report["totals"] == {"completed": 0, "cached": 3, "failed": 0}
    fixture.state.fail_at = fixture.state.calls + 2
    fixture.config["landmark_target_fps"] = 15
    config_path.write_text(yaml.safe_dump(json.loads(json.dumps(fixture.config, default=str))))
    assert main(args) == 1
    assert json.loads((report_dir / "report.json").read_text())["status"] == "failed"


def test_builder_schema_flushes_bounded_batches_without_dropping_rows(source_fixture):
    fixture = source_fixture
    path = fixture.root / "04/5.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (64, 48))
    assert writer.isOpened()
    for _ in range(1100):
        writer.write(np.full((48, 64, 3), 120, dtype=np.uint8))
    writer.release()
    fixture.manifest, errors = build_manifest(fixture.root, "uta_rldd")
    assert not errors
    fixture.manifest["fold_id"] = pd.array([1, 1, 1], dtype="Int8")
    fixture.manifest["fold_source"] = "official_archive_index"
    assert build_one(fixture)["videos"][0]["status"] == "completed"
    with pq.ParquetFile(fixture.output / "04_5.parquet") as stored:
        assert stored.metadata.num_rows == 1100
        assert [stored.metadata.row_group(i).num_rows for i in range(stored.metadata.num_row_groups)] == [1024, 76]


def test_independent_geometry_survives_real_pipeline_and_storage(source_fixture):
    fixture = source_fixture
    ids = (1, 152, 33, 263, 61, 291)
    literal = np.array([(0,-1.126865,7.475604),(0,-9.403378,4.264492),
        (-4.445859,2.663991,3.173422),(4.445859,2.663991,3.173422),
        (-2.456206,-4.342621,4.283884),(2.456206,-4.342621,4.283884)])
    xy, _ = cv2.projectPoints(literal * (1,-1,-1), np.zeros(3),
        np.array([0.,0.,60.]), np.array([[64.,0.,32.],[0.,64.,24.],[0.,0.,1.]]), np.zeros(5))
    points = np.zeros((478,3), dtype=np.float32)
    points[list(ids), :2] = xy.reshape(6,2) / (64,48)
    for index, xy in ((234,(.25,.5)),(454,(.75,.5)),(10,(.5,.25)),
        (78,(.46,.56)),(308,(.54,.56)),(82,(.48,.55)),(87,(.48,.57)),
        (13,(.5,.55)),(14,(.5,.57)),(312,(.52,.55)),(317,(.52,.57))):
        points[index, :2] = xy
    for outer, other, uppers, lowers, direction in (
        (33,133,(160,158),(144,153),1), (263,362,(385,387),(380,373),-1)):
        base = points[outer, :2].copy()
        points[other, :2] = base + (direction*.04,0)
        for fraction, upper, lower in zip((1,2), uppers, lowers):
            points[upper, :2] = base + (direction*.04*fraction/3,-.01)
            points[lower, :2] = base + (direction*.04*fraction/3,.01)
    fixture.state.points = points
    assert build_one(fixture)["videos"][0]["status"] == "completed"
    table = pq.read_table(fixture.output / "04_5.parquet")
    assert table["ear_mean"].to_pylist() == pytest.approx([.375] * 15, abs=1e-5)
    assert table["mar"].to_pylist() == pytest.approx([.1875] * 15, abs=1e-5)
    for name in ("pitch","yaw","roll"):
        assert table[name].to_pylist() == pytest.approx([0.] * 15, abs=.01)
    for name in ("face_detected","left_eye_valid","right_eye_valid","mouth_valid","pose_valid"):
        assert table[name].to_pylist() == [True] * 15
    metadata = json.loads((fixture.output / "04_5.metadata.json").read_text())
    assert metadata["camera_metadata"]["approximate"] is True


def test_interruption_between_replaces_rejects_old_marker_on_resume(source_fixture, monkeypatch):
    import src.preprocessing.builder as module
    fixture = source_fixture
    build_one(fixture)
    marker = fixture.output / "04_5.metadata.json"
    old_marker = marker.read_bytes()
    fixture.config["landmark_target_fps"] = 15
    real_replace = module.os.replace
    def replace(source, target):
        if str(target).endswith(".metadata.json"):
            raise OSError("interrupted after Parquet replacement")
        return real_replace(source, target)
    monkeypatch.setattr(module.os, "replace", replace)
    item = build_one(fixture)["videos"][0]
    assert item["status"] == "failed" and not item["output_current"]
    assert marker.read_bytes() == old_marker
    monkeypatch.setattr(module.os, "replace", real_replace)
    assert build_one(fixture)["videos"][0]["status"] == "completed"
    assert build_one(fixture)["videos"][0]["status"] == "cached"


def test_manifest_resolution_mismatch_cannot_publish(source_fixture):
    fixture = source_fixture
    fixture.manifest.loc[fixture.manifest["video_id"] == "04_5", "width"] = 63
    item = build_one(fixture)["videos"][0]
    assert item["status"] == "failed"
    assert item["reader_stats"]["status"] == "STOPPED"
    assert item["reader_stats"]["capture_released"]
    assert fixture.state.closed == 1
    assert not (fixture.output / "04_5.metadata.json").exists()


@pytest.mark.parametrize("field", ["max_abs_yaw_deg", "max_abs_pitch_deg", "min_blur_variance"])
def test_equivalent_numeric_policy_literals_accept_frozen_report(source_fixture, field):
    fixture = source_fixture
    fixture.config["quality"][field] = int(fixture.config["quality"][field])
    report = build_one(fixture)
    assert report["videos"][0]["status"] == "completed", report["videos"][0].get("error")
