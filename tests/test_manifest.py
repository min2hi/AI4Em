import json
import subprocess
import sys
import cv2
import numpy as np
import pandas as pd
import pytest

from src.datasets.manifest import build_manifest, validate_manifest


@pytest.fixture
def video_dataset(tmp_path):
    subject = tmp_path / "01"
    subject.mkdir()
    for label, color in [(0,40),(5,100),(10,180)]:
        path = subject / f"{label}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64,48))
        assert writer.isOpened()
        for _ in range(20):
            writer.write(np.full((48,64,3), color, dtype=np.uint8))
        writer.release()
    return tmp_path


def test_manifest_maps_labels_and_probes_real_video(video_dataset):
    frame, errors = build_manifest(video_dataset, "uta_rldd")
    assert errors == []
    assert frame.sort_values("source_label")["label_id"].tolist() == [0,1,2]
    assert frame["subject_id"].tolist() == ["01"] * 3
    assert frame["frame_count"].tolist() == [20] * 3
    assert frame["duration_s"].tolist() == pytest.approx([2.0] * 3)
    assert frame["fold_id"].isna().all()
    assert frame["sampled_decode_ok"].all()
    validate_manifest(frame, expected_subjects=1, expected_videos=3)
    out = video_dataset / "manifest.parquet"
    frame.to_parquet(out, index=False)
    reloaded = pd.read_parquet(out)
    assert reloaded["subject_id"].tolist() == ["01"] * 3


def test_corrupt_video_remains_visible_as_an_error(video_dataset):
    (video_dataset / "01" / "5.avi").write_bytes(b"not a video")
    frame, errors = build_manifest(video_dataset, "uta_rldd")
    assert len(frame) == 3
    assert frame.loc[frame["source_label"]==5,"status"].tolist() == ["error"]
    assert [e["relative_path"] for e in errors] == ["01/5.avi"]
    with pytest.raises(ValueError):
        validate_manifest(frame)


def test_duplicate_video_or_missing_class_is_rejected(video_dataset):
    frame, _ = build_manifest(video_dataset, "uta_rldd")
    with pytest.raises(ValueError):
        validate_manifest(pd.concat([frame,frame.iloc[:1]], ignore_index=True))
    with pytest.raises(ValueError):
        validate_manifest(frame.iloc[:2])


def test_unknown_label_is_not_guessed(video_dataset):
    (video_dataset / "01" / "other.mp4").write_bytes(b"invalid")
    frame, errors = build_manifest(video_dataset, "uta_rldd")
    assert any(e["relative_path"] == "01/other.mp4" for e in errors)
    assert len(frame) == 4
    with pytest.raises(ValueError):
        validate_manifest(frame)


@pytest.mark.parametrize("mismatch", ["subjects", "fold"])
def test_matching_counts_do_not_allow_a_different_acquisition_plan(video_dataset, mismatch):
    frame, _ = build_manifest(video_dataset, "uta_rldd")
    frame["fold_id"] = pd.array([1] * len(frame), dtype="Int8")
    keys = ["relative_path", "subject_id", "video_id", "source_label", "label_id", "fold_id", "size_bytes"]
    plan = frame[keys].to_dict("records")
    if mismatch == "subjects":
        for row in plan:
            row["subject_id"] = "02"
            row["relative_path"] = row["relative_path"].replace("01/", "02/")
            row["video_id"] = row["video_id"].replace("01_", "02_")
    else:
        for row in plan:
            row["fold_id"] = 2
    with pytest.raises(ValueError):
        validate_manifest(frame, expected_subjects=1, expected_videos=3, expected_records=plan)


def test_exploration_rejects_verified_receipt_from_different_source(video_dataset):
    from src.config import PROJECT_ROOT
    frame, _ = build_manifest(video_dataset, "uta_rldd")
    frame["fold_id"] = pd.array([1] * len(frame), dtype="Int8")
    keys = ["relative_path", "subject_id", "video_id", "source_label", "label_id", "fold_id", "size_bytes"]
    records = frame[keys].to_dict("records")
    for record in records:
        record.update(source_archive="Fold1_part1.zip", source_file_id="fixture")
    receipts = [{**record, "status":"verified", "sha256":frame.iloc[i]["sha256"]} for i,record in enumerate(records)]
    receipts[0]["source_file_id"] = "different-source"
    plan = video_dataset / "plan.json"
    downloaded = video_dataset / "downloaded.json"
    config = video_dataset / "config.yaml"
    report_dir = video_dataset / "reports"
    plan.write_text(json.dumps({"records":records,"subject_ids":["01"],"video_count":3,"strategy":"isolated fixture"}))
    downloaded.write_text(json.dumps({"receipts":receipts,"errors":[]}))
    config.write_text(json.dumps({"dataset_root":str(video_dataset),"manifest_path":str(video_dataset/"manifest.parquet")}))
    result = subprocess.run([sys.executable,"-m","scripts.explore_dataset","--config",str(config),
                             "--plan",str(plan),"--download-report",str(downloaded),"--report-dir",str(report_dir)],
                            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30)
    report = json.loads((report_dir / "exploration_report.json").read_text())
    assert result.returncode == 1
    assert report["errors"] == []
