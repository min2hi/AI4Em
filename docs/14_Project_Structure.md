# 14 — Cấu trúc project và config

Cây dưới mô tả đích cuối cùng. Phase 0–3 đã có config/contracts, acquisition, manifest, `preprocessing/video_reader.py`, `features/landmarks.py` và scripts tương ứng. Eye/mouth/pose/pipeline, training/realtime/UI còn lại là thiết kế, chưa được tạo; xem [báo cáo Phase 3](Phase3_Report.md).

```text
project/
  README.md                # Lệnh chạy và trạng thái implementation
  docs/                    # README thiết kế, 01–20, Dataset_Access.md, CHANGELOG.md
  data/raw/                 # Không commit
  data/processed/           # Parquet per video, không commit
  data/acquisition/        # Inventory, subset plan, permission map, download report
  data/splits/              # subject IDs, protocol, hash
  notebooks/               # Exploration, không chứa logic production
  src/
    contracts.py           # Shared data records + enums
    config.py              # Safe YAML, validation, paths project-relative
    preprocessing/video_reader.py, builder.py
    features/landmarks.py, eye.py, mouth.py, head_pose.py, pipeline.py, temporal.py
    datasets/acquisition.py, manifest.py, splits.py, normalization.py, sequence.py, summaries.py
    models/baseline.py, lstm.py, bundle.py
    training/trainer.py
    evaluation/evaluator.py, replay.py
    calibration/profile.py, manager.py
    realtime/buffer.py, smoother.py, detector.py, camera_worker.py
    alerts/manager.py
    ui/main_window.py
  scripts/
    check_environment.py, acquire_manifest.py, explore_dataset.py
    preview_landmarks.py, preprocess.py, build_splits.py
    train_baseline.py, train_lstm.py, evaluate.py, run_ablation.py
    webcam_demo.py, benchmark.py
  tests/                   # Synthetic deterministic fixtures
  models/assets/           # .task, canonical OBJ + URL/hash/license
  models/checkpoints/      # Không commit weights lớn
  configs/preprocessing.yaml, training.yaml, realtime.yaml
  runs/                    # Config/metrics/log từng experiment
  requirements.txt, requirements-dev.txt, requirements-lock.txt
  .gitignore, main.py      # main.py thuộc phase UI sau, chưa có
```

## Storage schema
Parquet per video, dtype `float32` cho feature, bool cho validity, int64 timestamp, int32 frame_index, int8 label_id. NaN/null là missing; timestamp luôn có. Dùng PyArrow engine; không CSV làm source of truth vì dtype/null và kích thước. CSV chỉ export vài hàng cho team xem.

Raw row: `dataset_name, subject_id, video_id, frame_index, timestamp_ms, ear_left, ear_right, ear_mean, mar, pitch, yaw, roll, face_detected, left_eye_valid, right_eye_valid, mouth_valid, pose_valid, reprojection_error_norm, label_id, label_source`.

Metadata companion `video_id.metadata.json`: schema version, source hash, asset/canonical hash, extraction config, actual sampling stats, failures. Với YawDD chưa annotate, label_id null; không điền lớp Drowsy chỉ vì video có ngáp. Derived Parquet versioned giữ normalized feature, event state và ordered feature names; sequence index trỏ vào mảng derived per video.

## Config — một nơi cho mỗi giá trị
YAML đọc bằng `safe_load`; validate ngay startup: FPS dương, window*FPS nguyên, stride≤window, threshold range đúng, paths tồn tại; schema/checkpoint mismatch báo lỗi. Runtime paths resolve từ project root hoặc config root thống nhất, không phụ thuộc shell cwd.

Ví dụ đoạn config; bản chạy hiện tại nằm trong `configs/`. Các gate blur/brightness và split path chỉ thêm ở phase cần dùng, không có placeholder giả chạy:
```yaml
# preprocessing.yaml
landmark_target_fps: 20
sequence_fps: 10
num_faces: 1
landmarker_mode: VIDEO
min_face_detection_confidence: 0.5
min_face_presence_confidence: 0.5
min_tracking_confidence: 0.5
asset_path: models/assets/face_landmarker.task
canonical_model_path: models/assets/canonical_face_model.obj
max_sample_age_ms: 100
statistics_window_s: 60
```

Parameter còn lại phải tập trung:
- **preprocessing:** raw/output/manifest paths; camera K/distortion; blur/brightness/pose gates; landmark IDs; EAR epsilon; blink enter/exit/duration; MAR/yawn gates; perclos proxy threshold=0.20, minimum history=30, coverage=0.80. Quality gate ban đầu đo trên clip rồi freeze, không chọn con số blur universal chưa có sample.
- **training:** splits path/outer fold; calibration_mode P0/P1; window=10; stride=1; missing_ratio=0.20; max_gap_s=1; ordered features; seed=42; RF/LSTM parameters; device; batch/epochs/loss/optimizer/early stopping; experiment path.
- **realtime:** camera index/resolution=640×480; model path; checkpoint mode; prediction interval=1; stale_ms=500; smoothing samples=3; calibration_seconds=30, min_valid_seconds=20, timeout_seconds=60; warning enter/exit thresholds và durations ở [12](12_Realtime_Inference.md); audio path/cooldown=15; mute; UI refresh=10 Hz; log retention.

Không duplicate sequence FPS/window trong realtime YAML: realtime đọc từ model bundle; chỉ override khi tương thích và được kiểm tra, không tự đổi 100 bước thành 50. Resolved config snapshot lưu mỗi run.

## Git và logging
`main` luôn chạy được; feature branch ngắn `feature/ear`, `feature/head-pose`, `feature/lstm`, `feature/realtime`; PR nhỏ theo phase, merge sau tests/smoke. `develop` chỉ thêm nếu team thật sự cần tích hợp nhiều người, không bắt buộc GitFlow.

Commit docs/config/code/test; không raw data, video mặt, weights lớn, credential, calibration profile cá nhân. Log JSONL sự kiện/error và CSV/Parquet predictions; rotation theo size/session, không log ảnh mặc định. Notebook gọi src, không giữ một bản thuật toán riêng. Pin environment sau Phase 0, cập nhật changelog theo milestone.

## Lệnh Phase 0–2 hiện tại
Chạy `.venv/Scripts/python.exe -m scripts.check_environment`, `-m scripts.acquire_manifest`, `-m scripts.explore_dataset` từ project root theo [README](../README.md). Smoke đầy đủ cần `--sample-video`; `--assets-only` chỉ tải asset, không chứng minh environment. `manifest.parquet` có thêm frame_count/part_id/fold_source/status/error; mọi video lỗi vẫn được giữ để audit.
