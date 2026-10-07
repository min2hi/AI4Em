# 15 — Module contracts

Contract này là chuẩn cho coding agent. Shared records trong `src/contracts.py` giữ nguyên; reader/detector có ở Phase3, EAR/MAR ở Phase4/5, signed pose/quality/shared pipeline/builder ở Phase6/7. Calibration/temporal/model/realtime còn lại là thiết kế. Modules nhận config/dependency qua constructor, không đọc config toàn cục. Thời gian ms, duration giây, góc độ, velocity độ/giây; probability/coverage ở0–1, **EAR/MAR không có upper bound1**.

## Shared records — `src/contracts.py`
| Type | Trường tối thiểu |
|---|---|
| FramePacket | `image_bgr:uint8[H,W,3], timestamp_ms:int, frame_index:int, source_id:str` |
| LandmarkResult | `points:float32[N,3] hoặc null` tọa độ normalized, `timestamp_ms`, `face_detected:bool`, `image_size:(W,H)`; không bịa visibility/confidence |
| EyeFeatures | `ear_left/right/mean:float hoặc NaN`, `left_eye_valid/right_eye_valid:bool` |
| MouthFeatures | `mar:float hoặc NaN`, `mouth_valid:bool` |
| PoseFeatures | `pitch/yaw/roll:float hoặc NaN`, `pose_valid:bool`, `reprojection_error_norm:float` |
| FeatureSample | timestamp, eye/mouth/pose fields, `face_detected`, source/frame IDs; **không label trong realtime record** |
| CalibrationProfile | baselines từng mắt/MAR/pose, valid, schema/asset/resolution, quality stats; baseline population có mode P0 |
| TemporalSample | timestamp, 16 giá trị theo [07](07_Temporal_Modeling.md), validity trước impute, `segment_id`, event summaries |
| SequenceWindow | `x:float32[100,16]`, timestamps start/end, quality, feature_names/schema; training wrapper thêm label_id/subject/video metadata ngoài X |
| Prediction | `timestamp_ms` của cuối window, `probabilities:float32[3]`, class_id argmax, `valid`, reason, model_id |
| DetectionResult | raw/smoothed Prediction hoặc null, system_status, quality, calibration status |
| AlertDecision | level 0/1/2, strong bool, audio command hoặc null, message, timestamp; không đổi class của Prediction |
| UiSnapshot | preview sở hữu bytes, FeatureSample/temporal display, DetectionResult, AlertDecision, FPS/latency/staleness |

`SystemStatus` theo [02](02_Problem_Definition.md); idle có thể hiển thị trước Start nhưng không là label AI. Metadata storage có nhãn ở manifest, không truyền nhãn vào FeatureExtractor.

### Nguồn timestamp
VideoReader ưu tiên thời gian decode `CAP_PROP_POS_MSEC`, kiểm tra tăng nghiêm ngặt và đối chiếu duration. Nếu backend trả timestamp lỗi, chỉ fallback `frame_index / fps` khi metadata/khảo sát xác nhận nguồn constant-FPS và fps dương; ghi rõ phương pháp vào metadata. Với variable-FPS không có timestamp tin cậy, dừng preprocessing video đó và ghi reason, không âm thầm giả 30 FPS. Webcam ghi monotonic timestamp lúc capture, không lúc inference hoàn tất; chênh lệch dùng để đo tuổi frame.

Implementation Phase 3: webcam dùng **`time.perf_counter_ns()`**, clock monotonic `QueryPerformanceCounter` trên Python 3.12 Windows; timestamp ghi ngay sau capture, trước inference. `monotonic_ns()` ở môi trường này dùng GetTickCount64/15.625 ms và đã gây timestamp trùng trong smoke. Consumer đo tuổi frame phải dùng **cùng clock perf_counter**, không trộn epoch giữa clocks. Stats ghi implementation/resolution. File dùng thời gian decode và chỉ fallback khi caller truyền `constant_fps_verified=True` sau verification độc lập.

Sampling chọn frame đầu tiên mỗi bucket `floor((source_ms-origin_ms)*target_fps/1000)`, tối đa một frame/bucket và target≤20; không duplicate nếu nguồn chậm, không bịa một chuỗi timestamp đều. Khoảng cách giữa hai frame có thể jitter theo cadence nguồn; causal 10 Hz tick/sample-age thuộc Phase 7/temporal, không phải reader này. Prefix stop là `STOPPED`, full EOF mới là `EOF`; disconnect/decode lỗi là `ERROR`, có release.

## CV và offline
| Module / file | Purpose, Input → Output; Interface | Dependencies và quy tắc |
|---|---|---|
| VideoReader / preprocessing/video_reader.py | File/camera → iterator FramePacket; `VideoReader()`, `iter_frames(path,target_fps,*,constant_fps_verified=False)`, `iter_camera(camera_index,target_fps,*,width,height)`, `close()`, `stats` | OpenCV; decode tuần tự, chọn theo source timestamp, không seek mỗi frame; lỗi duration/FPS phải ghi; camera failure không là EOF |
| FaceLandmarkDetector / features/landmarks.py | FramePacket → LandmarkResult; `FaceLandmarkDetector(config)`, `detect(packet)`, `close()`, `stats` | MediaPipe Tasks VIDEO, asset; BGR→RGB, timestamp/source-bound session; tạo detector mới mỗi video; no-face/malformed points trả null + false, không confidence/visibility giả |
| EyeFeatureExtractor / features/eye.py | LandmarkResult → EyeFeatures; `EyeFeatureExtractor(epsilon:float)`, `extract(result)` | Stateless NumPy, x*W/y*H trước khoảng cách, z ignored; canonical `LEFT_EYE`/`RIGHT_EYE` theo [06]; từng mắt độc lập, mean chỉ khi cả hai valid |
| MouthFeatureExtractor / features/mouth.py | LandmarkResult → MouthFeatures; `MouthFeatureExtractor(epsilon:float)`, `extract(result)` | Stateless NumPy, inner-lip `MOUTH_HORIZONTAL`/`MOUTH_VERTICAL`; sum3 dọc/(3*ngang), không clamp và không kết luận ngáp từ một frame |
| HeadPoseEstimator / features/head_pose.py | `HeadPoseEstimator(canonical_path,camera_model,*,max_reprojection_error_norm,epsilon)`, `estimate(result) -> PoseFeatures`, `camera_metadata(image_size)` | Stateless ITERATIVE solvePnP; canonical D=diag(1,−1,−1), R=Rz(roll) Ry(−yaw_display) Rx(pitch); pitch+ xuống, yaw+ phải ảnh, roll+ CW unmirrored. K approximate công khai hoặc calibrated resize; selected XY, hull/depth/Euler/RMS gates; invalid giữ NaN/false và RMS hữu hạn khi đo được |
| QualityGate / features/quality.py | `QualityGate(config["quality"])`, `evaluate(packet,result,eyes,mouth,pose) -> QualityResult`; `measure_quality(image_bgr,result)` | Immutable flags/reasons/metrics snapshot; fixed256 ROI, independent widths; optical failure mask all, oblique/invalid pose mask eyes+mouth, manual occlusion eyes-only; no classifier/temporal state |
| FeaturePipeline / features/pipeline.py | `FeaturePipeline(config)`, `process(packet) -> FeatureSample`, `reset()`, `close()`; read-only last_landmarks/last_quality/stats/pose_estimator | Frozen config validated before native allocation; one detector/session. Strict source/time/index; clear diagnostics before processing/errors/reset/close. Reset closes/recreates native and clears counters; close terminal/idempotent. Raw unnormalized/no labels; per-stage timings distinct from detect |
| FeatureDatasetBuilder / preprocessing/builder.py | `FeatureDatasetBuilder(config,*,constant_fps_verified=False)`, `build(manifest,output_dir,*,video_ids=None) -> dict` | Working validator accepts partial subject coverage, acquisition validator stays strict. Streaming1024 scalar rows, fullEOF and release, source SHA before/after. Metadata-last marker, Parquet SHA/schema/footer/fingerprint required for resume; per-source failure continues, nonzero CLI |
| Snapshot runner / preprocessing/snapshot.py | `freeze_snapshot(config_path,manifest_path,run_dir,output_dir) -> dict`, `run_snapshot(config_path,manifest_path,run_dir,output_dir) -> dict` | Byte-copy freeze + self-hash/source/program/signature provenance; drift refuse, sequential unchanged builder against full frozen manifest, checkpoint every pending/completed/cached/failed member; no downloads/source-status mutation |
| Feature audit / preprocessing/audit.py | `audit_snapshot(config_path,manifest_path,output_dir,snapshot,extraction_report) -> dict` | Read-only pair validation and streaming row/identity/null/time/count/release audit; frame-weighted video/subject/class coverage, explicit failed support and overlapping reason counts; CLI publishes separate extraction-status manifest |
| TemporalFeatureExtractor / features/temporal.py | FeatureSample + CalibrationProfile → `list[TemporalSample]`; `update(sample, profile)`, `reset()` | Cập nhật event ở tốc độ raw, trả 0 hoặc nhiều tick 10 Hz đã đến hạn theo [06–07]. Với tick trước timestamp sample mới, dùng lịch sử trước sample đó; tick trùng timestamp được dùng sample mới. Không đưa sample tương lai vào tick quá khứ; không future interpolation |
| SequenceDataset / datasets/sequence.py | derived data + index + scaler → `(x, label_id, metadata)`; `__len__`, `__getitem__(index)` | NumPy/PyTorch Dataset, 100×16, float32; missing transform/scaler đúng [07] |

### Eye/mouth implementation semantics — Phase4/5
- Epsilon lấy từ resolved preprocessing config (hiện1e-6), hữu hạn/dương/non-boolean; áp cho denominator pixel đầy đủ:2*eye_width hoặc3*mouth_width. Denominator≤epsilon invalid, không dùng epsilon để bịa một ratio.
- Missing face/points trả NaN + false masks. Khi face/points present, sai floating ndarray[478,3] hoặc sai positive-integer tuple(W,H) là ValueError, không pad/truncate. Chỉ selected x/y phải finite và trong[0,1]; selected z hoặc unselected points không ảnh hưởng ratio.
- Closed vertical span0 với horizontal width hợp lệ trả valid0; invalid không đổi thành0. Invalid một mắt giữ mắt kia, nhưng EAR mean NaN. Không mutate points, không đổi timestamp, không thêm confidence/visibility.
- Geometry validity không là quality. Phase6/7 mask raw eye/mouth measurements theo pose/ROI/optics, giữ independent masks và finite RMS diagnostic khi có; không clamp ratio để che profile-view artifacts.
- Preview dùng một FeaturePipeline cho headless/GUI; derive gated display records từ sample, không extraction/lifecycle path thứ hai. Summaries valid-only/null, `inference_ms` detect-only; source cleanup failures được báo, các closers còn lại và source tiếp theo vẫn chạy. Mặc định không lưu ảnh.


## Model và evaluation
| Module / file | Purpose, Input → Output; Interface | Dependencies |
|---|---|---|
| BaselineClassifier / models/baseline.py | RF summary `[N,D]`, labels → fitted model; `fit(X,y,sample_weight=None)`, `predict_proba(X)` → `[N,3]` | sklearn + train-only imputer; class order phải remap 0/1/2, không tin thứ tự library ngầm |
| LSTMClassifier / models/lstm.py | `forward(x:[B,T,F])` → logits `[B,3]` | torch.nn; F từ feature_names, head [08], không softmax trong forward |
| ModelTrainer / training/trainer.py | model, train_loader, val_loader, config → best checkpoint/history; `fit()` | PyTorch hoặc baseline adapter; val Macro F1, early stopping; không nhận test loader |
| ModelEvaluator / evaluation/evaluator.py | labels/probabilities/metadata/coverage → metrics + artifacts; `evaluate(...)`, `aggregate_folds(reports)` | sklearn, plotting; classes/support, per-subject, undefined metric handling |
| ModelBundle / models/bundle.py | trusted checkpoint/config → model + transforms; `load(path)`, `validate_schema(names,hashes)` | torch/joblib; reject mismatch, no silent fallback |

## Calibration, realtime và UI
| Module / file | Purpose, Input → Output; Interface | Dependencies và failure |
|---|---|---|
| CalibrationManager / calibration/manager.py | valid raw samples → profile/status; `start(timestamp_ms)`, `update(sample)`, `finish()`, `reset()` | Profile estimators ở calibration/profile.py, policy [11]; timeout fail; không fine-tune |
| PredictionBuffer / realtime/buffer.py | TemporalSample → SequenceWindow hoặc null; `append(sample)`, `get_window()`, `reset()` | deque, scaler/bundle; đủ 100 bước và quality gate; tên buffer giữ **feature**, không giữ probability |
| PredictionSmoother / realtime/smoother.py | valid Prediction → smooth Prediction hoặc null; `update(prediction)`, `reset()` | deque 3 probabilities, expire theo timestamp, chưa đủ mẫu trả null |
| DrowsinessDetector / realtime/detector.py | FramePacket → DetectionResult; `process(packet)`, `reset_session()`, `close()` | FeaturePipeline, calibration, temporal, buffer, model bundle, smoother; status trước class, prediction cadence |
| AlertManager / alerts/manager.py | DetectionResult + timestamp → AlertDecision; `update(result, timestamp_ms)`, `set_muted(bool)`, `reset()` | State machine [12]; unknown không kéo timer, audio chỉ command |
| CameraWorker / realtime/camera_worker.py | camera/config → signal UiSnapshot/status; `start()`, `request_stop()`, `request_calibration()` | QtCore, capture latest slot, detector, alerts; không mutate widget, stop thread-safe |
| MainWindow / ui/main_window.py | UiSnapshot/button actions → UI và worker commands; `render(snapshot)`, `start_session()`, `stop_session()` | PySide6 Widgets/Multimedia; owns audio/UI, clean thread shutdown |

## Quy tắc lỗi interface
Bad config/path/schema là exception startup có thông điệp; sample chất lượng kém là output invalid có reason, không exception mỗi frame. EOF video là iterator kết thúc; camera read error là status ERROR với release. Probability invalid NaN, tổng không gần 1 hoặc sai shape → reject, log, không phát warning từ output đó.
