# Người 2 — Calibration và realtime core: kế hoạch thực hiện

> Dành cho coding agent: dùng skill executing-plans hoặc subagent-driven-development khi triển khai; làm từng task và lưu bằng chứng. Đây là kế hoạch giao việc, không xác nhận phase đã hoàn thành.

**Mục tiêu:** stream video/webcam tạo prediction đúng thời gian, đúng calibration mode, không backlog và không dùng kết quả stale.

**Kiến trúc:** tái sử dụng FeaturePipeline, profile transforms, temporal extractor và ModelBundle; realtime chỉ thêm lifecycle/buffer/inference/smoothing. CLI và GUI dùng cùng detector; replay dùng timestamp nguồn.

**Công nghệ:** Python 3.12, OpenCV/MediaPipe, NumPy/PyTorch, deque; môi trường/lock file của repo. CPU inference mặc định, CUDA chỉ chọn sau đo forward + copy.

## 1. Phạm vi và tài liệu phải đọc

Bạn sở hữu Phase 16–18 và replay/parity của Phase 21. Không viết lại profile estimators Phase 9, temporal Phase 10, training hoặc UI/audio.

| Tài liệu | Nội dung cần dùng |
|---|---|
| [Roadmap](16_Development_Roadmap.md) | Phase 16–18, 21 và prerequisites |
| [Module Specification](15_Module_Specification.md) | Records, lifecycle APIs và clock contract |
| [Personalized Calibration](11_Personalized_Calibration.md) | P0/P1, 30/20/60 giây, stability và profile reset |
| [Realtime Inference](12_Realtime_Inference.md) | Cadence, stale/gap, smoothing và recovery |
| [Temporal Modeling](07_Temporal_Modeling.md), [Feature Engineering](06_Feature_Engineering.md) | Causal 10 Hz, masks, events và 100×16 sequence |
| [Machine Learning Models](08_Machine_Learning_Models.md) | Model semantics và features |
| [Project Structure](14_Project_Structure.md), [Testing Strategy](17_Testing_Strategy.md) | Storage, boundary/replay/hardware smoke |
| [Evaluation Strategy](10_Evaluation_Strategy.md) | Latency, coverage, measurement targets |
| [UI Architecture](13_UI_Architecture.md) | Handoff cho CameraWorker, latest-slot và shutdown |

## 2. Quyền sở hữu, interfaces và prerequisites

| File | Trách nhiệm |
|---|---|
| `src/calibration/manager.py`, `tests/test_calibration.py` | Calibration lifecycle |
| `src/realtime/buffer.py`, `tests/test_buffer.py` | Feature window buffer |
| `src/realtime/detector.py`, `tests/test_detector.py` | Orchestration/inference |
| `src/realtime/smoother.py`, `tests/test_smoother.py` | Probability smoothing |
| `scripts/webcam_demo.py` | CLI debug dùng chung pipeline, có video replay/webcam smoke |
| `src/evaluation/replay.py` | Replay/parity Phase 21 |
| `tests/test_integration.py` | Replay/parity cases; chia section với người 1 |
| `configs/realtime.yaml` | Calibration/cadence/device/smoothing; merge qua đầu mối config |

Contract bắt buộc:

- `CalibrationManager.start(timestamp_ms)`, `update(sample)`, `finish()`, `reset()` → profile/status.
- `PredictionBuffer.append(sample)`, `get_window()`, `reset()` → SequenceWindow hoặc null.
- `PredictionSmoother.update(prediction)`, `reset()` → smooth Prediction hoặc null.
- `DrowsinessDetector.process(packet)`, `reset_session()`, `close()` → DetectionResult.
- `replay_session(...)`: chốt arguments với người 1/3 trước triển khai; dùng cùng transforms/detector, không đường tính feature thứ hai.

Input cần nhận: Phase 9 pure profile functions, Phase 10 temporal extractor, Phase 13 bundle/checkpoint đúng mode/feature order/scaler. Phase 16 cần 9/13; Phase 17 cần 13/16/10; Phase 18 cần 17. Full Phase 21 còn phụ thuộc người 1, 3 và 4.

Người 3 sở hữu CameraWorker/latest-frame capture và alerts; người 4 sở hữu MainWindow, Qt audio playback và main.py. Thống nhất một capture implementation dùng chung CLI/worker; không tạo hai queue/capture pipelines. `src/contracts.py` không tự sửa; cần mở rộng record thì chốt với cả nhóm qua một owner.

## 3. Checklist thực hiện

### Task 1 — Phase 16: calibration lifecycle

- [ ] Viết regression mất mặt/timeout, unstable EAR, retry reset, baseline hai mắt khác nhau, profile đổi resolution/schema/asset; quan sát fail trước sửa và pass sau.
- [ ] Tái sử dụng estimator Phase 9; không fine-tune weights, không dùng prediction để xác nhận người tỉnh.
- [ ] Collect 30 giây wall-clock, ít nhất 20 giây valid; thiếu thì kéo dài nhưng timeout tổng 60 giây. Theo policy 11: eye candidate top50% rồi median, stability MAD/median ≤0.15, baseline >epsilon và vùng train-QC hợp lệ.
- [ ] Freeze profile trong session; không auto-adapt vào trạng thái ngủ. Retry/recalibrate/đổi camera hoặc schema invalidate profile và reset temporal/buffer/smoother.
- [ ] P1 fail → UNRELIABLE + retry, không silently fallback P0. P0 chỉ load checkpoint P0, mode hiển thị/log rõ.
- [ ] CLI thấy progress/complete/fail; chạy webcam calibration thật với người dùng xác nhận Alert ở nơi an toàn; chứng minh profile đã áp dụng.

### Task 2 — Phase 17: buffer và inference

- [ ] Viết regression 100 bước từ t0 đến t0+9.9s; missing>20%, gap>1s hoặc mẫu cuối invalid không predict; video/session đổi phải reset.
- [ ] Dùng temporal causal 10 Hz từ Phase 10; không future interpolation/forward-fill qua gap. Mask giữ semantics 0/1, scaler/feature order theo bundle.
- [ ] Load assets/model một lần/session; `model.eval()` và `torch.inference_mode()`. Reject schema/hash/mode mismatch, không fallback giả.
- [ ] Nhận FramePacket và trả DetectionResult với raw probabilities ba lớp, timestamp cuối window, quality/status; prediction mỗi 1 giây sau đủ 100 bước. Status vẫn cập nhật mỗi sample.
- [ ] Webcam timestamp từ capture bằng perf_counter clock theo contract; không dùng lúc inference hoặc trộn clock epochs. File replay dùng timestamp nguồn, không tốc độ đọc file.
- [ ] Current bad/stale >500ms → unavailable ngay. Gap landmark>1s reset buffer/segment, warmup lại 10s; no-face không thành Drowsy.
- [ ] Phối hợp người 3 dùng latest-frame slot dung lượng1, đếm drop; không backlog hàng đợi tăng vô hạn.
- [ ] Chạy webcam10 phút, ghi cadence/drop/latency/memory/probabilities; thử no-face rồi trở lại, restart và nguồn lỗi; chứng minh release tài nguyên.

### Task 3 — Phase 18: smoothing

- [ ] Regression mean ba vectors có đáp án độc lập; thiếu ba valid predictions → null; NaN/sai shape/sum invalid bị reject; ngay trước/đúng/sau gap2s và restart clear.
- [ ] Mean ba valid probabilities, không majority vote class. Log raw và smooth riêng; probabilities không được chỉnh để ép một class.
- [ ] Gap không có valid prediction ≥2s clear; current sample bad phải unavailable ngay, không đợi clear timer.
- [ ] Tích hợp vào detector, chạy replay thật so raw/smooth, báo độ trễ thêm và không dùng smooth cho classifier metrics của người 1.

### Task 4 — Replay/parity thuộc Phase 21

- [ ] Chạy cùng clip/profile/checkpoint qua offline và replay từ raw frames, cùng transforms/scaler/feature order.
- [ ] Đối chiếu valid features và probabilities từng timestamp; ghi tolerance số thực cụ thể trước so sánh, không bỏ timestamps lệch khỏi report.
- [ ] Replay giữ no-face/gap/restart và causal status. Class/alert policy theo thời gian nguồn phải khớp; lịch vẽ UI có thể khác.
- [ ] Cùng người 3 (worker/alerts) và người 4 (UI/audio/startup) chạy end-to-end raw/file hoặc camera → warning/UI; cùng người 1 đưa coverage/provenance vào integration report.

## 4. Cách người giao việc kiểm tra đã xong

| Scenario | Điều phải quan sát |
|---|---|
| Calibration thật | Progress đúng thời gian; complete chỉ khi đủ valid; profile applied; timeout/failure cho retry |
| Cold-start model | Load đúng mode/schema; warmup trước prediction; probabilities hữu hạn, ba lớp tổng gần1 |
| Che mặt rồi trở lại | NO_FACE/unavailable ngay; không hiện prediction cũ như current; gap dài warmup lại |
| Replay cố định | Output tái lập; offline/replay timestamp/features/probabilities khớp trong tolerance |
| Webcam10 phút | Prediction cadence/drop/memory có log, không backlog tăng; Stop release camera/model |
| Raw vs smooth | Mean đúng, đủ ba mới có result; gap2s clear; latency thêm được báo |

Không coi webcam all-no-face là chứng minh inference/live calibration hoạt động. Không dùng fixture giả thay cho full native path. Diễn nhắm mắt chỉ kiểm tra event, không là nhãn sinh lý Drowsy.

Lệnh scoped yêu cầu **sau khi đã triển khai các file test**:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_calibration.py tests/test_buffer.py tests/test_detector.py tests/test_smoother.py -q --tb=short
```

CLI chưa được xem là tồn tại/chạy được chỉ vì có trong roadmap. Bàn giao `--help` và lệnh thực với source/model/config/profile/output cụ thể; reviewer chạy lại replay cố định rồi webcam. Lệnh suite hiện có: `.venv/Scripts/python.exe -m pytest -q --tb=short`.

## 5. Evidence và bàn giao

Lưu dưới `runs/<id>/`: command/provenance, profile + valid-duration/quality, raw/smooth predictions có timestamps/status/reasons, cadence/drop/latency/memory, parity report và hardware checklist. CLI log hoặc CSV/Parquet là bề mặt kiểm tra không cần GUI; không chỉ in mỗi class cuối cùng.

Targets performance của tài liệu10 không phải số đã đạt: landmark≥15FPS ánh sáng tốt, forward p95≤20ms, capture-to-feature p95≤150ms, prediction mỗi1s. Báo measurement thật và criterion chưa đạt; không đo GPU mà thiếu synchronize. Không tự giảm feature cadence mà không retrain.

- **DONE:** mọi acceptance phase có behavioral tests + smoke đường chạy thật + evidence, reviewer chạy lại được.
- **IMPLEMENTED — CHƯA NGHIỆM THU:** core/tests xong nhưng thiếu bundle/profile/hardware hoặc native run.
- **BLOCKED:** ghi prerequisite thiếu cụ thể; test skip/mock không là hardware PASS.

Sau smoke cập nhật hướng dẫn CLI và changelog đúng phạm vi, bỏ script thử tạm. Handoff người 3: detector lifecycle/status/raw+smooth result, mode/profile state, reset/close và clock semantics. Handoff người 4 qua UiSnapshot/worker: status, mode/calibration, prediction age và startup bundle validation; không tạo inference path riêng trong UI.

Liên quan: [Người 1 — Evaluation](Team_01_Evaluation_Experiments.md), [Người 3 — Alerts/worker](Team_03_Alerts_UI.md), [Người 4 — Desktop UI/startup](Team_04_Desktop_UI.md).
