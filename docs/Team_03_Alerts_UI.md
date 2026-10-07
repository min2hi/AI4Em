# Người 3 — Alerts và camera worker: kế hoạch thực hiện

> Phân công mới cho team 4 người. Giữ tên file cũ để không phá liên kết; desktop UI, Qt audio playback và `main.py` đã chuyển sang [Người 4](Team_04_Desktop_UI.md). Dành cho coding agent: dùng executing-plans hoặc subagent-driven-development khi triển khai. Đây không phải xác nhận phase đã hoàn thành.

**Mục tiêu:** warning state machine đúng thời gian và worker camera cung cấp UiSnapshot không backlog, shutdown sạch.

**Kiến trúc:** AlertManager thuần trả audio command; CameraWorker dùng detector người2 và emit snapshot/status. Người4 render widgets và thực thi audio command trên Qt main thread, không inference trong UI.

**Công nghệ:** Python3.12, PySide6 QtCore, OpenCV capture dùng chung realtime path; dùng môi trường/lock file repo.

## 1. Phạm vi và tài liệu

Sở hữu Phase19 và phần camera worker của Phase20; hỗ trợ end-to-end Phase21. Không sở hữu `src/ui/main_window.py`, `main.py`, Qt audio playback, model hoặc detector.

| Tài liệu | Nội dung cần đọc |
|---|---|
| [Roadmap](16_Development_Roadmap.md) | Phase19–21 và prerequisites |
| [Realtime Inference](12_Realtime_Inference.md) | Threshold, timers, cooldown, recovery |
| [UI Architecture](13_UI_Architecture.md) | QObject worker, signals, latest-slot và shutdown |
| [Module Specification](15_Module_Specification.md) | DetectionResult, AlertDecision, UiSnapshot, lifecycle APIs |
| [Personalized Calibration](11_Personalized_Calibration.md) | Calibration/reset coordination |
| [Testing Strategy](17_Testing_Strategy.md) | Fake-clock boundaries và hardware smoke |
| [Evaluation Strategy](10_Evaluation_Strategy.md), [Risks](19_Risks_and_Limitations.md) | Latency và false-alarm claims |
| [Project Structure](14_Project_Structure.md) | File/config/artifact convention |

## 2. Files và hợp đồng bàn giao

| Owner người3 | Trách nhiệm |
|---|---|
| `src/alerts/manager.py`, `tests/test_alerts.py` | Warning policy, audio command và regression |
| `models/assets/warning.wav` hoặc UI resources | Asset âm thanh có license/source rõ; người4 load/play |
| `src/realtime/camera_worker.py` | Capture/latest-slot, orchestration, snapshots và lifecycle |
| `configs/realtime.yaml` | Warning thresholds/dwell/cooldown; merge qua đầu mối config |

- `AlertManager.update(result,timestamp_ms)`, `set_muted(bool)`, `reset()` → AlertDecision, không đổi class của Prediction.
- `CameraWorker.start()`, `request_stop()`, `request_calibration()`; signals `snapshot_ready(UiSnapshot)`, `status_changed(SystemStatus,str)`, `finished()`.
- Nhận detector/profile/status từ người2; trả snapshot có preview sở hữu bytes, metrics, DetectionResult, AlertDecision và timing cho người4.
- Chốt đường truyền mute/calibration/cancel/stop thread-safe với người4 trước triển khai; không gọi trực tiếp widget hoặc mutate state xuyên thread.
- Người2 sở hữu `scripts/webcam_demo.py` và detector. Capture dùng chung một implementation giữa CLI và worker, không hai queue pipelines. Owner CLI tích hợp audio command theo hợp đồng.
- Người4 sở hữu UI/audio rendering và startup validation. Không tự sửa shared records; nếu cần đổi `src/contracts.py` phải chốt với cả nhóm qua một owner.
- Phase19 cần18; full Phase20 cần17–19 và UI người4. Worker xong riêng không đồng nghĩa Phase20 DONE.

## 3. Checklist thực hiện

### Task1 — Phase19: alerts

- [ ] Viết fake-clock regression ngay trước/đúng/sau2/3/5/10/15s; fail-before/pass-after. Bao gồm oscillation, unknown, mute, escalation.
- [ ] Level1: p_risk=p_low+p_drowsy≥0.60 liên tục3s hợp lệ → UI command; Level2: p_drowsy≥0.65 liên tục2s → UI+audio command, Level2 ưu tiên.
- [ ] Strong vẫn Level2 sau Drowsy10s valid; escalation một lần không bị cooldown thường chặn.
- [ ] Exit Drowsy khi p_drowsy<0.40 liên tục5s; về Level1 nếu p_risk≥0.40, ngược lại0. Exit Level1 khi p_risk<0.40 liên tục5s.
- [ ] Audio cùng mức cooldown15s; strong repeat tối thiểu15s. Dwell theo valid time, không số frame; unknown hủy pending timers, không audio mới từ stale.
- [ ] Mute chỉ tắt audio, không sửa prediction hoặc xóa warning/log. No-face≥2s là thông báo kỹ thuật, không kết luận ngủ.
- [ ] Manager không sleep/play sound/chặn worker. Ghi license asset; cùng người4 nghe âm thật đúng command ở nơi an toàn để nghiệm thu Phase19.

Threshold theo tài liệu12; chỉ tune validation/replay theo protocol, không tune test hoặc đổi để demo đẹp.

### Task2 — Camera worker thuộc Phase20

- [ ] QObject moveToThread; detector/model load một lần/session; capture latest-frame slot1, frame mới thay frame cũ chưa xử lý, drop counts rõ.
- [ ] Capture clock perf_counter theo contract; snapshot có timestamp/age, không trộn clock epochs hoặc dùng inference time thay capture time.
- [ ] Publish snapshot mới nhất cho render≤10Hz, không signal queue flood. Không widget/audio call trong worker.
- [ ] Preview bytes có ownership rõ trước buffer reuse; mirror thuộc UI sau extraction, không đảo feature sides.
- [ ] Thread-safe interruption flag cho stop; không dựa queued stop slot khi loop blocking. Dừng capture, close detector/landmarker, release camera, emit finished; UI owner quit/wait thread, không terminate.
- [ ] Camera/startup failure vẫn release phần đã mở; stop idempotent, restart session không giữ timer/buffer/profile cũ sai policy.
- [ ] Calibration/mute commands được truyền thread-safe; reset theo detector, không estimator thứ hai trong worker.
- [ ] Cùng người4 chạy Start/Stop5 lần, camera unplug và close khi đang xử lý; cùng người2 chạy10 phút để ghi drop/cadence/latency/memory.

### Task3 — Tích hợp Phase21

- [ ] Với người2, replay clip thật qua detector→alerts, so warning timeline theo source timestamps.
- [ ] Với người4, cold-start camera→calibration→warmup→prediction→warning/UI; chứng minh mute/stop/close.
- [ ] Người1 kiểm metrics/coverage và external; không claim full Phase21 khi external blocked.
- [ ] Sau smoke cập nhật worker/alert run guidance và changelog đúng phạm vi; bỏ scripts thử.

## 4. Cách người giao việc kiểm tra đã xong

Lệnh scoped yêu cầu sau khi file test được triển khai:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_alerts.py -q --tb=short
```

Suite hiện có: `.venv/Scripts/python.exe -m pytest -q --tb=short`. CLI/app thuộc deliverable tương lai: bàn giao lệnh thật có arguments model/config/source cụ thể, không giả định tên trong roadmap là đã chạy được.

| Kiểm tra | Kết quả mong đợi |
|---|---|
| Fake-clock alert timeline | Một prediction cao chưa alarm; timers/exit/cooldown đúng; unknown hủy pending; strong vượt cooldown thường |
| Mute | Warning/log giữ nguyên, không audio command khi muted |
| Worker camera thật | Timestamp/drop/status và snapshots thật; không chỉ mock callback |
| Start/Stop5 lần cùng người4 | Mỗi lần release, không session cũ, camera dùng lại được |
| Unplug/close đang chạy | ERROR rõ, finished và cleanup; không treo/leak thread |
| Loa thật cùng người4 | Audio đúng episode/escalation; không audio AI từ stale |

Lưu `runs/<id>/`: code/model/config/input provenance, lệnh, timestamped predictions/status/AlertDecision/audio command, drop/latency/memory, hardware checklist và asset license. Ảnh tĩnh không chứng minh release/audio.

## 5. Điều kiện bàn giao

- **DONE:** Phase19 đủ behavior + audio thật; phần worker có hardware proof. Chỉ cùng người4 xác nhận full Phase20 khi UI checklist cũng đạt.
- **IMPLEMENTED — CHƯA NGHIỆM THU:** core có nhưng thiếu detector/model/hardware/native integration.
- **BLOCKED:** nêu prerequisite và evidence thiếu; mock/skip không thành hardware PASS.

Diễn nhắm mắt không ground truth sinh lý. Không claim≤2 false audio alarms/giờ từ clip10 phút; cần≥1 giờ Alert được xác minh, report unknown time.

Liên quan: [Người1 — Evaluation](Team_01_Evaluation_Experiments.md), [Người2 — Realtime core](Team_02_Calibration_Realtime.md), [Người4 — Desktop UI](Team_04_Desktop_UI.md).
