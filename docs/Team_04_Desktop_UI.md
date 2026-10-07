# Người 4 — Desktop UI và startup: kế hoạch thực hiện

> Phân công cho team4 người: tách UI khỏi người3. Dành cho coding agent: dùng executing-plans hoặc subagent-driven-development khi triển khai. Đây là kế hoạch, không phải xác nhận phase đã hoàn thành.

**Mục tiêu:** giao diện PySide6 hiển thị đúng trạng thái AI, điều khiển session/calibration/mute và đóng tài nguyên sạch.

**Kiến trúc:** MainWindow chỉ render UiSnapshot và gửi commands; CameraWorker của người3 giữ pipeline/detector người2. MainWindow quản lý Qt audio trên main thread, không viết logic warning hoặc inference riêng.

**Công nghệ:** Python3.12, PySide6 Widgets/QtCore/QtMultimedia; dùng môi trường/lock file repo, không thay stack UI.

## 1. Phạm vi và tài liệu phải đọc

Sở hữu phần desktop UI/audio playback Phase20, startup và end-to-end app thuộc Phase21. Phase19 warning policy/asset và camera worker thuộc người3; model/detector/replay thuộc người2, evaluation/external thuộc người1.

| Tài liệu | Nội dung cần dùng |
|---|---|
| [UI Architecture](13_UI_Architecture.md) | Layout, signals, rendering, audio và shutdown |
| [Module Specification](15_Module_Specification.md) | UiSnapshot, DetectionResult, AlertDecision, MainWindow APIs |
| [Roadmap](16_Development_Roadmap.md) | Phase20–21 acceptance/prerequisites |
| [Realtime Inference](12_Realtime_Inference.md) | Current/stale, warning policy, startup/recovery |
| [Personalized Calibration](11_Personalized_Calibration.md) | Hướng dẫn an toàn, progress/retry và P0/P1 |
| [Problem Definition](02_Problem_Definition.md) | SystemStatus không là class label |
| [Project Structure](14_Project_Structure.md) | Startup/config/assets convention |
| [Testing Strategy](17_Testing_Strategy.md) | Desktop smoke và parity |
| [Evaluation Strategy](10_Evaluation_Strategy.md), [Risks](19_Risks_and_Limitations.md) | Performance/false-alarm và giới hạn claims |

## 2. Files và hợp đồng với team

| Owner người4 | Trách nhiệm |
|---|---|
| `src/ui/main_window.py` | Widgets, commands, latest-snapshot rendering, Qt audio playback |
| `main.py` | Config/asset/bundle validation, startup và error presentation |
| `configs/realtime.yaml` | UI/audio settings; merge qua đầu mối config |
| README run instructions | Lệnh setup/cold-start/reproduce desktop; phối hợp section với owners khác |

- `MainWindow.render(snapshot)`, `start_session()`, `stop_session()` theo contract.
- Nhận `snapshot_ready(UiSnapshot)`, `status_changed(SystemStatus,str)`, `finished()` từ CameraWorker người3.
- Chốt stop/mute/calibrate/cancel commands và thread-safe handoff với người3. Không thêm lifecycle API ngầm hoặc tự sửa `src/contracts.py`.
- Audio chỉ thực thi AlertDecision command của người3; không tự áp probability thresholds, cooldown hoặc cảnh báo từ class string.
- ModelBundle load/validate dùng API người2/nhóm modeling. Thống nhất một lần load/session, không load lại trong mỗi render/frame.
- Phase20 full cần17–19 + worker người3 + UI này. Phase21 full cần14–20 và external evidence người1 nếu claim generalization.
- Có thể dựng render theo fixture UiSnapshot sớm để kiểm N/A/status/layout. Fixture không chứng minh native pipeline hoạt động; cuối cùng phải nối worker/model thật.

## 3. Checklist thực hiện

### Task1 — UI render đúng contract

- [ ] Tạo Start/Stop/Calibrate/Mute và preview; widget updates chỉ Qt main thread.
- [ ] Hiển thị system status, mode P1/P0, EAR trái/phải/mean, PERCLOS proxy + coverage/history, MAR/suspected yawn, pitch/yaw/roll, probabilities3 lớp, class, warning level, FPS và prediction age, calibration progress/errors.
- [ ] Probability chính là smooth và ghi rõ; raw ở debug nếu cần. Class prediction khác warning level, không sửa class theo policy warning.
- [ ] Missing/unavailable → N/A/status, không dùng0 giả. Current bad không được hiển thị probability cũ như hiện tại; age/stale rõ.
- [ ] Warning dùng chữ và màu, không chỉ màu. Startup idle không là AI Alert; no-face không là Drowsy.
- [ ] Render latest snapshot tối đa10Hz, không queue mọi frame. QImage sở hữu bytes trước frame buffer reuse; mirror preview sau extraction, không đảo feature sides.

### Task2 — Commands, audio và shutdown

- [ ] Start validate prerequisites và tạo session qua worker. Không inference/training/capture blocking trên main thread.
- [ ] Calibrate hướng dẫn người dùng xác nhận tỉnh ở nơi an toàn; cho cancel/retry; progress dùng elapsed/valid state upstream, không fake bằng animation timer độc lập.
- [ ] P1 fail hiện lỗi/retry, không tự dùng P0; P0 chỉ với checkpoint đúng mode và nhãn mode rõ.
- [ ] Mute indicator rõ, gửi command thread-safe; tắt audio đang phát, giữ warning/log. Qt audio trên main thread, không blocking sleep.
- [ ] Stop/close gửi interruption flag qua worker contract, dừng âm, chờ finished rồi quit/wait thread. Không terminate thread hoặc queued stop slot phụ thuộc event loop đang bị worker blocking.
- [ ] Camera/worker error vẫn giữ UI phản hồi, status/error rõ và cleanup. Start/Stop lại không giữ state session cũ.
- [ ] Asset âm có license từ người3; nghe loa thật đúng command. Không auto lưu video/ảnh mặt; evidence hình ảnh có consent.

### Task3 — Startup và integration Phase21

- [ ] `main.py` validate config/assets/bundle/schema/feature order/mode; exception startup có thông báo rõ, không fallback checkpoint sai mode hoặc no-op.
- [ ] Cold-start với bundle thật; cùng người2/3 chạy camera→calibration→warmup→prediction→smooth→warning→UI/audio.
- [ ] Cùng người2 chạy clip replay bằng shared pipeline; features/probabilities/alerts theo source timestamps khớp parity report. UI repaint schedule không phải lý do khác AI outputs.
- [ ] Người1 cung cấp coverage/metrics/external evidence; app chạy không đủ để gọi Phase21 generalization complete.
- [ ] Sau smoke cập nhật setup/run/model/config instructions và changelog đúng phần mình, bỏ fixture demo hard-coded khỏi production path và script thử.

## 4. Cách người giao việc kiểm tra đã xong

Không cần unit GUI chi tiết thay smoke; bắt buộc mở cửa sổ thật, không chỉ mocked widgets/headless screenshot.

| Scenario reviewer thao tác | Kết quả phải thấy |
|---|---|
| Cold start config/model đúng | App mở, đúng mode, status/calibration/warmup, không predict trước ready |
| Sai schema/mode/path | Lỗi cụ thể, không âm thầm fallback; tài nguyên startup release |
| Start/Stop5 lần | Không freeze/leak camera/thread, không probabilities session cũ |
| Calibrate fail/retry/complete/cancel | Progress/failure theo upstream; chỉ valid profile mới chạyP1 |
| Che mặt, hồi phục | NO_FACE/N/A/unavailable ngay; không audio AI mới từ stale |
| Mute/unmute | Indicator và âm đúng; class/warning/log không bị xóa |
| Resize khi worker bận | UI phản hồi, preview không hỏng bytes, queue không tăng dần |
| Rút camera | ERROR rõ, release; app không treo |
| Close trong calibration/inference/audio | Âm dừng, worker finished, thread/camera đóng; camera dùng lại được |
| Warning episode/escalation | Loa thật phát đúng command người3, class và warning riêng |

Bàn giao lệnh cold-start chính xác với model/config paths và dependency setup. `main.py` là deliverable tương lai: không coi tồn tại/chạy được chỉ vì roadmap có tên. Nếu CLI arguments được thêm, có `--help` và ví dụ thực với asset hợp pháp; không đưa command giả.

Suite hiện có:

```powershell
.venv/Scripts/python.exe -m pytest -q --tb=short
```

Không viết tests chỉ assert widget tồn tại, text copy, source code hoặc mock forwarding. Dùng tests hành vi cho boundary/lifecycle khó tái hiện nếu cần; desktop interaction vẫn là evidence bắt buộc.

## 5. Gói bàn giao và trạng thái

Lưu `runs/<id>/`: phiên bản code, input/model/config/hashes, environment, lệnh exact, checklist PASS/FAIL có timestamps, status/alert/audio log và ảnh/video có consent. Dùng artifact người2/3 thay log AI thứ hai không nhất quán.

- **DONE phần UI:** tất cả thao tác desktop có native proof, setup/run tái lập, không freeze/leak. Full Phase20 chỉ DONE khi worker/alerts người3 và detector người2 cũng nghiệm thu.
- **IMPLEMENTED — CHƯA NGHIỆM THU:** render có nhưng thiếu upstream/bundle/hardware/native run.
- **BLOCKED:** ghi prerequisite thiếu và scenario chưa kiểm; mock/skip không thành PASS.

Không claim safety từ UI đẹp/beep hoạt động. Diễn nhắm mắt chỉ kiểm event. Mục tiêu≤2 false audio alarms/giờ cần≥1 giờ Alert được xác minh;10 phút không đủ.

Liên quan: [Người1 — Evaluation](Team_01_Evaluation_Experiments.md), [Người2 — Realtime core](Team_02_Calibration_Realtime.md), [Người3 — Alerts/worker](Team_03_Alerts_UI.md).
