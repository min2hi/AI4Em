# 13 — PySide6 desktop UI

## Bố cục đơn giản
```text
DRIVER MONITORING SYSTEM        [Start] [Stop] [Calibrate] [Mute]
[ Camera preview             ]  Hệ thống: READY / NO_FACE / ...
[                            ]  Chế độ: P1 calibrated / P0 population
EAR trái/phải/trung bình        PERCLOS xấp xỉ (%), coverage, history
MAR — Ngáp nghi ngờ             Pitch / Yaw / Roll (độ)
Alert % | Low Vigilance % | Drowsy %
Class dự đoán — Warning level — FPS — Tuổi prediction (ms)
Calibration progress / thông báo lỗi / hướng dẫn nghỉ ngơi
```

Percentages dùng probability đã smooth và ghi rõ; probability raw có thể trong panel debug, không làm giao diện chính rối. N/A khi thiếu dữ liệu; không hiển thị EAR=0 hoặc probability=0% để giả vờ đo được. Cảnh báo vừa chữ vừa màu, không phụ thuộc màu duy nhất.

## Threading và signals
- `MainWindow` chỉ widget, button, render snapshot; mọi widget cập nhật trên main thread.
- `CameraWorker(QObject)` moveToThread, giữ detector/model/buffer; signals `snapshot_ready(UiSnapshot)`, `status_changed(SystemStatus,str)`, `finished()`.
- Capture worker riêng/latest-frame slot để decode không làm queue inference lớn; ui render qua timer tối đa 10 Hz và chỉ giữ snapshot mới nhất.
- Stop: request interruption, dừng capture, close landmarker, release camera, worker finished, thread quit/wait. Không dùng terminate thread; không gọi slot Stop qua queued signal nếu worker đang loop blocking và không xử lý event queue—dùng interruption flag thread-safe.
- `QImage` phải sở hữu copy bytes trước khi array frame được reuse; không giữ con trỏ đến buffer đã overwrite. Mirror preview sau extraction.
- Audio `QSoundEffect`/QtMultimedia quản lý trên Qt thread; AlertManager chỉ emit command, không ngủ/chặn worker.

## UI actions
Start kiểm tra asset/model/config và camera; Stop đưa về idle và tắt âm đang phát. Calibrate chỉ khi session cho phép, reset pipeline sau complete; Mute có indicator rõ. Không tự ghi video; lưu log feature/prediction tùy chọn với consent.

## Acceptance demo
Start/Stop 5 lần không leak camera/thread; resize cửa sổ vẫn mượt; mất face thì probability có nhãn stale/N/A và trạng thái kỹ thuật; calibration progress đúng thời gian; Mute không xóa UI warning; đóng cửa sổ release camera. Smoke trực tiếp trên desktop bắt buộc, không dùng mocked GUI làm bằng chứng chạy được.

Nguồn thread/UI: [Qt for Python](https://doc.qt.io/qtforpython-6/gettingstarted.html), [QThread](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html).
