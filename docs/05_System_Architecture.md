# 05 — Kiến trúc hệ thống

## Hai luồng, một bộ tính feature
```text
Offline: video + manifest → landmark → raw feature Parquet
         → split theo người → calibration/temporal transform
         → sequence index → train → checkpoint → evaluation

Realtime: webcam → worker → cùng landmark/feature/temporal transform
          → buffer → checkpoint → smoother → alert → PySide6
```

**Không chạy MediaPipe trong mỗi epoch**. Không viết hai công thức EAR hay hai cách normalize khác nhau cho offline/realtime. Chỉ VideoReader/CameraWorker và nơi lưu output khác nhau.

## Trách nhiệm và ranh giới
| Nhóm | Chịu trách nhiệm | Không chịu trách nhiệm |
|---|---|---|
| preprocessing | Đọc video, timestamp, chạy extraction và ghi Parquet | Tuning model |
| features | EAR/MAR/pose; event và thống kê nhân quả | UI hoặc đọc nhãn |
| datasets | Split, calibration transform, cửa sổ và tensor | Camera |
| models/training | Forward, loss, optimizer, checkpoint | Trích landmark |
| evaluation | Metrics, fold report, external test | Sửa threshold trên test |
| calibration | Ước lượng baseline, transform, lifecycle | Fine-tune mạng |
| realtime | Điều phối capture, buffer, prediction | Vẽ widget |
| alerts/ui | Hysteresis, audio, trình bày trạng thái | Thay đổi feature schema |

## Quy ước xuyên suốt
- Định nghĩa dữ liệu ở `src/contracts.py`; tên trường snake_case.
- `timestamp_ms` là thời gian nguồn, tăng nghiêm ngặt; webcam dùng monotonic clock. `frame_index` chỉ định danh, không dùng thay timestamp khi FPS thay đổi.
- BGR `uint8[H,W,3]` vào pipeline; đổi RGB đúng một lần trước MediaPipe. Tính khoảng cách trên tọa độ pixel, không dùng x/y normalized như cùng tỷ lệ trên ảnh không vuông.
- Preview có thể mirror; **feature và head pose dùng ảnh không mirror**.
- Giá trị chưa đo được là NaN/null + validity mask, không là số 0 giả. Chỉ trước model mới impute theo [07](07_Temporal_Modeling.md).
- `num_faces=1`, camera hướng cố định vào người lái. Không tự nhận dạng người; thay người phải restart calibration.
- Không có mặt hoặc dữ liệu quá cũ → không gọi model như thể có tín hiệu.

## Artifact contract
Checkpoint gồm weights, model config, ordered feature names, class mapping, scaler, feature schema version, landmark asset hash, calibration mode, preprocessing config hash và split hash. Loader từ chối nếu feature order/schema khác; không silently reorder bằng phỏng đoán.

Offline lưu raw feature để có thể đổi threshold và calibration mà không chạy lại landmark. Derived feature và sequence index được version theo config; kết quả experiment ở `runs/<experiment_id>/`.

## Xử lý lỗi
- Video hỏng: ghi error + video_id; report số thất bại, không bỏ im lặng.
- Timestamp trùng/đi lùi: reject sample và log; không reset về 0 giữa cùng session.
- Asset/checkpoint không có: fail rõ trước mở inference; UI `ERROR`.
- Mất face: hiện `NO_FACE`, tạm dừng cảnh báo AI mới; cảnh báo kỹ thuật riêng nếu kéo dài.
- CPU quá tải: bỏ frame capture cũ thay vì xếp hàng dài; vẫn ghi thời gian đã mất.

Interface chi tiết: [15](15_Module_Specification.md). Config: [14](14_Project_Structure.md).
