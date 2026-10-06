# 12 — Realtime và cảnh báo

## Luồng vận hành
Camera capture → latest-frame slot → CameraWorker (MediaPipe VIDEO) → FeaturePipeline → CalibrationManager → TemporalFeatureExtractor → PredictionBuffer → checkpoint → PredictionSmoother → AlertManager → UI.

Dùng worker riêng, không inference trong Qt main thread. Capture có slot dung lượng 1: frame mới thay frame cũ chưa xử lý; đo dropped-frame count. Landmark tối đa 20 FPS, resample causal 10 FPS; inference mỗi giây khi buffer đủ 100 bước. `deque` chứa feature/timestamp tối đa 100 timestep; temporal statistics giữ riêng 60 giây.

`model.eval()` + `torch.inference_mode()`; load asset/checkpoint một lần/session. CPU inference mặc định; CUDA chỉ chọn nếu đo forward + copy nhanh hơn. Không chuyển tensor qua GPU mỗi frame để “tận dụng RTX”.

## Smoothing được chọn
Trung bình probability **3 prediction hợp lệ gần nhất**, cadence 1 giây; không majority voting class vì làm mất confidence. Chỉ xét trạng thái AI khi có đủ 3 mẫu. Probability gốc và smooth được lưu riêng.

- Xóa smoother và pending transition khi không có prediction hợp lệ ≥2 giây, restart camera/model, hoặc recalibrate.
- Sample hiện tại thiếu/già >500 ms → ngay lập tức UNRELIABLE/NO_FACE, dù prediction trước còn tốt. Không đợi 2 giây để ngừng hiển thị kết quả như hiện tại.
- Gap landmark >1 giây → buffer/segment reset và warmup 10 giây sau phục hồi. Hysteresis không cộng dồn thời gian unknown.

## State machine cảnh báo
Các probability threshold dưới đây phải tune validation/replay; không lấy test để chỉnh.

| Trạng thái | Điều kiện vào liên tục trên prediction hợp lệ | Cảnh báo |
|---|---|---|
| Level 0 — Alert | Sau warmup/smoothing, không đủ điều kiện nguy cơ | Không cảnh báo |
| Level 1 — Low Vigilance | `p_low+p_drowsy ≥0.60` trong 3 giây, chưa Level 2 | UI: “Bạn đang có dấu hiệu giảm tỉnh táo.” |
| Level 2 — Drowsy | `p_drowsy ≥0.65` trong 2 giây | UI + một audio episode |
| Strong warning — vẫn Level 2 | Drowsy giữ liên tục 10 giây hợp lệ | Audio mạnh: “Phát hiện dấu hiệu buồn ngủ. Hãy dừng xe và nghỉ ngơi.” |

Ưu tiên Level 2 khi cả hai điều kiện đúng. Probability threshold cho warning là policy vận hành, không đổi class argmax trong metric model. UI hiển thị **class prediction** và **warning level** riêng để không gọi argmax Alert là Low Vigilance mà không giải thích.

Thoát Drowsy khi `p_drowsy<0.40` liên tục 5 giây; sau đó về Level 1 nếu p_risk≥0.40, ngược lại Level 0. Thoát Level 1 khi p_risk<0.40 liên tục 5 giây. Phải đủ dữ liệu valid trong thời gian duy trì; không tính bằng số frame.

Audio phát khi vào episode; cooldown 15 giây cho lặp audio cùng mức. Strong warning được phát một lần ngay khi escalation, không bị cooldown audio thường chặn; sau đó strong repeat tối thiểu 15 giây. Mute chỉ tắt âm, không tắt UI/log và không đổi prediction. Unknown hủy pending timer, giữ lịch sử episode trong log nhưng không phát audio AI mới từ probability cũ. Face mất ≥2 giây → thông báo kỹ thuật “Không quan sát được khuôn mặt”, không nói người đang ngủ.

## Startup và recovery
`CALIBRATING → WARMING_UP → READY`. Với model P1, calibration fail → UNRELIABLE + retry; không fallback model sai mode. Camera lỗi → ERROR, release tài nguyên. Resume hoặc đổi nguồn tạo session mới; warmup lại, không nối lịch sử.

## Pseudocode điều phối — không phải implementation
```text
on sample(timestamp):
    update raw quality; if stale/invalid: publish system status
    if calibration incomplete: collect or report failure; return
    update causal temporal features and 10 Hz buffer
    if scheduled decision and window accepted:
        raw_probability = model(window)
        smoothed = smoother(raw_probability, timestamp)
        warning = alert_manager(smoothed, timestamp, quality)
        publish snapshot with prediction timestamp
```

Đánh giá performance và false alarms theo [10](10_Evaluation_Strategy.md); unit scenario theo [17](17_Testing_Strategy.md).
