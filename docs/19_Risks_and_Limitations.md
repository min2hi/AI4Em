# 19 — Rủi ro và giới hạn

| Rủi ro | Tác động | Hành động mặc định |
|---|---|---|
| UTA weak video labels | Window không luôn đúng state, không đo onset thật | Ghi label_source; metrics nhiều cấp; không tuyên bố thời gian cảnh báo sớm |
| 60 người, giới/tuổi lệch | Nhiều window nhưng ít cá thể độc lập | Cross-validation theo người, CI theo người, báo cáo nhóm khi đủ mẫu |
| UTA indoor, NTHU mô phỏng IR | Khác biệt lớn với xe thật | External test và demo an toàn; không nói đã xác minh ngoài đường |
| Kính phản sáng | EAR/iris sai dù có landmark | Kiểm tra quality và coverage; điều chỉnh ánh sáng/camera |
| Kính râm | Landmark mắt có thể là suy đoán | Chọn chế độ mắt bị che → UNRELIABLE; v1 không chạy full model như thể còn quan sát được mắt |
| Thiếu sáng | Blur, landmark fail; RGB khác IR | Thông báo kỹ thuật yêu cầu cải thiện ánh sáng; không thêm xử lý ảnh chưa kiểm chứng |
| Quay đầu | EAR/MAR biến dạng; nhìn gương không phải buồn ngủ | Pose gate; unknown khi mắt không tin cậy; không “nhìn ngang=Drowsy” |
| Mất mặt/che mặt | Không có bằng chứng trạng thái | NO_FACE, ngừng warning AI mới, thông báo kỹ thuật; timer không nối qua gap |
| Camera rung | Pose velocity tăng vì camera, không vì đầu | Mount cố định; blur/gap gate; ghi giới hạn, không hứa bất biến với camera rung |
| Nói/cười/hát | Mở miệng giống ngáp | Chỉ gọi ngáp nghi ngờ; dùng duration; YawDD annotation tùy chọn |
| Calibration khi đã ngủ gà | Normalize sai, bỏ sót nguy cơ | Người dùng xác nhận Alert, kiểm tra quality/stability, retry rồi freeze; camera không tự chứng minh tỉnh |
| Một class bị reject nhiều | F1 accepted cao nhưng hệ thống bỏ nhiều nguy cơ | Coverage và rejected ratio theo class; số thời điểm nguy cơ bị abstain |
| Window chồng lặp, scaler leak | Score đẹp giả | Split theo người trước fitting, scaler train-only, bootstrap theo người |
| Smoothing/dwell | Giảm rung nhưng trì hoãn cảnh báo | Báo riêng warmup, latency tính toán và quyết định; tune trên validation replay |
| CPU bottleneck | RTX rảnh nhưng landmark chậm | Đo từng bước, giữ frame mới nhất, điều chỉnh resolution/FPS bằng số đo |
| Dataset access/license | Không làm được external test hoặc vi phạm quyền ảnh | Request từ Phase 1, access ledger, không raw GitHub; công bố ảnh đúng permission |
| Model bundle sai schema | Pipeline chạy nhưng thứ tự feature sai | Check schema/hash/mode và dừng rõ khi mismatch |
| Library update | Import/ABI/API thay đổi | Smoke rồi lock; nâng version phải kiểm thử và chạy demo lại |
| Privacy | Lộ khuôn mặt/profile | Consent, không tự ghi video, storage local và rotation; không upload raw |
| Cảnh báo giả/bỏ sót | Người dùng mất tin hoặc hiểu sai an toàn | Tune validation, giải thích giới hạn; không điều khiển xe hoặc bảo đảm tránh tai nạn |

## MediaPipe có đủ chính xác không?
**Đủ hợp lý để bắt đầu prototype**, chưa có bằng chứng đủ cho mọi điều kiện lái xe. Face Landmarker đo landmarks/expressions, không chứng nhận drowsiness. Phase 3–7 kiểm tra overlay và mắt mở/đóng/ngáp/pose trên subset có nhãn tay; Phase 21 đo coverage ngoài dataset train. Kính râm có thể tạo landmark sai dù detector không fail, nên chỉ `face_detected` là chưa đủ.

## Bottleneck dự kiến — cần đo, không khẳng định đã đo
Decode video, MediaPipe CPU, resize/chuyển màu, copy/render ảnh Qt và ghi dữ liệu preprocessing đáng đo trước LSTM nhỏ. Threading không bảo đảm tăng tốc tuyến tính; GPU có chi phí chuyển dữ liệu. Không tối ưu GPU sớm hoặc đọc raw video mỗi epoch.

## Những điều không làm
Không YOLO chỉ để detect face khi MediaPipe đủ cho prototype; không Transformer vì mới; không CNN raw video v1; không random split theo frame; không Accuracy-only; không EAR threshold chung làm hệ thống chính; không web frontend; không xử lý toàn dataset trước QC clip nhỏ; không assume thêm feature luôn tăng score.

## Các điểm chưa chắc phải ghi vào báo cáo
- Trạng thái Low Vigilance khó phân biệt và nhãn chủ quan.
- EAR proxy không phải PERCLOS80 đã calibrated theo mí/con ngươi; threshold/window/FPS cần experiment.
- V1 không suy luận tin cậy khi mắt bị che; không âm thầm chỉ dùng miệng/pose trên model chưa train cho chế độ đó.
- Chưa kiểm tra fold membership và tải đủ archive trực tiếp; số lượng từng partition NTHU và unique subjects YawDD cần manifest sau access.
- Compatibility mới kiểm tra docs/metadata, chưa cài toàn stack; chưa có metrics model hoặc FPS đo thực tế trong bộ tài liệu.
