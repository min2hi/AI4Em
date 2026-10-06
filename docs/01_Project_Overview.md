# 01 — Tổng quan project

**Adaptive Driver Drowsiness Detection Using Facial Features and Temporal Analysis**  
Hệ thống phát hiện dấu hiệu giảm tỉnh táo qua đặc trưng khuôn mặt theo thời gian.

## Mục tiêu và phạm vi
- Camera quan sát một tài xế; dự đoán **Alert / Low Vigilance / Drowsy**.
- Ưu tiên nhận biết Low Vigilance, thay vì chỉ bắt mắt đã nhắm lâu.
- Desktop Python + PySide6, chạy trên laptop có RTX 4050 hoặc tương đương. Không điều khiển xe, không embedded, không web frontend.
- Đây là đồ án nghiên cứu và demo hỗ trợ cảnh báo, **không phải thiết bị an toàn được chứng nhận**.

## Pipeline được chọn
Video → OpenCV → MediaPipe Face Landmarker → EAR/MAR/head pose → đặc trưng thời gian → chuỗi → LSTM → làm mượt → trạng thái → UI/âm thanh.

Landmark được xử lý tối đa **20 FPS**; chuỗi model **10 FPS × 10 giây = 100 bước**, stride 1 giây. PERCLOS dùng lịch sử 60 giây riêng, không bị giới hạn bởi chuỗi 10 giây. Thông số là **đề xuất ban đầu**, sẽ kiểm chứng bằng ablation.

## Ba nhóm hệ thống cần so sánh
| Nhóm | Cách quyết định | Giá trị |
|---|---|---|
| Rule-based | EAR và thời gian nhắm mắt theo ngưỡng | Kiểm tra pipeline, baseline dễ giải thích |
| Traditional ML | Random Forest trên thống kê cửa sổ | Baseline mạnh, ít chi phí train |
| Temporal Deep Learning | LSTM trên chuỗi đặc trưng | Học thứ tự và diễn biến theo thời gian |

Không mặc định LSTM tốt hơn Random Forest. Nếu baseline tốt hơn, báo cáo trung thực và giải thích.

## Mốc bàn giao
1. **MVP 1 — CV:** webcam, landmark, EAR/MAR, cảnh báo rule có thời gian duy trì; chưa tuyên bố phát hiện sớm.
2. **MVP 2 — ML:** UTA-RLDD → Parquet → split theo người → Random Forest → báo cáo ba lớp.
3. **MVP 3 — Temporal:** LSTM, webcam, calibration, smoothing, warning.
4. **Final:** MVP 3 + PySide6 + ablation + external test + báo cáo hiệu năng.

Thứ tự triển khai và tiêu chí hoàn thành: [16](16_Development_Roadmap.md). Nguồn và các giả định: [03](03_Research_Background.md), [04](04_Dataset_Analysis.md).
