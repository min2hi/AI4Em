# 02 — Định nghĩa bài toán

## Nhãn đầu ra
| ID nội bộ | Class | Ý nghĩa theo UTA-RLDD |
|---|---|---|
| 0 | Alert | Tỉnh táo, tương ứng KSS 1–3 |
| 1 | Low Vigilance | Có dấu hiệu buồn ngủ nhưng chưa phải gắng sức giữ tỉnh, KSS 6–7 |
| 2 | Drowsy | Phải cố gắng không ngủ, KSS 8–9 |

Nhãn gốc UTA `0/5/10` được đổi thành `0/1/2`. **Số 5 của nhãn gốc không phải KSS 5**. KSS 4–5 không được tự tạo thành lớp mới. Nguồn: [trang UTA](https://sites.google.com/view/utarldd/home).

Input học máy là chuỗi feature, không phải ảnh raw. Output model là ba logits; inference đổi sang probability có thứ tự cố định `[Alert, Low Vigilance, Drowsy]`.

## Vì sao không EAR-only?
EAR đo độ mở mắt, không đo trực tiếp tỉnh táo. Chớp mắt, nheo mắt, hình dạng mắt, kính và quay đầu đều có thể làm EAR nhỏ. Low Vigilance đôi khi chưa có nhắm mắt kéo dài. Cần xem thêm thời lượng, xu hướng, miệng và tư thế đầu; thêm feature chỉ được giữ khi thực nghiệm chứng minh giá trị.

## Vì sao cần thời gian?
Một frame mắt nhắm có thể là chớp mắt bình thường. Chuỗi giúp phân biệt một lần nhắm ngắn với nhiều lần đóng mở chậm hoặc nhắm kéo dài. Thống kê cửa sổ cũng là temporal analysis; không chỉ LSTM mới sử dụng thời gian.

## Phân biệt trạng thái người và trạng thái hệ thống
- `driver_state`: một trong ba lớp, hoặc `null` khi chưa đủ dữ liệu.
- `system_status`: `CALIBRATING`, `WARMING_UP`, `READY`, `UNRELIABLE`, `NO_FACE`, `ERROR`.
- Unknown/NO_FACE **không phải lớp thứ tư để train**, cũng không phải Alert.
- Khi mất dữ liệu, UI hiện lý do và thời điểm kết quả cuối; không hiển thị probability cũ như kết quả hiện tại.

## Giới hạn tuyên bố
UTA gán nhãn theo trạng thái chủ đạo của cả video, không xác định chính xác lúc bắt đầu giảm tỉnh táo. Có thể đánh giá khả năng phân loại Low Vigilance; **không thể suy ra số giây cảnh báo sớm trước ngủ gật** nếu chưa có annotation chuyển trạng thái riêng.

## Tiêu chí project
- Không người nào xuất hiện ở cả train/validation/test trong một lượt đánh giá.
- Báo cáo Macro F1, recall Low Vigilance và Drowsy, tỷ lệ không thể dự đoán.
- Target kỹ thuật: inference mỗi giây, landmark hữu hiệu ≥15 FPS khi ánh sáng tốt trên máy demo; đo thực tế trước khi kết luận đạt realtime.
- Không đặt một con số Accuracy bắt buộc khi chưa có baseline; kết quả thấp nhưng protocol đúng vẫn có giá trị nghiên cứu.
