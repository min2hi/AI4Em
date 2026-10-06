# 17 — Chiến lược kiểm thử

Unit tests dùng `pytest`, fixture tổng hợp deterministic, không phụ thuộc webcam/GPU/download. Integration và hardware smoke tách marker; CI không cố mở camera. **Tests phải kiểm tra hành vi hoặc số đúng**, không chỉ “không exception” hay shape không rỗng.

## Test matrix bắt buộc
| Thành phần | Case và kết quả mong đợi |
|---|---|
| EAR | p1=(0,0), p4=(4,0), hai khoảng dọc bằng 1 → EAR=0.25; phóng to/dịch chuyển không đổi kết quả; mẫu số 0 hoặc NaN → invalid; ảnh không vuông phải đổi normalized sang pixel |
| MAR | Chiều ngang 4, ba khoảng dọc bằng 2 → MAR=0.5; miệng đóng gần 0; chiều ngang 0 → invalid |
| Head pose | Dùng projectPoints tạo góc biết trước ±10° → sai số dưới 1° khi không có nhiễu; resize kèm scale K giữ góc; lỗi chiếu lại lớn → invalid |
| Timestamp/resample | Nguồn 15/24/30 FPS và timestamp không đều → chuỗi 10 Hz chỉ dùng quá khứ; mẫu cũ hơn 100 ms → unknown; timestamp trùng/đi lùi → reject |
| Blink/closure | Nhắm 0.15 giây rồi mở → blink; trên 0.8 giây → prolonged closure, không đếm blink; mất mẫu giữa event → cắt event, không nối |
| PERCLOS proxy | Nhắm 2 giây / quan sát hợp lệ 10 giây = 0.2; thêm 2 giây unknown không vào mẫu số, coverage=10/12; cắt biên 60 giây đúng; lịch sử dưới 30 giây hoặc coverage<0.8 → chưa ready |
| Yawn candidate | MAR delta vượt ngưỡng mở ít nhất 2 giây rồi xuống ngưỡng đóng → candidate; nói ngắn không đủ duration; missing cắt event, không gọi là ngáp thật |
| Split | Tập subject không giao nhau; video cùng người không tách; mỗi người làm outer test đúng một lượt; thiếu fold → reject; calibration prefix không vào window |
| Normalization | EAR 0.25 / baseline 0.25 = 1; baseline mắt trái 0.33, mắt phải 0.25 áp dụng riêng; baseline MAR bằng 0 không gây phép chia; thay test data không đổi scaler/baseline chung |
| Sequence/buffer | Từ t0 đến t0+9.9 giây có 100 mẫu; stride 1 giây tạo đúng index; missing>20%, gap>1 giây hoặc mẫu cuối invalid → không predict; đổi video → reset; mask giữ 0/1 |
| Model bundle | Sai thứ tự feature/schema/hash/mode → reject; reload cùng weights cho probability tương đương; thứ tự class `[0,1,2]` đúng |
| Metrics | Labels `[0,0,1,1,2,2]`, predictions `[0,1,1,1,2,0]` → Accuracy=4/6; recall=[0.5,1,0.5], Macro F1=59/90≈0.65556; support mỗi lớp bằng 2 |
| Smoother | Ba vector probability có mean đúng, tổng bằng 1; thiếu ba mẫu → null; gap≥2 giây → clear; NaN hoặc sai shape → reject |
| Alerts | Fake clock kiểm tra đúng các mốc 2/3/5/10 giây; một prediction cao chưa cảnh báo; unknown hủy timer chờ; strong warning không bị cooldown thường chặn; mute chỉ tắt audio |

Case PERCLOS 10 giây chỉ kiểm tra phép tích phân bên trong, không yêu cầu UI ready trước 30 giây. Kiểm tra ngay trước, đúng và ngay sau mỗi ngưỡng thời gian; không chỉ kiểm tra trường hợp thuận lợi.

## Integration và smoke
1. **Offline mini clip:** đọc video → landmark → features → Parquet → reload → window → cold-load model → prediction. Kiểm tra timestamps/schema và raw/replay equivalence.
2. **Dataset mini run:** người train/val/test thật tách biệt, train một fold; metrics và rejected report xuất được. Không dùng vài frame cùng người ở cả hai tập để smoke “training đúng”.
3. **Webcam:** nhìn thẳng, chớp nhanh, nói, nghiêng/quay đầu, che mặt, kính; low-light và kính râm phải hiện uncertainty. Diễn nhắm mắt chỉ kiểm tra event/warning, không làm ground truth trạng thái sinh lý.
4. **UI/audio:** Start/Stop năm lần, mute, recalibrate, rút camera, đóng cửa sổ; nghe âm thật, camera/thread được giải phóng; UI vẫn phản hồi khi worker bận.
5. **Performance:** chạy 10 phút và ghi latency từng bước, số frame bỏ, memory; đánh giá cảnh báo giả/giờ cần ít nhất một giờ Alert có consent, không suy từ clip ngắn.

## Offline/realtime parity
Replay dùng timestamp nguồn, không dùng tốc độ đọc file. Cùng frame stream/profile/checkpoint phải tạo feature và prediction giống trong sai số số thực cho phép; lịch UI có thể khác nhưng class/alerts theo thời gian nguồn phải khớp. Sai khác là lỗi pipeline, không giải thích chung bằng “khác vì realtime”.

## Lệnh bàn giao dự kiến
`pytest tests/ -q` cho unit; `pytest -m integration` khi có fixtures/model hợp pháp; `python scripts/webcam_demo.py --config configs/realtime.yaml` và `python main.py --config configs/realtime.yaml` cho smoke thật. Các lệnh này **chưa chạy được trong repo chỉ có tài liệu**, sẽ có sau các phase tương ứng.

Không cần unit test GUI chi tiết. Không test source text, tên file hay mock forwarding để chứng minh AI hoạt động. Không download dataset trong unit suite. Ghi rõ test skip do hardware/access; không coi skip là evidence pass realtime.
