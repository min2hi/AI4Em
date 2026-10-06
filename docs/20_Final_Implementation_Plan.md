# 20 — Kế hoạch triển khai cuối cùng

> Dành cho coding agent: triển khai từng phase của [16](16_Development_Roadmap.md), đối chiếu contract [15](15_Module_Specification.md). Có thể dùng workflow executing-plans hoặc subagent-driven-development; không triển khai toàn project trong một lượt không checkpoint.

**Goal:** Demo desktop phân loại ba mức tỉnh táo từ feature khuôn mặt theo thời gian, có evaluation đúng người và cảnh báo ổn định.

**Architecture:** Một pipeline raw feature dùng offline/realtime; dữ liệu Parquet cho training; RF và LSTM so sánh cùng windows; live calibration nhẹ, buffer và smoothing/alerts có timestamp.

**Tech Stack:** Python 3.12, OpenCV contrib, MediaPipe Tasks, NumPy/Pandas/PyArrow, scikit-learn, PyTorch, PySide6; XGBoost tùy chọn. Phiên bản đề xuất và bước kiểm tra compatibility ở tài liệu 03.

## Global constraints
- Không random split theo frame; không fit các biến đổi chung trên validation/test.
- Classes nội bộ `0/1/2` cố định; no-face là trạng thái hệ thống, không Alert hoặc lớp thứ tư.
- Landmark tối đa 20 FPS; chuỗi 10 Hz, dài 10 giây, stride 1 giây; tensor đầy đủ 100×16. Thay đổi phải cập nhật config/hash và train lại.
- PERCLOS proxy dùng 60 giây gần nhất, coverage≥0.80, lịch sử tối thiểu 30 giây; tên và định nghĩa không nói quá bằng chứng.
- P0/P1 có checkpoint riêng; P1 cần profile hợp lệ, không tự đổi chế độ khi lỗi.
- Không hoàn thành phase bằng mock, scaffold hoặc metrics giả. Test và chạy thử thật là bắt buộc.
- Không commit raw data, ảnh mặt, profile cá nhân hoặc credentials; tuân thủ consent/license.

## Checklist triển khai theo checkpoint
- [ ] **Checkpoint A — Dữ liệu:** Phase 0–2; môi trường đã chạy thử, quyền truy cập hợp lệ, manifest và folds chính thức theo người. Review nguồn/quyền sử dụng trước xử lý toàn bộ.
- [ ] **Checkpoint B — CV/MVP 1:** Phase 3–7, bổ sung Phase 10 sau khi có profiles; EAR/MAR/pose và overlay đúng, mẫu invalid có lý do. Một teammate xem video overlay, không chỉ đọc tests.
- [ ] **Checkpoint C — Dataset/MVP 2:** Phase 8–11; Parquet qua QC, loại calibration prefix, split/scaler không leak, RF có báo cáo một fold. Khóa schema trước LSTM.
- [ ] **Checkpoint D — Temporal ML:** Phase 12–15; train LSTM, chạy năm folds và ablation. Review Macro F1, recall Low Vigilance/Drowsy, khoảng tin cậy và coverage; không chọn checkpoint bằng test.
- [ ] **Checkpoint E — Live/MVP 3:** Phase 16–19; quản lý calibration, buffer 100 bước, smoothing ba mẫu và cảnh báo nhiều mức. Chạy webcam/audio thật; kiểm tra mất mặt, phục hồi và dữ liệu cũ.
- [ ] **Checkpoint F — Final desktop:** Phase 20–24; Qt UI, external test trên NTHU có nhãn, đo hiệu năng, kiểm thử release và báo cáo/demo có consent.

Checkpoint B có Phase 10 phụ thuộc Phase 9; thứ tự số phase ở tài liệu 16 là chuẩn khi triển khai tuần tự. Nếu cần demo CV trước xử lý toàn dataset, chỉ preview landmark/EAR/MAR, chưa gọi ML hoàn chỉnh. Không bỏ dependencies để tuyên bố milestone đã đạt.

## Handoff template cho một phase
“Implement Phase N theo tài liệu 16. Đọc contracts ở 15 và feature policy ở 06/07. Chỉ sửa files trong phase hoặc dependency bắt buộc; không đổi class mapping/schema âm thầm. Trả lại files, tests/chạy thử đã thực hiện, output artifacts và blockers có bằng chứng. Chưa đủ acceptance thì không đánh dấu done.”

## 20 câu hỏi nghiên cứu — câu trả lời ngắn và nơi đọc
| # | Câu hỏi | Quyết định / câu trả lời |
|---|---|---|
| 1 | EAR-only không đủ? | Hình dạng mắt, blink và pose gây nhầm; Low Vigilance chưa chắc nhắm mắt lâu. 02/06. |
| 2 | Cần temporal? | Thời lượng và thứ tự phân biệt blink với closure; thống kê RF cũng dùng thời gian. 02/07/08. |
| 3 | Low Vigilance khác Drowsy? | KSS 6–7 chưa cần gắng sức giữ tỉnh; 8–9 phải cố không ngủ. Nhãn video vẫn là nhãn yếu. 02/04. |
| 4 | Vì sao UTA? | Có ba lớp cần thiết, buồn ngủ thật, 60 người và folds theo người; không phải dữ liệu chạy xe ngoài đường. 04. |
| 5 | NTHU train chung? | V1 chỉ external test, proxy nhị phân; IR, hành động diễn và labels khác. Không tạo lớp Low Vigilance giả. 04/10. |
| 6 | MediaPipe đủ chính xác? | Hợp lý cho prototype, chưa chứng minh mọi điều kiện; phải kiểm tra overlay, nhãn tay và coverage ngoài dataset train. 03/19. |
| 7 | PERCLOS tính thế nào? | Thời gian closed proxy/thời gian quan sát hợp lệ trong 60 giây; loại missing. EAR proxy không phải phép đo PERCLOS80 sinh lý chính xác. 06. |
| 8 | Window bao lâu? | Mặc định 10 giây, thử 5/20 giây; thống kê 60 giây giữ riêng. 07/18. |
| 9 | FPS inference? | Landmark tối đa 20 FPS, chuỗi 10 Hz, prediction 1 Hz; blink ngắn cần FPS nguồn đủ cao. 07/12. |
| 10 | Vì sao LSTM? | Chuỗi input nhỏ, học diễn biến, dễ giải thích; một layer, hidden size 64. 08. |
| 11 | LSTM cần hơn RF? | Chưa biết; RF summary là baseline mạnh. So sánh công bằng và thử giá trị của thứ tự thời gian. 08/18. |
| 12 | Feature nào có giá trị? | Eye/time có bằng chứng; mouth/pose/personalization cần ablation A–E, không thêm chỉ để phức tạp. 06/18. |
| 13 | Tránh leakage? | Folds theo người, split trước fit, loại calibration prefix, scaler chỉ fit train, không tune trên test. 09/11. |
| 14 | Người chưa gặp đánh giá? | Năm outer folds; P0 không có thông tin cá nhân trước, P1 có calibration nên phải gọi đúng protocol. 04/10/11. |
| 15 | Calibration mức nào? | 30 giây, cần 20 giây hợp lệ, timeout 60 giây; normalize feature và threshold, không fine-tune. 11. |
| 16 | Kính/đêm/quay đầu/mất mặt/rung? | Quality gate, unknown và thông báo kỹ thuật; mount cố định. Không hứa suy luận tin cậy khi kính râm che mắt. 19. |
| 17 | Không detect face? | NO_FACE, mẫu invalid, hủy timer chờ/reset theo gap; không tự Alert hoặc Drowsy. 12/15. |
| 18 | Tránh cảnh báo giả? | Mean probability, hysteresis, thời gian duy trì, cooldown và quality; tune bằng validation replay, đo cảnh báo giả/giờ. 10/12. |
| 19 | Smoothing? | Mean ba prediction hợp lệ mỗi giây, gap≥2 giây thì reset; log raw và smooth riêng. 12. |
| 20 | Bottleneck? | Đo decode, MediaPipe CPU, chuyển màu, UI và storage trước LSTM nhỏ; không assume GPU giải quyết mọi thứ. 10/19. |

## Definition of Final
Có source/module thật; môi trường lock đã chạy thử; model/checkpoint, config và splits tái lập được; báo cáo baseline/LSTM/ablation/external/performance; UI/camera/audio thật; tests và giới hạn/quyền sử dụng rõ. Bộ tài liệu hiện tại **chưa có các artifact implementation này**; đó là sản phẩm coding agent phải bàn giao sau từng phase.
