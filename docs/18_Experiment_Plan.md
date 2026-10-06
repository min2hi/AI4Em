# 18 — Kế hoạch thực nghiệm

## Câu hỏi và experiment tối thiểu
| Experiment | Giả thuyết cần kiểm tra | So sánh |
|---|---|---|
| E01 — Model family | Chuỗi có thêm thông tin ngoài summary | EAR rule, temporal rule, RF và LSTM trên cùng split/cohort |
| E02 — Feature ablation | Mouth/pose hữu ích hay chỉ nhiễu | A–E dưới đây |
| E03 — Window | 10 giây cân bằng score và độ trễ | 5/10/20 giây, cùng 10 Hz, stride 1 giây |
| E04 — Sampling | Giảm landmark FPS có mất blink không? | Target 10/20 FPS; lưới model luôn 10 Hz; nguồn native cao hơn mới có ý nghĩa |
| E05 — Calibration | Baseline cá nhân giảm khác biệt mắt | P0 population vs P1 calibration-assisted, cùng architecture |
| E06 — Thứ tự thời gian | LSTM thực sự dùng order? | LSTM đã train: giữ nguyên vs đảo timestep để chẩn đoán; train lại một control với timestep đảo |
| E07 — External | Model UTA chuyển sang IR/kính/đêm được không? | P0 frozen trên NTHU, binary proxy và coverage theo bối cảnh |
| E08 — Deployment | Smoothing giảm báo giả đổi lấy độ trễ nào? | Raw, mean 3, mean 5 probabilities; CPU vs GPU forward + copy |

E01/E02/E05/E07/E08 là ưu tiên final; E03/E04/E06 mở rộng khi pipeline ổn. Không thêm model mới trước hoàn thành các so sánh cơ bản.

## Feature ablation A–E
| Model | Feature signal | Calibration mode |
|---|---|---|
| A | EAR trái/phải + thời gian nhắm mắt hiện tại | P0 |
| B | A + PERCLOS proxy | P0 |
| C | B + MAR delta + thời gian mở miệng nghi ngờ ngáp | P0 |
| D | C + pitch/yaw/roll + pitch velocity | P0 |
| E | D | P1 personalized |

Validity masks thuộc nhóm signal được giữ; `perclos_ready` chỉ khi có PERCLOS; `calibration_valid` có trong mọi biến thể (0 cho P0, 1 cho P1). Mỗi model có `feature_names`/input_size riêng và phải train lại; không đặt kênh về 0 trên model full rồi gọi là ablation tương đương.

Chạy LSTM cho A–E; RF full là control bắt buộc, RF subsets thêm nếu muốn phân biệt lợi ích feature với kiến trúc. Báo cáo **Macro F1, Drowsy Recall, Low Vigilance Recall**, thêm precision Low Vigilance để tránh tăng recall bằng cảnh báo mọi thứ; kèm latency và coverage.

## Fair comparison
- Cùng outer folds chính thức/validation policy/seed/quality gate. A–D trên cùng P0 windows; E so D trên **giao cohort hợp lệ P1**, loại calibration prefix khỏi D tương ứng; thêm toàn cohort coverage riêng để không giấu calibration failure.
- E được cấp dữ liệu Alert cá nhân trước inference; nói rõ đây là thêm thông tin, không phải model tự học người chưa gặp mà không dữ liệu phụ.
- Mỗi variant scaler train-only, checkpoint riêng; số lần thử hyperparameter tương đương. Không chỉnh model A theo test và bỏ qua cho B.
- Summary RF và LSTM cùng end timestamp/history; prediction raw dùng cho metrics classifier, smooth chỉ dùng đánh giá policy realtime.
- Feature benefit không chỉ một seed/fold; báo paired per-subject difference và cluster bootstrap CI. Không khẳng định significance nếu CI lớn hoặc số người ít.

## Bảng kết quả cần điền sau chạy
`experiment_id, model, feature_set, calibration_mode, outer_fold, seed, train_subjects, accepted_windows, coverage, macro_f1, weighted_f1, low_precision, low_recall, drowsy_recall, video_accuracy, p95_forward_ms, false_alarm_per_hour`.

Không có số kết quả mẫu giả trong tài liệu. Nếu một experiment chưa chạy, báo “chưa thực hiện” và nguyên nhân, không điền giá trị kỳ vọng như measurement.

## Quyết định sau experiment
Giữ feature khi cải thiện ổn định hoặc bổ sung độ bền trong điều kiện cần mà chi phí chấp nhận được. Nếu MAR/pose làm giảm score, kiểm tra extraction/quality trước; nếu không có lợi, chọn model gọn hơn và ghi bằng chứng. Nếu LSTM không hơn RF, vẫn trình bày experiment nhưng demo có thể chọn RF theo tiêu chí đã chốt, không ép kết luận neural network luôn tốt hơn.
