# Driver Drowsiness Detection — Bộ tài liệu thiết kế

**Hệ thống AI phát hiện sớm dấu hiệu giảm tỉnh táo của tài xế qua khuôn mặt và diễn biến theo thời gian.**

Bộ này gồm **20 tài liệu kỹ thuật + README**, đặt trong `docs/`. **Phase0 đạt smoke; full acquisition Phase1–2 mới có34/45 video. Phase3–7 đã có reader/detector/EAR/MAR/signed pose, frozen quality, shared raw pipeline và streaming Parquet builder.** Phase7 đã QC3 video full EOF/36,755 rows, round-trip/resume và394 tests; chưa chạy toàn snapshot34. Live sign/physical occlusion/face transitions/disconnect vẫn chưa quan sát; camera45s Phase7 có806 no-face frames. Xem [roadmap Phase6/7](16_Development_Roadmap.md#phase-6--head-pose-estimation), [báo cáo Phase3 lịch sử](Phase3_Report.md), [Dataset Access](Dataset_Access.md), [changelog](CHANGELOG.md) và [README project](../README.md). Chưa train classifier; Phase8–24 chưa thực thi.

## Đọc theo nhu cầu
- **Mới vào team:** đọc 01 → 02 → 05, sau đó xem các mốc trong 16.
- **Làm computer vision:** 03 → 06 → 07 → 15, rồi Phase 3–7.
- **Làm dữ liệu/model:** 04 → 08 → 09 → 10 → 11 → 18.
- **Làm realtime/UI:** 11 → 12 → 13 → 15 → 17.
- **Giao việc cho AI agent:** 14 → 15 → phase tương ứng trong 16; dùng template ở 20.

Tài liệu mỗi phần giữ ngắn; roadmap dài hơn vì phải đủ input/output, files, interfaces, tests và acceptance cho **25 phase (0–24)**. Không cần đọc toàn bộ cùng lúc.

## Mục lục
| Tài liệu | Nội dung |
|---|---|
| [01 — Project Overview](01_Project_Overview.md) | Mục tiêu, phạm vi, ba MVP và Final |
| [02 — Problem Definition](02_Problem_Definition.md) | Ba class, nhãn và trạng thái không đủ dữ liệu |
| [03 — Research Background](03_Research_Background.md) | Papers, compatibility/API, lựa chọn và trade-off |
| [04 — Dataset Analysis](04_Dataset_Analysis.md) | UTA/NTHU/YawDD, access/license, folds và điểm chưa xác minh |
| [05 — System Architecture](05_System_Architecture.md) | Hai pipeline offline/realtime, artifact và lỗi |
| [06 — Feature Engineering](06_Feature_Engineering.md) | EAR, MAR, blink, PERCLOS proxy, pose, quality |
| [07 — Temporal Modeling](07_Temporal_Modeling.md) | Resample, missing, windows, nhãn sequence |
| [08 — Machine Learning Models](08_Machine_Learning_Models.md) | Rules, Random Forest, LSTM; XGBoost tùy chọn |
| [09 — Training Strategy](09_Training_Strategy.md) | Train, tuning, seed, logs/checkpoints và tái lập |
| [10 — Evaluation Strategy](10_Evaluation_Strategy.md) | Macro F1, Low Vigilance, coverage, external và hiệu năng |
| [11 — Personalized Calibration](11_Personalized_Calibration.md) | Baseline cá nhân, P0/P1, không fine-tune |
| [12 — Realtime Inference](12_Realtime_Inference.md) | Buffer, smoothing, warning state machine |
| [13 — UI Architecture](13_UI_Architecture.md) | PySide6, worker, preview, audio và shutdown |
| [14 — Project Structure](14_Project_Structure.md) | Thư mục, storage schema, YAML, logging và Git |
| [15 — Module Specification](15_Module_Specification.md) | Input/output/interface/dependencies của các module |
| [16 — Development Roadmap](16_Development_Roadmap.md) | Phase 0–24 có thể giao triển khai từng phần |
| [17 — Testing Strategy](17_Testing_Strategy.md) | Công thức, boundary, integration và smoke thật |
| [18 — Experiment Plan](18_Experiment_Plan.md) | Ablation A–E, calibration, temporal order, deployment |
| [19 — Risks and Limitations](19_Risks_and_Limitations.md) | Kính/đêm/rung/mất mặt, privacy và giới hạn tuyên bố |
| [20 — Final Implementation Plan](20_Final_Implementation_Plan.md) | Checkpoints, handoff và trả lời 20 câu hỏi nghiên cứu |

## Thiết kế mặc định — tra nhanh
| Mục | Chọn |
|---|---|
| Dataset train/evaluation | UTA-RLDD working snapshot34 videos/12subjects, restricted official fold membership; không full60 benchmark |
| Validation trong mỗi lượt | Một fold test, fold kế tiếp val, ba fold train; cardinality/eligibility thực tế, không cố định36/12/12 hoặc9/3/3 |
| External test | NTHU, binary risk proxy và coverage; không bịa lớp Low Vigilance |
| CV | OpenCV + MediaPipe Tasks Face Landmarker, CPU trước |
| Lấy landmark / chuỗi / dự đoán | Tối đa 20 FPS / 10 Hz / mỗi giây |
| Chuỗi đầy đủ | 10 giây × 10 Hz × 16 feature = 100×16 |
| PERCLOS | EAR proxy, trailing 60 giây, chỉ thời gian quan sát hợp lệ |
| Storage | Parquet mỗi video; không đọc raw video mỗi epoch |
| Model | RF baseline; LSTM một layer, hidden 64 |
| Calibration | 30 giây, tối thiểu 20 giây hợp lệ, timeout 60 giây; freeze baseline |
| Smoothing | Mean ba probability vector + hysteresis/thời gian duy trì |
| UI / environment | PySide6 Widgets / Python 3.12 x64 + venv |
| Metric chọn model | Macro F1; báo cáo riêng Low Vigilance Recall và Drowsy Recall |

## Thuật ngữ cho thành viên mới
- **Subject:** một người tham gia dataset; không phải một frame/video.
- **Landmark:** điểm trên mắt, miệng hoặc khuôn mặt; **feature:** số tính từ các điểm đó.
- **Temporal / sequence:** quan sát nhiều thời điểm liên tiếp, thay vì một ảnh đơn lẻ.
- **Window / stride:** đoạn dữ liệu dùng một lần dự đoán / khoảng dịch đến lần tiếp theo.
- **Baseline:** cách đơn giản làm mốc so sánh; không phải kết quả cuối được mặc định tốt nhất.
- **Calibration:** đo baseline riêng của người dùng trước vận hành; không đồng nghĩa train lại mạng.
- **P0 / P1:** không có calibration cá nhân / có calibration cá nhân riêng đã khai báo.
- **Normalization / scaler:** đổi thang đo feature; scaler chung chỉ học từ train.
- **Leakage:** thông tin của test lọt vào training/tuning, làm kết quả đẹp giả.
- **Mask / missing:** cờ cho biết giá trị đo được hay đang thiếu; số 0 không tự nghĩa là mắt nhắm.
- **Coverage / abstention:** tỷ lệ thời điểm đủ dữ liệu / hệ thống từ chối kết luận khi không đủ dữ liệu.
- **Weak label:** nhãn áp cho cả video, không bảo đảm đúng từng giây.
- **Ablation:** bỏ/thêm một nhóm feature để kiểm tra nó có thật sự hữu ích.
- **Hysteresis / dwell / cooldown:** ngưỡng vào/ra khác nhau / giữ điều kiện đủ lâu / chờ trước phát lại âm.
- **Smoke test:** chạy thử đường xử lý thật để thấy nó hoạt động; không chỉ kiểm tra code import.
- **Definition of Done / acceptance:** bằng chứng phải có trước khi gọi một phase hoàn thành.

## Bằng chứng và phần còn phải kiểm chứng
Thông tin nghiên cứu kiểm tra ngày **06/10/2026**, có link nguồn ở 03/04/06. Phân biệt:
1. **Đã đọc nguồn:** UTA có 60 người/180 video và five-fold protocol; NTHU cần ký agreement; YawDD archive có 322+29 video và chưa có nhãn event.
2. **Đề xuất thiết kế:** FPS, window, proxy thresholds, architecture, warning policy; phải kiểm chứng bằng experiment.
3. **Đã runtime/chưa đủ acceptance:** stack CPU Phase0 và CV Phase3–5 đã chạy; downloaded subset có official membership/provenance. Chưa tải đủ acquisition45, chưa human webcam/talking/disconnect acceptance, chưa trained-model metrics/false alarms hay GPU smoke. Không gọi throughput offline là FPS camera hoặc dùng metadata package thay lockfile đã tested.

UTA không có onset labels chính xác nên chưa đo được “cảnh báo sớm hơn ngủ gật bao nhiêu giây”. EAR proxy không phải phép đo PERCLOS80 sinh lý chuẩn. Kính râm che mắt và mất mặt phải hiện không đủ tin cậy, không giả vờ Alert. Demo chỉ trong điều kiện an toàn; không dùng thay hệ thống an toàn đã chứng nhận.

## Cách giao phase tiếp theo
> Implement Phase 6 according to 16_Development_Roadmap.md. Đọc coordinate/sign/reprojection convention ở06 và contract15; viết tests và chạy clip có consent. Không tự đổi feature schema hoặc triển khai toàn project.

User đã chốt tiếp tục với **34 video hiện có**, không chờ full-acquisition45. Sau QC Phase7, Phase8 có thể xử lý working snapshot đã freeze và Phase9 xây subject-independent splits/eligibility theo counts thực tế. Full-acquisition gate45 vẫn chưa đạt; không che missing/P1 calibration failure. Không cần dựng React/FastAPI, YOLO, Transformer hoặc tối ưu GPU trước khi CV/ML được kiểm chứng.
