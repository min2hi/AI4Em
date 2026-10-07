# Người 1 — Evaluation và experiments: kế hoạch thực hiện

> Dành cho coding agent: dùng skill executing-plans hoặc subagent-driven-development khi triển khai; làm từng task và lưu bằng chứng nghiệm thu. Tài liệu này giao việc, không xác nhận các phase đã hoàn thành.

**Mục tiêu:** đo chất lượng rules/RF/LSTM đáng tin cậy; thực hiện ablation; đánh giá external khi có dữ liệu hợp pháp.

**Kiến trúc:** evaluator dùng chung cho baseline và LSTM; metrics classifier dùng raw predictions. Realtime policy được đánh giá riêng, không trộn smoothing vào metrics classifier.

**Công nghệ:** Python 3.12, NumPy, scikit-learn, Parquet, plotting; dùng môi trường và lock file của repo.

## 1. Phạm vi và tài liệu phải đọc

Bạn sở hữu metrics core của Phase 11, Phase 14–15 và external evaluation của Phase 21. Không sở hữu trainer, realtime detector hoặc GUI.

| Tài liệu | Nội dung cần dùng |
|---|---|
| [Roadmap](16_Development_Roadmap.md) | Objective, prerequisites và acceptance của Phase 11, 14, 15, 21 |
| [Evaluation Strategy](10_Evaluation_Strategy.md) | Window/video/subject metrics, abstention, bootstrap, NTHU protocol |
| [Experiment Plan](18_Experiment_Plan.md) | Feature A–E, fair comparison và cột báo cáo |
| [Module Specification](15_Module_Specification.md) | ModelEvaluator, ModelBundle, class order và shared records |
| [Training Strategy](09_Training_Strategy.md) | Split, selection bằng validation, không tune test |
| [Personalized Calibration](11_Personalized_Calibration.md) | P0/P1, reserved prefix và failure coverage |
| [Dataset Analysis](04_Dataset_Analysis.md), [Dataset Access](Dataset_Access.md) | Memberships, weak labels, quyền truy cập external |
| [Project Structure](14_Project_Structure.md), [Testing Strategy](17_Testing_Strategy.md) | Output convention và behavioral tests |

## 2. Quyền sở hữu và phụ thuộc

| File | Việc làm |
|---|---|
| `src/evaluation/evaluator.py` | Metrics core Phase 11; mở rộng fold/video/subject Phase 14 |
| `scripts/evaluate.py`, `tests/test_evaluation.py` | CLI evaluation và regression |
| `scripts/run_ablation.py` | Điều phối experiment theo matrix đã freeze |
| `src/datasets/summaries.py` | Chỉ sửa subset selection Phase 15, phối hợp owner modeling |
| `scripts/external_test.py` | NTHU binary evaluation Phase 21 |
| `tests/test_integration.py` | External cases; chia section với người 2 |
| `configs/training.yaml` | Chỉ section experiment/subset; merge qua đầu mối config |

- Input cần nhận từ nhóm data/model: frozen snapshot, outer splits, reserved ranges/profile maps, model bundles/checkpoints, scaler, feature order, class order `[0,1,2]`, config/seed/hashes.
- Phase 14 cần Phases 11–13. Phase 15 cần Phase 14 và matrix đã freeze. Full Phase 21 cần Phases 14–20 và NTHU access nếu claim external complete.
- Có thể làm metrics core với fixture biết đáp án trước khi có model thật; đó chưa phải nghiệm thu Phase 14.
- Người 2 sở hữu `src/evaluation/replay.py`; dùng chung artifacts/provenance nhưng không viết replay thứ hai.
- Không tự sửa `src/contracts.py`, split, quality gate hoặc threshold. Thống nhất signature cụ thể của `evaluate(...)` với caller trước khi triển khai; `aggregate_folds(reports)` theo contract.

## 3. Checklist thực hiện

### Task 1 — Metrics core của Phase 11

- [ ] Viết regression đáp án độc lập: labels `[0,0,1,1,2,2]`, predictions `[0,1,1,1,2,0]` phải cho Accuracy `4/6`, recalls `[0.5,1,0.5]`, Macro F1 `59/90`, support mỗi lớp `2`.
- [ ] Chạy test trước sửa, quan sát fail đúng hành vi; triển khai core, chạy lại.
- [ ] Xuất Accuracy, Macro/Weighted F1, balanced accuracy, precision/recall/F1/support từng lớp, confusion matrix và coverage.
- [ ] Kiểm tra class order, absent-class undefined metrics, zero accepted support và abstention; không biến undefined thành 0 giả.
- [ ] Chạy baseline evaluation thật cùng owner Phase 11; đối chiếu một nhóm predictions với metrics độc lập.

### Task 2 — Phase 14: evaluation đầy đủ

- [ ] Dùng frozen model/splits; test không tham gia chọn checkpoint, threshold hoặc hyperparameter.
- [ ] Tách window-level raw predictions, video-level mean probability rồi argmax, subject-level metrics và chronological realtime policy reports.
- [ ] Video không đủ coverage là abstain, không tự gán Alert. Giữ eligible/accepted/rejected/blocked counts theo fold/lớp/người.
- [ ] Kiểm tra mỗi outer subject làm test đúng một lần; windows cùng người không rơi vào nhiều partition; prefix calibration không vào metrics.
- [ ] Chạy tất cả năm outer-fold slots và báo trạng thái từng fold. Chỉ tổng hợp metric đã định nghĩa, ghi số fold hợp lệ; mean/std và pooled confusion có denominator rõ.
- [ ] Bootstrap theo subject, không coi overlapping windows là mẫu độc lập.
- [ ] Xuất `runs/<id>/metrics.json`, `predictions.parquet`, `confusion_matrix.png`, command log và provenance; người duyệt chạy lại một fold.

**Giới hạn dataset hiện được mô tả trong docs:** working snapshot34; fold5 chỉ có Low Vigilance và subject51 thiếu Alert calibration. P1 fold này không có accepted subjects phải blocked/undefined. Kiểm tra snapshot thực dùng; không hard-code số cũ, không gọi kết quả snapshot thiếu lớp là full benchmark.

### Task 3 — Phase 15: ablation A–E

- [ ] Freeze matrix/config trước chạy: A EAR trái/phải + closure duration; B thêm PERCLOS; C thêm MAR delta + suspected-yawn duration; D thêm pose + pitch velocity; E dùng D với P1.
- [ ] Giữ masks theo nhóm, `perclos_ready` chỉ khi có PERCLOS, `calibration_valid` ở mọi variant. Mỗi variant có feature order/input size, scaler train-only và checkpoint riêng.
- [ ] Train lại LSTM từng variant; RF full là control bắt buộc. Không zero channels của full model rồi gọi là ablation.
- [ ] A–D cùng P0 cohort/splits/seeds/quality policy; E so D trên giao cohort hợp lệ P1, loại prefix tương ứng khỏi D. Báo full-cohort coverage riêng.
- [ ] Xuất `runs/ablation_<id>/results.csv`: experiment/model/features/mode/fold/seed/support/coverage, Macro F1, Weighted F1, Low precision/recall, Drowsy recall, video accuracy, p95 latency.
- [ ] Báo paired per-subject gain/loss và CI; kết luận giữ/bỏ feature từ evidence, không ép LSTM thắng RF.
- [ ] E03/E04/E06 là mở rộng theo tài liệu 18, không tự biến thành acceptance mới. Chưa chạy experiment phải ghi chưa thực hiện và lý do; không điền số dự kiến.

### Task 4 — External evaluation thuộc Phase 21

- [ ] Xác minh access/annotation/permissions, giữ official partitions; không tải/lưu dữ liệu trái điều kiện nguồn.
- [ ] Freeze checkpoint P0/scaler/policy từ UTA; threshold chọn UTA validation. Không dùng nhãn NTHU để calibration trong protocol mặc định.
- [ ] Dùng `p_risk=p_low+p_drowsy`, binary labels nguồn; báo binary Macro F1/Recall/PR-AUC và coverage theo scenario accessible.
- [ ] Không tạo lớp Low Vigilance từ NTHU hoặc báo Low Vigilance Recall cho external binary task.
- [ ] Ghi modality/domain shift và scenario thiếu; missing access không thay bằng synthetic external score.

## 4. Cách người giao việc kiểm tra đã xong

1. Yêu cầu lệnh đầy đủ với input/model/config thật, chạy lại một fold từ cold load.
2. Mở predictions và đối chiếu fixture metrics biết đáp án; chọn một video, kiểm tra mean probabilities/support/abstention so với report.
3. Mở per-fold report: đủ membership, không leakage; fold thiếu lớp/profile ghi rõ; coverage không biến mất khỏi bảng F1.
4. Mở ablation configs/checkpoints: feature order khác đúng variant, scaler/checkpoint riêng và cùng comparison cohort.
5. External: kiểm tra nguồn nhãn/quyền truy cập, mode P0, binary mapping và per-scenario support trước đọc score.

Lệnh suite hiện có:

```powershell
.venv/Scripts/python.exe -m pytest -q --tb=short
```

Lệnh scoped **sau khi triển khai file test**:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_evaluation.py -q --tb=short
```

CLI `evaluate`, `run_ablation`, `external_test` là deliverable tương lai: phải có `--help`, báo lỗi config/schema rõ, và bàn giao command thực với đủ arguments. Không coi tên script trong roadmap là lệnh đã hoạt động.

## 5. Gói bàn giao và trạng thái

Mỗi run phải ghi phiên bản code, snapshot/split/model/config/seed/hashes, lệnh chạy chính xác, tests, metrics/artifacts, acceptance checklist với đường dẫn bằng chứng và giới hạn. Bàn giao các lệnh reproduce vào README liên quan; cập nhật changelog đúng phạm vi, không sửa phần của người khác.

- **DONE:** toàn bộ acceptance của phase có evidence thật, reviewer chạy lại smoke được.
- **IMPLEMENTED — CHƯA NGHIỆM THU:** core/tests có, nhưng thiếu model/data/run thật; không gọi phase hoàn tất.
- **BLOCKED:** nêu prerequisite thiếu và việc đã làm. External blocked không làm mất kết quả UTA, nhưng không claim full Phase 21 generalization complete.

Liên quan: [Người 2 — Realtime core](Team_02_Calibration_Realtime.md), [Người 3 — Alerts/worker](Team_03_Alerts_UI.md), [Người 4 — Desktop UI/startup](Team_04_Desktop_UI.md). Phân công mới: người 3 sở hữu warning và CameraWorker; người 4 sở hữu MainWindow, Qt audio playback và main.py. Phạm vi evaluation của người 1 không thay đổi.
