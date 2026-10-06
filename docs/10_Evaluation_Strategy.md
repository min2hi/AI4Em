# 10 — Đánh giá model và realtime

## Metrics classification
- Accuracy; **Macro F1 là metric chọn model**; Weighted F1; balanced accuracy.
- Precision/Recall/F1/support **mỗi class**, đặc biệt Low Vigilance và Drowsy.
- Confusion matrix đếm tuyệt đối và normalized theo nhãn thật. Chú ý Low Vigilance → Alert (bỏ sót sớm), Alert → Drowsy (cảnh báo giả).
- Optional: OvR ROC-AUC, macro ROC-AUC; per-class PR-AUC/average precision hữu ích khi lệch lớp. Nếu fold không có positive/negative của lớp thì ghi undefined, không gán 0 rồi trung bình.
- Probability có thể chưa calibrated: reliability diagram/ECE nếu dùng threshold; temperature scaling chỉ fit validation, không fit test.

## Đơn vị báo cáo
1. **Window-level:** stride 1 giây, nhãn yếu từ video; metrics trên prediction chưa smooth.
2. **Video-level:** trung bình probability các window hợp lệ rồi argmax; video không đủ coverage là abstain, không tự Alert. Báo cáo riêng để gần cách benchmark theo video, không nhập chung window accuracy.
3. **Subject-level:** trung bình metric theo người và cluster bootstrap theo subject (ví dụ 1,000 lần) cho CI; không bootstrap các window overlap như mẫu độc lập.
4. **Realtime replay:** smoother/alert trên stream chronological; thêm coverage, alarm count và latency.

Với full60 benchmark, báo cáo trung bình±std5 outer folds và out-of-fold confusion matrix. **Development snapshot34** giữ per-fold support/eligible/abstained/blocked; chỉ tổng hợp metrics đã định nghĩa, ghi rõ số fold hợp lệ và phạm vi, không gọi full-five-fold result nếu fold thiếu class hoặc P1 không có accepted subjects. Fold5 hiện chỉ có classLow Vigilance, thiếu Alert/Drowsy; missing class metrics phải ghi undefined/support0 theo loại metric, không tạo observations/score0 giả để tính trung bình. Giữ seed và per-fold report. Window cùng người/video tương quan mạnh; overlap90% không tăng số người độc lập.

## Không che failure bằng loại dữ liệu
- `coverage = số decision timestamps hợp lệ / tổng scheduled decision timestamps` sau thời gian khởi động đã khai báo.
- Báo cáo missing/rejected/no-face theo lớp, người, điều kiện. Kính râm có coverage thấp phải ghi, không chỉ đưa F1 trên vài mẫu còn nhìn thấy mắt.
- Metrics classification trên accepted windows là **conditional performance**; thêm số true risky timestamps bị abstain. Không tuyên bố an toàn end-to-end chỉ bằng conditional F1.
- Với baseline abstain startup khác LSTM, dùng một tập timestamp giao nhau cho bảng so sánh trực tiếp và thêm coverage toàn tập từng model.

## Low Vigilance và cảnh báo sớm
UTA không có transition ground truth. Đánh giá Low Vigilance recall, precision, F1 và lỗi sang hai lớp còn lại. Muốn đo time-to-warning cần subset annotation onset riêng, ít nhất hai người gán độc lập và quy tắc adjudication; không lấy lúc EAR qua ngưỡng làm nhãn “buồn ngủ thật”.

## External test NTHU
Freeze model/scaler/calibration policy/threshold từ UTA. Giữ official partitions và chỉ dùng phần có annotation truy cập hợp pháp. Mặc định **không calibration lấy nhãn Alert NTHU**: dùng checkpoint population-only, đo domain shift IR→RGB landmark pipeline. Có thể thử calibration hỗ trợ trên clip normal riêng nhưng báo cáo là protocol khác, không âm thầm trộn.

Dùng `p_risk=p_low+p_drowsy`, binary labels nguồn; threshold chọn UTA validation. Report binary Macro F1/Recall/PR-AUC theo năm scenario, **không report Low Vigilance Recall trên NTHU**. Eye/mouth/head annotations nếu có chỉ dùng đánh giá event tương ứng, không đổi thành ba class.

## Hiệu năng và cảnh báo giả
Chạy 10 phút webcam và replay trên máy demo; ghi CPU/GPU/RAM/VRAM, resolution, nguồn FPS, landmark throughput, dropped-frame rate, prediction cadence và p50/p95:
- decode/capture, landmark, feature, model forward, smoothing, UI;
- capture-to-result latency; model forward latency riêng;
- cold start/calibration/warmup riêng.

Target dự án: landmark ≥15 FPS ánh sáng tốt, model forward p95 ≤20 ms, prediction mỗi 1 giây, capture-to-feature p95 ≤150 ms, UI không đứng. Đây là **acceptance target**, không benchmark đã đạt. Dùng `time.perf_counter`, `psutil`; CUDA đo phải synchronize; không dựa utilization screenshot duy nhất.

False alarms/hour: episode Drowsy/strong warning trên đoạn Alert được kiểm chứng, tính theo thời gian quan sát hợp lệ; thêm unknown time và Low Vigilance UI alarm riêng. Target demo ban đầu ≤2 audio false alarms/giờ, cần đủ ít nhất 1 giờ Alert có consent để đánh giá; clip 10 phút không đủ xác nhận target đó. Không cố tình gây buồn ngủ khi lái xe thật.
