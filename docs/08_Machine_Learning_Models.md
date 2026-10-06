# 08 — Model và baseline

## Baseline 0: rule-based
Hai bảng kết quả khác nhau:
- **EAR-only baseline:** mean EAR tuyệt đối <0.20 → closed; closed ≥1 giây → Drowsy. Đây là baseline cố ý đơn giản, không dùng làm hệ thống chính.
- **Temporal rule ba lớp:** normalized EAR blink gate theo [06](06_Feature_Engineering.md); Drowsy nếu closure_elapsed ≥1.5 giây hoặc perclos_60 ≥0.25; Low Vigilance nếu không Drowsy và perclos_60 ≥0.15; còn lại Alert. PERCLOS phải ready; nếu chưa ready và không closure kéo dài thì abstain, không tự Alert.

Ngưỡng 0.15/0.25/1.5 là đề xuất. Tune trên validation, freeze trước test. EAR-only không thực sự học Low Vigilance: report recall lớp này và hạn chế, không che đi bằng binary Accuracy. So sánh cùng timestamp, cùng quality gate; báo cáo coverage riêng khi rule abstain.

## Baseline 1: Random Forest
Input là **thống kê từ cùng window 10 giây** của LSTM:
- Với EAR hai mắt, MAR, pitch/yaw/roll và pitch_velocity: mean, std, min, max, slope (OLS theo giây); chỉ dùng mẫu valid.
- PERCLOS cuối window, closure maximum/mean duration, blink count/rate/mean duration, yawn candidate count/rate/duration; event lấy từ cùng lịch sử nhân quả và metadata ready.
- Missing/valid duration ratio, perclos_ready, calibration_valid. Impute summary thiếu bằng median fit train-only, kèm cờ missing theo cột.

Mặc định `n_estimators=300, max_depth=12, min_samples_leaf=5, class_weight=balanced, random_state=42`; giới hạn `n_jobs` để không làm máy treo. Grid nhỏ depth `{8,12,None}` và leaf `{3,5,10}`. Không dùng ID người làm feature.

**XGBoost tùy chọn:** sau RF ổn, thử `objective=multi:softprob`, depth 3, 300 cây, learning_rate 0.05, subsample/colsample 0.8; sample weight theo train, early stopping theo validation. Không bắt buộc thêm LightGBM hoặc cả hai boosting library.

## Model chính: LSTM PyTorch
```text
[B,100,16] → LSTM(input_size=16, hidden_size=64,
                    num_layers=1, batch_first=True, bidirectional=False)
           → hidden cuối [B,64]
           → Linear(64,32) → ReLU → Dropout(0.3)
           → Linear(32,3) → logits [B,3]
```

- LSTM nội bộ `dropout=0` vì chỉ một layer. Head dropout vẫn 0.3.
- Không stateful hidden qua các cửa sổ; reset hidden mỗi forward. Dễ train/replay và tránh nối nhầm session.
- `CrossEntropyLoss` nhận logits; softmax chỉ lúc inference/đo probability. Không softmax hai lần.
- Các ablation có ít kênh hơn phải derive `input_size=len(feature_names)` và retrain; 16 là full feature model.
- Chọn CPU inference trước; tensor nhỏ có thể chậm hơn trên GPU vì chi phí copy. Train dùng CUDA khi driver/wheel đã smoke.

Nguồn API: [torch.nn.LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html). Nguồn động lực temporal blink: [UTA paper](https://arxiv.org/html/1904.07312); architecture trên là **đề xuất đồ án**, không lấy nguyên HM-LSTM paper.

## LSTM có cần thiết hơn RF không?
Chưa biết trước experiment. RF đã dùng thông tin thời gian qua summary; LSTM có lợi tiềm năng khi thứ tự và diễn biến mang thêm thông tin. Chạy ablation **shuffle thứ tự timestep** với inference diagnostic và model retrain control để xem thứ tự có giá trị hay chỉ phân bố. Giữ RF nếu LSTM không cải thiện Macro F1/Low Vigilance recall đủ ổn định hoặc latency không phù hợp.

Alternative: GRU hidden 64 giảm tham số; TCN nhỏ cho convolution thời gian. Không triển khai CNN-LSTM, Transformer hoặc raw-video CNN trong v1 nếu chưa có bằng chứng baseline thiếu khả năng.
