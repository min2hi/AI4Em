# 09 — Chiến lược training và tái lập

## Thứ tự bắt buộc
1. Download, kiểm tra quyền và manifest.
2. Landmark/feature thử trên vài clip mỗi lớp; kiểm tra overlay, missing và timestamps.
3. Trích raw feature toàn **working snapshot đã freeze** (hiện34 video), lưu Parquet; video fail nằm trong report. Không đợi45/180 nhưng không gọi34 là full dataset.
4. Gán split theo subject trước khi fit scaler, tuning hoặc tạo sequence index.
5. Khóa protocol calibration ở [11](11_Personalized_Calibration.md); reserve đoạn calibration, không tính metrics trên đoạn này.
6. Tạo sequence và RF summary cùng quy tắc quality.
7. Train RF trước LSTM; debug một fold trước khi chạy 5 folds.

## Working-set split policy
User chốt dùng **34 video/12 subjects** hiện có. Freeze manifest membership/source hashes trước extraction/splits; giữ tất cả source rows, missing/rejection và protocol eligibility riêng. Official folds1–5 có3/3/3/2/1 subjects; testfold k, valfold kế tiếp theo vòng, ba fold còn lại train. Không frame/window split và không ép9/3/3.

Trước profile/QC, số subject train/val/test cho testfold1..5 là **6/3/3, 6/3/3, 7/2/3, 9/1/2, 8/3/1**. Đây là counts của snapshot, **không guarantee usable P1 counts**. Subject51 không có Alert: P1 abstain/missing-profile; fold5 không còn accepted subject P1, fold dùng fold5 làm validation không thể tune P1. Mark fold blocked/undefined, không tạo empty metrics hay báo5 valid folds. P0 là experiment/checkpoint riêng theo11, không tự chuyển mode mặc định để che failure.


## LSTM training ban đầu
| Parameter | Mặc định |
|---|---|
| seed | 42; final repeat 42/43/44 nếu có điều kiện |
| batch_size | 64 |
| optimizer | Adam, learning_rate=0.001, weight_decay=0.0001 |
| max_epochs | 50 |
| early_stopping | patience=8, chọn validation Macro F1 |
| gradient_clip_norm | 1.0 |
| loss | CrossEntropyLoss |
| scheduler | Chưa dùng; chỉ thêm khi loss trace cho thấy cần |

Shuffle **window trong train loader** được phép, không shuffle timestep trong window chính. Validation/test không random sampling. Chọn `num_workers=0` trước trên Windows; tăng sau khi đo, entrypoint có main guard. Không thêm multiprocessing trước smoke dataset.

Snapshot hiện tại có video class counts **11/12/11**, không cân bằng tuyệt đối; window hợp lệ có thể lệch thêm. Tính class weight `N/(3*N_c)` từ **train windows** nếu cần; nếu train thiếu class phải block model ba lớp, không chia cho0/tạo sample giả. Không vừa oversample vừa class weight mà chưa so sánh. Report người/video/window và rejected ratio theo lớp; nhiều window không tương đương nhiều subject độc lập.

Không augment raw face trong training feature model. Augmentation feature chỉ thêm khi có plausible noise study và giữ mask/timestamps đúng; không tạo sample giả vượt sang test.

## Tuning và test isolation
- Grid RF nhỏ; LSTM chỉ thử hidden `{32,64}`, dropout `{0.2,0.3}` trên validation, không tìm kiếm quá lớn.
- Mỗi outer fold có scaler/model riêng. Không dùng checkpoint tốt nhất trên test để lựa chọn kiến trúc.
- Sau lựa chọn config có thể refit trên các subject của train+validation trong snapshot với số epoch chốt từ inner validation; scaler refit train-only. Default giữ train/validation tách biệt. Số48/36/12 chỉ là full60 reference, không là cardinality của development snapshot34.
- Test chỉ chạy sau config frozen. Nếu sau khi xem test cần sửa thiết kế, đánh dấu exploratory và cần holdout mới để kết luận.

## Reproducibility
Seed Python, NumPy, PyTorch, CUDA. Bật deterministic algorithms nếu được hỗ trợ; log operation không deterministic và version driver/CUDA/cuDNN. Không hứa bitwise identical giữa các máy/GPU.

Experiment ID: `YYYYMMDD_model_feature-set_cal-mode_outer-k_seed42`. Mỗi `runs/<id>/` gồm:
- `config_resolved.yaml`, `environment.txt`, `split.json`, `dataset_manifest_hash.txt`;
- `feature_names.json`, `scaler.json`, `checkpoint.pt` hoặc `model.joblib`;
- `history.csv`, `metrics.json`, `predictions.parquet`, `confusion_matrix.png`;
- `errors.jsonl`, calibration/rejection coverage và runtime device.

TensorBoard dùng cho loss/F1 theo epoch; không cần MLflow/W&B. Checkpoint chỉ load từ nguồn tin cậy; lưu state_dict + metadata, không nhận pickle/model lạ từ Internet.

Train scaler không dùng calibration/test làm global fit; quyền dùng calibration người test chỉ ở biến đổi cá nhân đã khai báo. Chi tiết [11](11_Personalized_Calibration.md).
