# 11 — Calibration cá nhân

## Ba khái niệm khác nhau
- **Feature normalization:** đổi đơn vị/phân bố, ví dụ EAR/baseline hoặc StandardScaler train-only.
- **Personalized threshold:** ngưỡng eye_closed dựa trên baseline mắt từng người.
- **Model personalization:** cập nhật weights neural network cho người đó. **V1 không làm việc này**.

Paper UTA dùng phần đầu blink ở video Alert để normalize, loại phần này khỏi train/test: [section 4.2](https://arxiv.org/html/1904.07312). Project dùng calibration nhẹ 30 giây, **không tuyên bố tái hiện protocol calibration của paper**.

## Quy trình webcam
1. Người dùng dừng xe/ở nơi an toàn, xác nhận đang tỉnh, nhìn thẳng, không nói; UI hướng dẫn và cho hủy. Không yêu cầu calibration trong lúc lái.
2. Thu **30 giây wall-clock**, cần ít nhất **20 giây dữ liệu hợp lệ**; thiếu thì tiếp tục tối đa tổng 60 giây.
3. Bỏ mẫu no-face, blur/thiếu sáng mạnh, quay đầu vượt gate, chớp/nhắm mắt rõ, mở miệng bất thường. Khi chưa có baseline, lọc eye candidate bằng percentile: dùng 50% EAR cao nhất của mỗi mắt, rồi lấy median (gần percentile 75). Không chọn baseline từ EAR nhỏ chỉ vì đa số mắt đang nhắm.
4. `ear_left_baseline`, `ear_right_baseline`: median phần mắt mở đã chọn. `mar_baseline`: median MAR; pose baseline: median pitch/yaw/roll. EAR baseline phải >epsilon và nằm trong vùng hợp lý xác định từ train QC; stability MAD/median ≤0.15 cho từng mắt, mouth không mở kéo dài. Fail nếu kiểm tra không đạt.
5. Lưu baseline + camera resolution + asset/schema hash + quality stats + thời gian; freeze trong session. Không auto cập nhật baseline khi tài xế dần buồn ngủ.
6. Reset temporal history/buffer sau calibration, chờ 10 giây cho sequence. PERCLOS cần ≥30 giây lịch sử và coverage ≥80%; mask xử lý startup.

Normalization: `ear_norm=EAR/baseline`; `mar_delta=MAR−baseline`; pose trừ baseline và wrap góc. Không chia MAR cho số gần 0. Blink rate/duration có thể ghi trong profile nhưng **30 giây quá ngắn để dùng làm baseline tin cậy**, không normalize chúng mặc định.

Không thể chứng minh người đang Alert chỉ từ camera. Calibration khi đã buồn ngủ có thể làm baseline sai; UI giải thích và cho retry. Đổi góc camera, người hoặc ánh sáng nhiều → yêu cầu recalibrate, không sửa baseline âm thầm.

## Hai protocol training/evaluation phải tách
### P0 — Population-only
Không sử dụng clip calibration của người validation/test. Baseline EAR/MAR/pose chung tính từ **train subjects**; dùng cùng transform cho mọi người, `calibration_valid=0`. Thực nghiệm non-personalized vẫn có threshold population nhưng không phải threshold absolute duy nhất làm hệ thống chính.

### P1 — Calibration-assisted, mặc định demo
Mỗi người có calibration từ **30 giây đầu video Alert**, tối đa 60 giây nếu chưa đủ quality. Chọn nhãn Alert để lấy clip là điều kiện oracle của dataset: đã biết người có video Alert, không phải tự nhận biết Alert trên stream.

- Thực hiện cho train/val/test theo cùng policy; không dùng Low Vigilance/Drowsy để chọn baseline.
- Reserve **toàn prefix đã sử dụng** khỏi mọi window/metric của video Alert; temporal history sau prefix reset. Không dùng đoạn calibration làm labeled training sample.
- Baseline cá nhân người test được dùng transform, **không** fit global scaler/model/hyperparameter. Đây là **calibration-assisted evaluation**, không phải fully unseen/no prior data.
- Failure: người đó abstain trong P1, ghi tỷ lệ; có thể chạy P0 riêng nhưng không gộp metrics P1 bằng fallback tùy tiện.
- Lưu profile map theo subject cho offline; webcam dùng profile session do chính người dùng xác nhận Alert, không mang profile test vào training.

Với working snapshot34 hiện tại, **subject51 thiếu video Alert**, chỉ có Low Vigilance. P1 phải giữ missing-profile/abstention trong coverage; không lấy clip5 hoặc profile người khác làm baseline. Fold5 chỉ có subject51 nên P1 validation/test ở fold này không có accepted subjects: explicit blocked/undefined, không báo5 folds P1 đã đạt. Dùng dataset hiện tại không thay đổi mặc định P1; P0 nếu chạy phải là experiment/checkpoint/metrics riêng.

Train/export **hai checkpoint riêng** P0/P1; không đưa feature normalized kiểu P0 vào model chỉ train P1. Demo mặc định P1; nếu fail UI yêu cầu retry. Optional “không calibration” chỉ load đúng checkpoint P0, gắn nhãn chế độ rõ.

## Kiểm thử
Hai mắt có baseline khác nhau phải normalize đúng từng mắt; prefix không vào sequence; test-subject profile không ảnh hưởng global scaler; mất face không làm complete; timeout fail; EAR zero/MAR zero không chia lỗi; retry reset state; profile khác resolution/hash bị từ chối.
