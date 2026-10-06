# 07 — Tạo chuỗi theo thời gian

## Mặc định thống nhất
| Parameter | Giá trị | Ý nghĩa |
|---|---|---|
| landmark_target_fps | 20 | Không vượt FPS thật nguồn |
| sequence_fps | 10 | Một timestep mỗi 100 ms |
| sequence_window_s | 10 | 100 timesteps |
| stride_s | 1 | Overlap 90%; prediction một lần/giây |
| statistics_window_s | 60 | PERCLOS và event rate, độc lập chuỗi |
| max_sample_age_ms | 100 | Không kéo mẫu cũ qua đoạn thiếu |
| max_missing_ratio | 0.20 | Tối đa 20% timestep không đủ eye/mouth/pose |
| max_gap_s | 1 | Gap lớn hơn → chia segment/reset buffer |

Mọi feature đều **nhân quả**: ở thời điểm t chỉ dùng dữ liệu ≤t. Không dùng centered filter hoặc nội suy bằng frame tương lai cho realtime.

## Feature order chuẩn — 100 × 16
10 kênh liên tục, theo thứ tự:
1. `ear_left_norm`
2. `ear_right_norm`
3. `mar_delta`
4. `pitch_delta`
5. `yaw_delta`
6. `roll_delta`
7. `perclos_60`
8. `closure_elapsed_s`
9. `yawn_elapsed_s`
10. `pitch_velocity_dps`

6 kênh binary cuối: `left_eye_valid`, `right_eye_valid`, `mouth_valid`, `pose_valid`, `perclos_ready`, `calibration_valid`.

`closure_elapsed_s` cần cả hai mắt valid; `yawn_elapsed_s` cần mouth_valid. Derivative invalid nếu pose gap, impute 0 như mô tả dưới; mask pose_valid của timestep cho model đặt 0 khi derivative chưa tính được. PERCLOS chưa ready được phân biệt bởi kênh riêng. Calibration_valid=0 trong protocol không personalization; đó là chế độ đã được train riêng, không tự dùng checkpoint calibrated.

Không đưa subject_id, label, video_id, fold, timestamp tuyệt đối hoặc file path vào X. EAR_mean, blink summary và yawn frequency nằm ở báo cáo/RF summary, không phải kênh LSTM v1.

## Resample và missing
1. Tạo lưới `t0 + k*100 ms` theo từng segment; t0 là sample đầu segment.
2. Ở mỗi điểm lấy sample mới nhất **không muộn hơn điểm lưới**, tuổi ≤100 ms; không average qua cửa sổ có blink vì có thể làm mất peak.
3. Không có sample hợp lệ → NaN và mask 0 theo nhóm. Không forward-fill vô hạn, không tăng event count do lặp sample.
4. Scale kênh liên tục bằng scaler train-only; impute các giá trị invalid bằng **0 sau scaling**. Binary mask không scale. Không dùng nội suy future/backfill.
5. Reject window nếu missing ratio >20%, có gap >1 giây, hoặc timestep cuối thiếu bất kỳ eye/mouth/pose. `perclos_ready=0` không tự loại window: model học startup bằng mask này.
6. Chỉ nhận window đủ 100 bước; không pad thiếu để tự nhận Alert. Video/session thay đổi → reset lịch sử, không nối video của cùng người.

Missing ratio: timestep bị thiếu nếu bất kỳ left_eye/right_eye/mouth/pose mask bằng 0. Derivative đầu segment không có tiền sử cũng tính thiếu pose tại bước đó. Calibration thất bại không impute để chạy calibrated checkpoint; chuyển trạng thái hệ thống.

## Nhãn sequence
UTA: kế thừa nhãn video, ghi `label_source=video_weak`. Một sequence có thể thực tế không khớp nhãn chủ đạo; không gán lại bằng EAR rule rồi train model để “chứng minh” chính rule đó.

Dataset có annotation thời gian: window gán majority **theo thời lượng**, chỉ chấp nhận ≥80% duration cùng nhãn; loại đoạn chuyển trạng thái và báo cáo số loại. NTHU chỉ dùng external mapping nhị phân, không tạo Low Vigilance từ eye state.

## Sequence index và scaler
- Parquet raw theo video; index lưu `video_id, segment_id, end_timestamp_ms, start_row, end_row, label_id, split`.
- Dataset đọc mảng feature cache mỗi video, cắt cửa sổ; không ghi hàng triệu file `.npy` chồng lặp. Có thể dùng `.npz` per video nếu Parquet trở thành bottleneck, không là mặc định.
- Fit scaler trên timestep hợp lệ **không trùng** của train, sau personalized transform nếu protocol dùng calibration. Không fit trên từng window lặp 90% overlap.
- Dùng cùng scaler cho validation/test/webcam; order và schema lưu checkpoint.

## Window length và độ trễ
10 giây là lựa chọn thiết kế, không phải kết luận khoa học. Thử 5/10/20 giây trên cùng split; PERCLOS giữ 60 giây. Warmup 10 giây sau calibration cộng smoothing có thể gây trễ. Báo cáo riêng thời gian chờ khởi động, latency tính toán và độ trễ quyết định; không gọi tất cả là “inference latency”.
