# 06 — Feature engineering

“Feature” là đại lượng số rút ra từ hình ảnh. Tất cả threshold dưới đây là **giá trị khởi đầu cần tune trên validation**, không phải chuẩn sinh lý.

## 1. Landmark và EAR
Chọn sáu điểm quanh mỗi mắt theo topology MediaPipe, thứ tự `p1…p6`:
- Mắt trái **theo giải phẫu người trong ảnh**: `[362,385,387,263,373,380]`.
- Mắt phải: `[33,160,158,133,153,144]`.

Vẽ overlay để xác nhận trái/phải trên ảnh chưa mirror. Mapping là lựa chọn implementation, không phải mapping của paper dlib. Nguồn topology: [MediaPipe official mesh](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/data).

`EAR = (distance(p2,p6)+distance(p3,p5))/(2*distance(p1,p4))`.

Lưu `ear_left`, `ear_right`, `ear_mean=(left+right)/2`; mean chỉ hợp lệ khi cả hai mắt hợp lệ. Denominator ≤epsilon → invalid, không chia 0. EAR gần bất biến theo scale, **không bất biến theo pose/occlusion**. Nguồn công thức: [Soukupová & Čech 2016](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf).

### Blink và closure
Dùng EAR đã chia baseline mỗi mắt. Blink gate: cả hai ratio <0.65 để bắt đầu closed, >0.75 để mở lại; hysteresis tránh rung ngưỡng. Khi đang đóng giữ trạng thái cho đến ngưỡng mở. Đầu interval = timestamp sample đầu vượt ngưỡng, kết thúc = sample đầu mở; sai số thời gian phụ thuộc FPS.

- Event đã kết thúc, kéo dài 0.10–0.80 giây → blink candidate.
- >0.80 giây → prolonged closure, không cộng vào blink count; vẫn cộng closure duration.
- Bị mất landmark → event bị cắt/censored; không gọi là blink hoàn chỉnh, không nối qua gap.
- Lưu blink frequency/phút hợp lệ, mean blink duration, mean/max closed duration và closed elapsed hiện tại. Summary 60 giây; mẫu quá ít hiện insufficient, không kết luận rate “chuẩn”.

20 FPS tốt hơn 10 cho blink ngắn, nhưng nguồn FPS thấp vẫn có thể bỏ sót. Không nhân đôi frame để giả vờ tăng độ chính xác.

## 2. PERCLOS: định nghĩa và proxy của project
PERCLOS80 sinh lý là phần trăm thời gian mắt đóng **hơn 80%**, tương đương mở <20%. [Review Abe 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10108649/) cho thấy nghiên cứu khác nhau về cách đo và việc loại blink nhanh.

`EAR / baseline_EAR` **không đo chính xác tỷ lệ diện tích mí che con ngươi**. V1 dùng proxy `perclos_60` khi cả hai normalized EAR ≤0.20. Ghi trong UI tooltip/báo cáo: **“PERCLOS xấp xỉ từ EAR”**. Ngưỡng 0.20 phải kiểm tra trên subset mắt mở/đóng gán tay; không nhầm với EAR tuyệt đối 0.20 hoặc blink gate 0.65.

Công thức cho cửa sổ trailing `[t−60s,t]`:

`perclos_60 = Σ(closed_proxy_i × valid_eye_i × dt_i) / Σ(valid_eye_i × dt_i)`.

- Tỷ lệ lưu 0–1, UI nhân 100%. Cắt interval tại biên cửa sổ.
- Sample giữ đến sample tiếp theo nhưng tối đa **100 ms**; phần dài hơn chưa biết, không tính vào tử/mẫu. Sample cuối chỉ tính đến thời điểm hiện tại và cùng giới hạn.
- Mất face/eye, kính râm che mắt, pose vượt gate → thời gian unknown. **Không** xem unknown là mắt mở hoặc nhắm.
- `coverage = valid_eye_time / observed_span`; ready khi lịch sử ≥30 giây và coverage ≥0.80. Hiển thị chiều dài lịch sử thật trước đủ 60 giây; thiếu coverage → NaN + `perclos_ready=0`.
- Mặc định **bao gồm blink** trong proxy để tính online nhân quả, tránh giả vờ loại chính xác blink nhanh ở FPS thấp. Có thể thử bản loại event <250 ms nhưng phải báo cáo khác định nghĩa, độ trễ và FPS.
- Không thể kết luận riêng từ một tỷ lệ PERCLOS mà bỏ qua chất lượng tín hiệu.

## 3. MAR và ngáp
Chọn inner lip: chiều ngang `(78,308)`; ba cặp dọc `(82,87)`, `(13,14)`, `(312,317)`.

`MAR = (d82,87 + d13,14 + d312,317) / (3*d78,308)`.

Đây là **định nghĩa MAR của project**, không trộn threshold từ công thức MAR khác. Normalization dùng `mar_delta=MAR−median_closed_mouth`, không chia cho baseline gần 0.

Yawn candidate: mar_delta >0.35 mở event; <0.25 đóng; mouth opening kéo dài ≥2 giây được tính candidate. Lưu elapsed, duration và candidates/phút hợp lệ. Nói/cười/hát cũng mở miệng; duration và biên độ chỉ giảm nhầm, **không chứng minh true yawning**. UI ghi “ngáp nghi ngờ”. Nếu dùng YawDD, gán tay start/end ngáp để đo event precision/recall; chỉ train detector phụ khi có nhu cầu và nhãn đủ.

## 4. Head pose
Dùng `solvePnP` với 2D pixel landmarks `[1,152,33,263,61,291]` (mũi, cằm, hai góc mắt ngoài, hai góc miệng), 3D tương ứng từ [canonical face OBJ chính thức](https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj). OBJ vertex thứ `index+1`; không dùng normalized landmark z làm tọa độ metric.

Canonical dùng `D=diag(1,−1,−1)` (determinant +1): hệ object trung tính x phải ảnh, y xuống, z xa camera; hướng giải phẫu ra trước mặt là `(0,0,−1)`. Không dùng `diag(1,−1,1)` vì đó là reflection. OpenCV camera x phải, y xuống, z xa camera. Fit `SOLVEPNP_ITERATIVE`; `Rodrigues` thành R; frontal lý tưởng có R=I, không trừ offset180°. Euler hiển thị theo `R=Rz(roll) Ry(−yaw_display) Rx(pitch)`; đổi độ, wrap [−180,180). Pitch dương cúi xuống, yaw hiển thị dương quay về phải ảnh, roll dương theo chiều kim đồng hồ trên ảnh không mirror. Raw Euler yaw có dấu ngược yaw hiển thị. **Phải xác minh camera và synthetic projection riêng**, không suy QC người thật từ synthetic PASS.

K thật từ camera calibration là lựa chọn tốt nhất. Demo ban đầu có thể dùng `fx=fy=max(W,H), cx=W/2, cy=H/2, distortion=0` và ghi pose xấp xỉ. Nếu resize, scale K theo kích thước; không dùng K cũ. Reject khi solve fail hoặc reprojection error >0.03 đường chéo ảnh. Không coi góc tuyệt đối từ model mặt chung là đo y tế.

Feature: pitch/yaw/roll trừ median calibration; `pitch_velocity` độ/giây chỉ khi hai mẫu liên tiếp valid và gap ≤100 ms. Gật đầu là thay đổi pitch xuống rồi hồi; nhìn bên không đồng nghĩa buồn ngủ. Ban đầu để model học pose/velocity, không thêm nhãn “head nod” giả.

## 5. Chất lượng tín hiệu và feature bổ sung
- FaceLandmarkerResult không bảo đảm có confidence/visibility cho từng landmark. `min_*_confidence` là cấu hình, **không phải score trả về để lưu**.
- Tạo cờ quality của project từ: có face, điểm hữu hạn/in-frame, chiều rộng mắt, ánh sáng, blur, pose/reprojection và vùng mắt nhìn được. Không gọi các cờ này là xác suất tin cậy.
- Raw Phase7 chưa có calibration baseline: `quality.max_abs_yaw_deg=35`, `max_abs_pitch_deg=25`; vượt góc hoặc pose invalid → cả eye và mouth invalid, không cap EAR/MAR. Pose hình học valid nhưng oblique vẫn giữ góc để chẩn đoán. Poor light/blur/ROI invalid → mọi channel invalid. Các `max_yaw_delta_deg`/`max_pitch_delta_deg` dành cho tín hiệu calibrated tương lai, không ngầm baseline0. Kính râm có thể vẫn có landmark “đẹp”: `eyes_occluded` thủ công chỉ disable eyes; không hứa tự phát hiện kính râm.
- `eyeBlinkLeft/Right` blendshape là expression coefficient, không phải calibrated eye-closure probability. Chỉ thêm trong experiment, không mặc định.
- EAR_mean trùng thông tin hai mắt nên chỉ dùng UI/summary; derivatives chỉ giữ pitch_velocity trước, tránh feature phình to.
- Optical policy `raw_quality_v1`: bbox half-open floor-min/ceil-max từ IDs234/454/10/152; grayscale resize giữ aspect với max-side256, mean brightness và variance Laplacian CV_64F. Eye width là khoảng cách pixel hai horizontal endpoints chia bbox face width, không dùng vertical lid span nên mắt nhắm không tự invalid. Null ROI/metrics không bị biến thành0.
- `QualityGate` kiểm tra geometry AND policy; equality ở optical/góc boundary được nhận. Frozen config/report phải cùng numeric values và exact report SHA; nested path resolve từ project root. Profile hiện tại trong `configs/quality_policy_v1.json`:12 clean frames đã xem trực tiếp,36 degraded controls bị reject. Không có profile thì pipeline startup fail, không disable gate fallback.
- Đây là development envelope subject04, không universal/calibrated physiological truth. Full04_0 mouth chỉ2873/12291 valid dưới raw absolute gate; phải giữ missing và báo coverage theo source/lớp, không nới ngưỡng bằng nhãn test. Future calibrated deltas/velocity vẫn chưa triển khai.

Các cờ/missing value vào tensor theo [07](07_Temporal_Modeling.md); feature nào có ích do [18](18_Experiment_Plan.md) quyết định.
