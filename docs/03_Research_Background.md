# 03 — Nghiên cứu và lựa chọn kỹ thuật

Thông tin online kiểm tra ngày **06/10/2026**. Nguồn đọc được và stack chạy được là hai loại bằng chứng khác nhau; Phase 0 hiện đã smoke stack trên Windows/Python 3.12.10.

## Bằng chứng chính
| Nguồn | Nội dung dùng cho thiết kế | Không được suy diễn |
|---|---|---|
| [Ghoddoosian et al., CVPR Workshops 2019](https://arxiv.org/html/1904.07312) | UTA-RLDD, đặc trưng blink, HM-LSTM, calibration trên dữ liệu Alert | LSTM đơn giản của đồ án không phải tái hiện HM-LSTM; accuracy paper không là target của ta |
| [Soukupová & Čech, CVWW 2016](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf) | Công thức EAR, blink cần temporal information | Paper dùng landmark khác MediaPipe; không sao chép ngưỡng trực tiếp |
| [Abe, Sleep Advances 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10108649/) | PERCLOS, khác biệt định nghĩa, cần kết hợp tín hiệu | PERCLOS không luôn nhạy với buồn ngủ mức vừa |
| [Weng et al., ACCV Workshop 2016 / NTHU](http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/) | Generalization theo kính/đêm/biểu hiện mô phỏng | Nhãn nhị phân không tương đương ba lớp UTA |
| [Abtahi et al., ACM MMSys 2014](https://doi.org/10.1145/2557642.2563678), [YawDD archive](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset) | Talking/singing/yawning để kiểm tra miệng | Video không có annotation ngáp sẵn theo trang archive |

Đây là nền tảng phù hợp đồ án; không tuyên bố đã khảo sát toàn bộ state-of-the-art. Transformer/video foundation model tăng chi phí và khó giải thích, chưa có nhu cầu rõ để dùng.

## Compatibility và API hiện tại
**Chọn Python 3.12 x64 + venv**. So với 3.11, 3.12 phù hợp cả metadata hiện tại của NumPy/XGBoost; không chọn Python mới nhất chỉ vì mới.

Metadata đọc trực tiếp từ PyPI JSON:
| Package | Version quan sát | Requires-Python | Wheel Windows x64 |
|---|---|---|---|
| torch | 2.14.1 | ≥3.10 | cp312 |
| mediapipe | 1.0.1 | Không khai báo trong metadata | py3 |
| opencv-contrib-python | 5.0.0.93 | ≥3.6 | cp37 abi3 |
| PySide6 | 6.11.2 | ≥3.10, <3.15 | cp310 abi3 |
| numpy | 2.5.3 | ≥3.12 | cp312 |
| pandas | 3.0.6 | ≥3.11 | cp312 |
| scikit-learn | 1.9.1 | ≥3.11 | cp312 |
| pyarrow | 25.0.1 | ≥3.10 | cp312 |
| xgboost — tùy chọn | 3.4.1 | ≥3.12 | py3 |

Nguồn metadata: `https://pypi.org/pypi/<package>/json`, ví dụ [MediaPipe](https://pypi.org/pypi/mediapipe/json), [torch](https://pypi.org/pypi/torch/json), [OpenCV contrib](https://pypi.org/pypi/opencv-contrib-python/json), [PySide6](https://pypi.org/pypi/PySide6/json), [NumPy](https://pypi.org/pypi/numpy/json), [pandas](https://pypi.org/pypi/pandas/json), [scikit-learn](https://pypi.org/pypi/scikit-learn/json), [PyArrow](https://pypi.org/pypi/pyarrow/json), [XGBoost](https://pypi.org/pypi/xgboost/json).

Các package bắt buộc trong bảng đã được cài và smoke ở Phase 0; torch thực tế là **2.14.1+cpu**, chỉ có một OpenCV distribution `opencv-contrib-python`. `pip check` không có dependency lỗi; Tasks VIDEO trả 478 landmarks trên hai frame UTA subject 04; LSTM random forward `[1,100,16] → [1,3]`; cửa sổ Qt thật mở/đóng và lưu screenshot. Chi tiết: `runs/phase0/environment_report.json`, `config_resolved.json`, `qt_smoke.png` và `requirements-lock.txt`. XGBoost tùy chọn chưa cài/kiểm thử. CUDA training chưa kiểm thử; bản CPU tiết kiệm dung lượng, không phải model đã train.

### Những điểm dễ nhầm
- [MediaPipe setup hiện tại](https://ai.google.dev/edge/mediapipe/solutions/setup_python) ghi Python **3.9 trở lên**, không còn ghi cứng trần 3.12. Wheel thực tế và smoke test mới quyết định cài được.
- [Face Landmarker Tasks](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python): `.task` asset, 478 landmark, tùy chọn 52 blendshape; `detect_for_video` nhận timestamp ms. Không dùng API `mp.solutions.face_mesh` cũ cho thiết kế này.
- `LIVE_STREAM/detect_async` có thể bỏ frame khi bận. Chọn **VIDEO mode trong worker riêng** cho cả file và webcam để luồng xử lý dễ kiểm soát. VIDEO không có nghĩa chỉ nhận video file.
- Smoothing landmark tích hợp chỉ áp dụng khi `num_faces=1`; không thay thế smoothing probability của model.
- MediaPipe phụ thuộc `opencv-contrib-python`: chỉ cài **một** package cung cấp `cv2`; không cài đồng thời `opencv-python`, contrib và headless. Bản OpenCV quan sát yêu cầu NumPy ≥2 trên Python ≥3.9.
- [PyTorch installer](https://pytorch.org/get-started/locally/): chọn Windows/Pip/Python và CUDA phù hợp NVIDIA driver; lưu lệnh cài. RTX không bảo đảm mọi CUDA wheel tương thích. MediaPipe CPU và PyTorch CUDA là hai việc riêng.
- [PySide6](https://doc.qt.io/qtforpython-6/gettingstarted.html) yêu cầu Python ≥3.10 và có Qt trong wheel. Dùng Widgets; không cần cài Qt hệ thống.
- [LSTM reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html): dropout bên trong LSTM có tác dụng giữa các layer; với 1 layer đặt 0 và dùng dropout ở head.
- [OpenCV solvePnP reference](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html): endpoint bị HTTP 403 khi đọc; API phải smoke tại Phase 6. Đây là reference cần kiểm tra, không ghi là đã đọc thành công.

## Mặc định và alternative
| Quyết định | Chọn | Vì sao / khi đổi |
|---|---|---|
| CSV vs Parquet | Parquet, CSV chỉ export nhỏ | Giữ dtype/null, đọc theo cột, dung lượng tốt; thêm PyArrow là chi phí chấp nhận được |
| RF vs XGBoost | RF bắt buộc; XGBoost tùy chọn | RF ít tuning; chỉ thêm XGBoost sau baseline ổn |
| LSTM vs GRU/TCN | LSTM 1 layer | Dễ giải thích và phù hợp chuỗi feature; GRU/TCN là experiment mở rộng, không prerequisite |
| 10 vs 30 FPS | Landmark tối đa 20; model 10 | Model không cần nhận mọi frame; blink ngắn cần tốc độ cao hơn 10 |
| 5 vs 10 giây | 10 giây, thử 5/20 | Cân bằng diễn biến và độ trễ; không khẳng định tối ưu |
| PySide6 vs Streamlit | PySide6 Widgets | Đúng desktop, worker và audio; không dựng server web |

GitHub chính thức MediaPipe chỉ là nguồn implementation/topology; quyết định sinh lý lấy từ paper, không từ repository random.
