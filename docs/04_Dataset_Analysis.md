# 04 — Phân tích dataset và quyền sử dụng

## 1. UTA-RLDD — dataset chính
Nguồn đã đọc: [trang tác giả](https://sites.google.com/view/utarldd/home), [paper 2019](https://arxiv.org/html/1904.07312).

- **60 người, 180 video RGB, khoảng 30 giờ, 111.3 GB**; mỗi người một video mỗi lớp, khoảng 10 phút/video.
- 51 nam, 9 nữ; tuổi 20–59. Kính xuất hiện ở 21/180 video. Camera cá nhân, độ phân giải khác nhau, FPS dưới 30.
- Ba nhãn gốc `0=Alert, 5=Low Vigilant, 10=Drowsy`. Nhãn do người tham gia cung cấp theo trạng thái chủ đạo; không phải nhãn frame chính xác.
- Quay trong nhà, không phải chạy xe ngoài đường. Cần external test và demo riêng để kiểm tra domain shift.
- Trang hiện tại có [Google Drive trực tiếp](https://drive.google.com/drive/folders/1d_QwgpMXnLY_FmLYXDn7TLcw0D-svXEl); không thấy request form. Đã đọc index của **10 ZIP chính thức** bằng HTTP Range/ZIP64; downloader chỉ lấy các thành viên đã chọn, không tải toàn archive.
- Trang yêu cầu trích dẫn paper; chỉ 36/60 người cho phép công bố ảnh, phải đối chiếu bảng “Image Publishable”. Không thấy giấy phép mở kiểu MIT/CC cho raw data; không tự kết luận được thương mại hoặc tái phân phối.

### Official protocol và protocol của project
Tác giả chia **5 folds × 12 người**; khuyến nghị mỗi lượt 1 fold test, 4 fold train, lấy trung bình 5 lượt. **Không tự tạo lại danh sách folds**: lấy từ gói chính thức, lưu manifest subject IDs. Nếu chưa tìm thấy membership, liên hệ tác giả; split tự tạo phải ghi rõ custom, không gọi official.

Với **full60 benchmark**, project bổ sung validation bằng fold kế tiếp theo vòng: test fold k, validation fold kế tiếp, ba fold còn lại train (36/12/12 người nếu đủ nguồn). Đây không phải nguyên văn protocol tác giả. Tune bằng validation; outer test không chọn model.

**Development hiện tại theo quyết định user:** dùng34 video/12subjects với fold counts3/3/3/2/1, không chờ đủ15 hoặc60. Giữ membership official nhưng gọi protocol **restricted development subset**, không official full benchmark. Cùng quy tắc một fold test/một fold val/ba fold train; số người/video lấy từ snapshot và eligibility thực tế, không ép36/12/12 hoặc9/3/3. Fold5 hiện chỉ subject51/classLow Vigilance: phải ghi thiếu class và P1 missing Alert/abstention, không bù bằng frame split hoặc calibration từ clip5. Chi tiết [09](09_Training_Strategy.md), [10](10_Evaluation_Strategy.md), [11](11_Personalized_Calibration.md).

### Phạm vi triển khai Phase 1–2 theo giới hạn ổ đĩa
Chỉ chọn **15 người × 3 trạng thái = 45 file**, ba người từ mỗi official fold, tổng raw dự kiến **13,968,447,679 bytes (~13.01 GiB)**. Chọn subject đầy đủ nhỏ nhất theo dung lượng trong từng fold, không ngẫu nhiên; đây là development subset thiên lệch, không full benchmark. Official ZIP index thực tế có **182 file vật lý / 180 subject-state recordings**: subject 32 và 49 có Drowsy tách `10_1`/`10_2`; parser giữ `part_id`, không nhầm dấu `_` thành nhãn khác. Subset chọn không gồm recording tách.

Fold IDs lấy từ tên archive/member chính thức, không suy theo thứ tự subject. Chưa gán train/val/test ở Phase 2 (`split="unassigned"`); Phase 9 phải giữ subject-independent split và gọi kết quả đúng là subset, không so sánh như đã chạy đủ 60 người. NTHU/YawDD chưa tải theo phạm vi hiện tại. Nguồn, subject IDs và quyền ảnh: [Dataset Access](Dataset_Access.md).

Kế hoạch45 ở trên là acquisition target và provenance lịch sử; **working set hiện tại là34**, không phải target đã hoàn tất. Phase8 freeze danh sách/hashes manifest hiện có sau QC; nếu bổ sung nguồn phải tạo snapshot/experiment mới, không thay dữ liệu giữa một experiment.

## 2. NTHU-DDD — external test
Nguồn chính thức: [HTTP page](http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/) và [license PDF](http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/NTHU-DDD-LicenseAgreement.pdf). HTTPS không kết nối được khi kiểm tra; HTTP đọc được.

- **36 người, khoảng 9.5 giờ**. Train gồm 18 người, 5 bối cảnh, 4 dạng sequence/người/bối cảnh → **360 sequence theo mô tả**; cần đếm archive xác nhận.
- Evaluation + testing: **90 video từ 18 người còn lại**. Trang đọc được không phân tách rõ số người/video của từng tập; không ghi 4/14 và 20/70 như sự thật đã xác minh. Ghi số thật từ manifest khi được cấp dữ liệu.
- Bối cảnh: BareFace, Glasses, Sunglasses, Night_BareFace, Night_Glasses. Thu bằng **IR**, 640×480 AVI; trang mô tả 30 FPS ngày, 15 FPS đêm.
- Người tham gia diễn các hành động khi chơi game mô phỏng lái xe: yawning, slow blinking, nodding; negative gồm nói, cười, nhìn hai bên.
- License nói mỗi frame drowsy/non-drowsy; xác nhận định dạng và file annotation trong archive trước khi đánh giá. Không có lớp Low Vigilance tương đương UTA.
- **Phải request:** điền agreement và email `cvlablai636@my.nthu.edu.tw`, subject “Dataset on Driver Drowsiness Detection from Video”. Agreement cần trưởng khoa hoặc giám đốc lab ký.
- Chỉ nghiên cứu/giáo dục phi thương mại; **không chia sẻ cho bên thứ hai**. Chỉ ảnh người 010 & 021 được dùng cho academic publication theo agreement. NTHU có quyền thu hồi access.
- License ghi chung 30 FPS, khác mô tả 15 FPS đêm trên website. Đọc timestamp/FPS thực tế, không hard-code theo PDF.

**Mặc định không training chung**: nhãn và modality khác nhau, trộn dễ che mất câu hỏi phát hiện Low Vigilance. Freeze model UTA, dùng phần có annotation để external test. Do collapsed label chưa hoàn toàn tương đương, báo cáo proxy nhị phân `p_risk=p_low+p_drowsy`; threshold chọn trên UTA validation. Báo cáo riêng từng bối cảnh, cùng tỷ lệ abstention. Nếu official test không có nhãn, chỉ báo cáo trên subset có nhãn và gọi đúng tên, không tạo nhãn giả.

## 3. YawDD — tùy chọn cho miệng
Nguồn: [trang tác giả](https://www.site.uottawa.ca/~shervin/yawning/), [IEEE DataPort](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset).

- **322 video** camera dưới gương trước; mỗi người 3–4 clip. **29 video** camera dashboard, mỗi người 1 clip nhiều hành động. Tổng **351 video** theo hai set, không gọi toàn dataset là 322.
- RGB 640×480, 30 FPS AVI không âm thanh; silent, talking/singing, yawning. Archive hiện liệt kê khoảng **4.94 GB**.
- Trang archive **không ghi rõ tổng unique subjects**; không chốt 107 khi chưa đối chiếu paper và Participants Information. Có thể cùng người xuất hiện giữa hai set; đếm theo participant ID, không theo video.
- Trang nói video **unlabeled**. Team phải gán event start/end ngáp trên subset nếu cần metric; không assume có nhãn từng frame.
- Tác giả và phần Open Access nói download miễn phí bằng tài khoản IEEE miễn phí. Trang đồng thời có modal chung nói cần subscription; có xung đột UI, phải kiểm tra sau login, không khẳng định đã tải thành công.
- Cho phép nghiên cứu; công bố screenshot phải kiểm tra cột “Allow picture to be shared publicly?”. Không tự gán một license thương mại chưa thấy.
- Không xác minh được official split đề nghị từ trang archive. Nếu dùng, chia người độc lập, ghi custom split và event annotation guideline.

**Vì sao chưa chốt số người YawDD:** [paper Sensors 2023, section 3.1](https://pmc.ncbi.nlm.nih.gov/articles/PMC10650052/) đưa bảng 119 người nhưng phần mô tả lại nêu 110 ở Mirror, không khớp chính bảng đó và số video archive chính thức. Nguồn thứ cấp có mâu thuẫn nên không dùng một con số phổ biến làm sự thật. Đếm Participants Information sau access; cũng không áp dụng random image split của paper này cho project.

## Manifest phải tạo sau download
`dataset_name, subject_id, video_id, relative_path, source_label, label_id, fold_id, split, fps_reported, duration_s, width, height, sha256, image_publishable, annotation_path`.

Giữ nguyên bytes từng video raw; adapter map đường dẫn sang manifest, không tái mã hóa. Full ZIP không được lưu do hạn mức ổ đĩa. Lưu URL/ngày và SHA256/CRC mỗi file trong receipt; agreement nếu có đặt riêng. `.gitignore` raw/processed và thông tin người. Không đăng dataset, calibration cá nhân, webcam video lên GitHub.

**Điều chưa xác minh:** toàn bộ bytes của những thành viên UTA không được chọn; annotation/package layout NTHU; số unique subjects YawDD. Membership UTA đã lấy từ official ZIP index. Không bù phần chưa có bằng mirror không rõ nguồn.
