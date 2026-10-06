# Dataset Access — Phase 1–2

## Trạng thái thực tế — 06/10/2026
**Phase 1–2 chưa đạt full acceptance:** đã tải và xác minh **34/45 video**, thuộc **12/15 subjects** (11 người đủ ba trạng thái, subject 51 chỉ có `5`). Raw data **8,206,607,880 bytes (~7.64 GiB)**, nằm trong project. Có 34 verified receipts; không có video ngoài plan, full ZIP hay partial còn lại. Dung lượng trống tại lần kiểm tra cuối: **51,347,931,136 bytes (~47.82 GiB)**.

**Quyết định user sau Phase3:** trước mắt dùng chính **snapshot34 video/12subjects** để làm các phase tiếp theo, không chờ hoặc tự tải thêm11 file. Snapshot không đồng nghĩa hoàn tất acquisition45; giữ nguyên plan/receipts/missing ledger và exit1 của full-acquisition gate. Future builder phải freeze danh sách/hashes34 source đã verified, kiểm tra mọi source trong snapshot có output hoặc explicit failure; denominator working-set34 tách riêng planned45. Subject51 chỉ có Low Vigilance: giữ source/coverage, P1 thiếu Alert phải abstain theo protocol11; không silently loại hoặc gộp P0 fallback.

| Archive bị chặn | Video còn thiếu |
|---|---|
| Fold4_part2.zip | `46_0`, `46_5`, `46_10` |
| Fold5_part1.zip | `51_0`, `51_10`, `54_0`, `54_5`, `54_10` |
| Fold5_part2.zip | `60_0`, `60_5`, `60_10` |

Google Drive trả trang **“Quota exceeded” / HTTP 200 thay vì byte Range 206**. Đã thử Range member thông thường, range nhỏ, route `/uc` chính thức và form “Download anyway” với UUID thực tế; chưa có transfer hoàn chỉnh cho 11 file này. Một số central-index request cũng bị HTTP 200. Không đổi subject, dùng mirror hoặc coi một probe Range nhỏ là bằng chứng tải xong.

Exploration trên 34 file thật: nhãn nội bộ **11/12/11**, subjects/fold **3/3/3/2/1**, khoảng **5.796 giờ**; SHA256 và decode mẫu hợp lệ cho tất cả 34 file. Manifest giữ `split="unassigned"`. CLI **exit 1** vì subset chưa đủ; report ghi missing IDs và download errors. `errors.json=[]` chỉ nói các file hiện có hợp lệ, không nói đủ dữ liệu.

Notebook đã chạy cả ba code cells trong kernel `.venv` thật; output là **34/45**, không phải 45. Bằng chứng: `runs/phase2/notebook_smoke.json`, `storage_check.json`, `exploration_report.json`. ipykernel cảnh báo transport TCP không mã hóa; không expose kernel lên mạng công cộng.

Khi quota nguồn được gỡ, chạy lại acquisition rồi exploration theo README. `--inventory data/acquisition/uta_source_inventory.json` giữ archive IDs nhưng vẫn re-fetch index, **không bypass quota**. File đã verified chỉ được kiểm tra lại; file chưa hoàn tất phải tải lại từ đầu.


## Nguồn UTA-RLDD
Kiểm tra **06/10/2026**, tải trực tiếp từ nguồn tác giả, không dùng mirror:
- [Trang dataset](https://sites.google.com/view/utarldd/home).
- [Google Drive chính thức](https://drive.google.com/drive/folders/1d_QwgpMXnLY_FmLYXDn7TLcw0D-svXEl).
- [Image Publishable CSV](https://docs.google.com/spreadsheets/d/1w0eXPj8hadPeEwTlnbdBCKlnGp_tJXJ-axYs5S-hqyE/export?format=csv).
- [Paper CVPR Workshops 2019](https://arxiv.org/html/1904.07312).

Mục đích project là nghiên cứu. Trang công khai không đồng nghĩa raw data có giấy phép thương mại/tái phân phối; trích dẫn tác giả, không đăng video lên GitHub. Bảng tác giả cho phép công bố ảnh của **36/60 người**. Map boolean lưu `data/acquisition/uta_image_permissions.json`; manifest giữ `image_publishable`, không suy từ việc file tải được. Subject **04** được phép công bố ảnh; environment smoke xử lý hai frame cục bộ, screenshot Qt không chứa mặt.

## Quyết định dung lượng theo yêu cầu user
Kế hoạch chỉ lấy **45 physical video files / 15 complete subjects**, tương ứng một phần tư số 180 subject-state recordings danh nghĩa. Chọn ba subject đầy đủ, dung lượng nhỏ nhất trong từng official fold:

| Official fold | Subject IDs |
|---|---|
| 1 | 04, 05, 10 |
| 2 | 16, 17, 18 |
| 3 | 27, 31, 35 |
| 4 | 44, 45, 46 |
| 5 | 51, 54, 60 |

Mỗi người trong kế hoạch đủ `0=Alert`, `5=Low Vigilant`, `10=Drowsy`; map nội bộ **0/1/2**. Dự kiến **13,968,447,679 bytes raw (~13.01 GiB)**, payload nén **13,948,400,002 bytes** cộng một lượng nhỏ ZIP headers/index. Không tái mã hóa/rút ngắn video để giả đủ số lượng. Selection thiên lệch theo dung lượng, không random/đại diện demographic hay full benchmark. Preflight yêu cầu đủ chỗ cho phần raw còn thiếu cộng **8 GiB** reserve; kiểm tra reserve trong lúc stream là **best-effort**, không bảo đảm tuyệt đối trước concurrent/external writes. Không tải NTHU/YawDD trong đợt này.

## Cơ chế selective download và provenance
Folder nguồn gồm **10 ZIP**; index ZIP64 có **182 files vật lý cho 180 logical recordings**, vì subject 32/49 có Drowsy tách `10_1`/`10_2`. Subject tách/duplicate/missing class không đủ điều kiện cho subset ba file/người này; parser vẫn giữ `part_id` nếu dùng inventory đầy đủ sau.

Downloader đọc central directory và local header qua HTTP byte Range, rồi stream chỉ compressed member đã chọn. Bắt buộc **206 + Content-Range khớp**; server trả 200 bị từ chối trước khi tiêu thụ full archive. Không lưu `.zip` nào. Output `data/raw/uta_rldd/<subject>/<source_filename>`, vẫn giữ đuôi MP4/MOV/M4V nguồn. CRC32 đối chiếu ZIP index và size trước rename `.part` thành video; SHA256 tự tính ghi sidecar. SHA256 này **không phải checksum do tác giả cung cấp**, và không có checksum toàn ZIP vì không tải full ZIP.

- `uta_source_inventory.json`: source file ID, archive size, member path, offsets, CRC32, original labels, official fold của toàn index.
- `uta_subset_plan.json`: exact 45 members, subject IDs, strategy và expected bytes.
- `uta_download_report.json`: verified receipts và lỗi từng download; chỉ đủ 45 verified receipts, zero errors mới hoàn tất Phase 1.
- `<video>.receipt.json`: provenance + SHA256/CRC/size của file đó.

Fold lấy từ **official archive/member paths**, không tự chia theo số subject. Phase 2 chưa gán train/val/test; `split="unassigned"`. Phase 9 mới xây subject-independent splits trên subset và phải báo đúng là development-subset protocol.

## Exploration và giới hạn bằng chứng
`python -m scripts.explore_dataset` tạo `data/processed/manifest.parquet` và `runs/phase2/{exploration_report.json,errors.json,manifest_summary.csv,dataset_summary.png}`. Mỗi raw video đều có hàng hoặc error reason; invalid không bị im lặng loại. Manifest hash được đối chiếu verified receipts, giữ leading-zero ID và official fold source.

Decode **đầu/giữa/cuối từng video**, không decode/check mọi frame. `fps_reported` lấy metadata OpenCV; duration = frame_count/FPS là ước lượng container, không chứng minh constant-FPS. Timestamp/VFR khảo sát sâu thuộc Phase 3. Nhãn là trạng thái chủ đạo toàn video, không nhãn chính xác từng frame. Không có face-preview saved by default, không có model đã train hoặc accuracy được báo.

## Dataset chưa dùng
- **NTHU:** chưa request/tải; không claim external test. Agreement phải được ký khi team quyết định dùng.
- **YawDD:** chưa login/tải/annotate; không claim drowsiness labels hay adapter đã chạy.
