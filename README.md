# Driver Drowsiness Detection

Tài liệu thiết kế đã được chuyển vào **[docs/README.md](docs/README.md)**. Roadmap triển khai: [docs/16_Development_Roadmap.md](docs/16_Development_Roadmap.md).

## Phase 0–3
**Trạng thái nghiệm thu:** Phase 0 đạt. Code Phase 1–2 đã có nhưng dataset mới **34/45 video, 12/15 subjects, 8,206,607,880 bytes (~7.64 GiB)**. Drive chặn 11 file bằng “Quota exceeded”; **Phase 1–2 chưa đạt full acceptance**. Exploration chạy trên dữ liệu thật và exit 1 đúng để chặn báo thành công sai. Xem [Dataset Access](docs/Dataset_Access.md) và [changelog](docs/CHANGELOG.md).

- Python **3.12 x64**, môi trường riêng `.venv/`.
- `configs/`: cấu hình preprocessing/training/realtime; `src/contracts.py`: dữ liệu dùng chung.
- `scripts/check_environment.py`: tải asset chính thức và kiểm tra MediaPipe, PyTorch, cửa sổ Qt thật.
- `scripts/acquire_manifest.py`: chỉ tải thành viên ZIP đã chọn bằng HTTP Range, không tải toàn archive.
- `scripts/explore_dataset.py`: kiểm tra video thật, checksum, nhãn/fold; xuất manifest Parquet và report.

Kế hoạch UTA subset: **15 người × 3 trạng thái = 45 video**, ba người từ mỗi official fold; **chưa tải đủ**. Chọn các subject đầy đủ có tổng dung lượng thấp nhất để tiết kiệm ổ đĩa. Đây là **development subset thiên lệch theo dung lượng**, không đại diện full benchmark 60 người. Dữ liệu đã tải nằm tại `data/raw/uta_rldd/`, ngay trong project; không commit vào Git.

**Phạm vi làm việc được user chốt:** tiếp tục development với **34 video hiện có / 12 subjects**, không đợi đủ45, không tự tải thêm. Kế hoạch acquisition45 và ledger11 missing được giữ để truy nguyên, không sửa thành “đã đủ”. Phase8 xử lý snapshot34 sau khi Phase7 qua QC; Phase9 chia theo subject/official fold với số lượng thực tế, không cố định9/3/3 hoặc36/12/12. Subject51 thiếu Alert nên P1 phải abstain, không lấy Low Vigilance làm baseline hoặc tự đổi sang P0.

Phase 3 đã có `VideoReader`, `FaceLandmarkDetector` và preview chưa mirror. Ba clip thật, no-face, camera capture, timestamp/sampling và release đã chạy; test **rút camera vật lý/chuyển face→no-face trực tiếp** chưa được quan sát. Xem [báo cáo Phase 3](docs/Phase3_Report.md); đủ prerequisite để bắt đầu Phase 4–6 trên clip nhỏ, không phải nghiệm thu toàn hệ thống.

## Cài lại môi trường — PowerShell
```powershell
py -3.12 -m venv .venv
uv pip install --python .venv/Scripts/python.exe --no-cache torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe --no-cache -r requirements-lock.txt
```

Nếu chưa có `uv`, dùng `.venv/Scripts/python.exe -m pip install --no-cache-dir ...` với cùng package/index. `requirements-lock.txt` chỉ được tạo sau khi full environment smoke đạt. Bản CPU được chọn cho Phase 0 để giảm dung lượng; máy có GPU không có nghĩa CPU wheel chạy CUDA. Khi đến training, chọn CUDA wheel theo [PyTorch installer chính thức](https://pytorch.org/get-started/locally/) và kiểm tra lại `torch.cuda.is_available()`; không cần cài CUDA Toolkit cho Phase 0.

### CUDA chỉ khi đến training — chưa cài/chưa smoke
Official [CUDA 13.0 wheel index](https://download.pytorch.org/whl/cu130/torch/) có `torch-2.14.1+cu130-cp312-cp312-win_amd64.whl`. Máy hiện có RTX 4050 Laptop 6 GB, driver 610.62; cần kiểm tra lại driver và dung lượng trước cutover. Lệnh tham khảo, **không chạy ở Phase 0**:
```powershell
uv pip install --python .venv/Scripts/python.exe --no-cache --reinstall-package torch torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/python.exe -c "import torch; assert torch.cuda.is_available(); print(torch.rand(1, device='cuda').item())"
```
Sau khi GPU smoke đạt, cập nhật training `device`, environment report và lockfile riêng; lockfile CPU hiện tại không được dùng để ép CUDA wheel trở lại CPU.

## Chạy công cụ
Chạy tại project root, không dùng Python global/Conda mặc định:

```powershell
.venv/Scripts/python.exe -m scripts.check_environment --download-assets --assets-only
.venv/Scripts/python.exe -u -m scripts.acquire_manifest --workers 3
.venv/Scripts/python.exe -m scripts.explore_dataset
.venv/Scripts/python.exe -m scripts.check_environment --sample-video data/raw/uta_rldd/04/0.mp4
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m scripts.preview_landmarks --video data/raw/uta_rldd/04/0.mp4 --seconds 12
.venv/Scripts/python.exe -m scripts.preview_landmarks --camera 0 --seconds 8
```

Kiểm tra đuôi/tên video trong `data/acquisition/uta_subset_plan.json` nếu chạy smoke với video khác. Downloader kiểm tra file đã có bằng size/CRC, không tải lại video hợp lệ; partial thuộc kế hoạch được dọn trước preflight và video chưa hoàn tất phải tải lại từ đầu. Không thêm NTHU/YawDD và không tải các video ngoài kế hoạch 45 file.

Tạm thời không cần chạy lại acquisition để tiếp tục development. Nếu sau này mở rộng dữ liệu và quota nguồn được gỡ, chạy acquisition rồi exploration tuần tự. `--inventory data/acquisition/uta_source_inventory.json` giữ archive IDs nhưng vẫn re-fetch indexes; không bypass quota. File verified không tải lại; không đổi subject/classes để che phần thiếu.

Notebook: mở `notebooks/01_dataset_exploration.ipynb` bằng VS Code và chọn kernel `.venv` (Python 3.12); `ipykernel` đã nằm trong dev dependencies. Notebook in số file/subject thật, không coi plan 45 là kết quả tải.

Preview dùng MediaPipe Tasks VIDEO, RGB và session mới mỗi video. `--headless` không mở cửa sổ; `--save-overlay` là opt-in lưu ảnh mặt, phải kiểm tra quyền công bố. Mặc định không lưu ảnh. Không bật `--constant-fps-verified` chỉ vì FPS metadata dương: flag này là xác nhận của caller sau khảo sát constant-FPS độc lập. `--seconds` chỉ kiểm tra prefix; chạy file headless không pace như thời gian thực, FPS report là throughput xử lý, không FPS camera.

## Output
- `models/assets/`: Face Landmarker `.task`, canonical OBJ và metadata URL/SHA256.
- `data/acquisition/`: remote inventory, kế hoạch subset, permissions và download receipts/report.
- `data/processed/manifest.parquet`: dữ liệu video, nhãn nội bộ `0/1/2`, fold provenance, checksum và lỗi.
- `runs/phase0/`: báo cáo môi trường, resolved config và ảnh cửa sổ Qt thử nghiệm.
- `runs/phase2/`: exploration report, manifest CSV và biểu đồ không có ảnh mặt.
- `runs/phase3/`: reports clip/camera/no-face, throughput/drop và optional overlay; screenshot bằng chứng chỉ lấy subject 04 có quyền công bố.
- `notebooks/01_dataset_exploration.ipynb`: đọc manifest để team xem thống kê.

Video, ảnh thử nghiệm, environment và output lớn được `.gitignore`. `image_publishable` chỉ là quyền công bố ảnh, không phải quyền tái phân phối dataset. Nhãn UTA áp cho cả video, không chính xác từng frame. Phase 4 trở đi, training và UI cuối cùng chưa thuộc phần triển khai này.
