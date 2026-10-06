# Changelog

## Quyết định phạm vi — dùng dataset hiện tại

- User chốt tiếp tục với **34 video/12subjects** hiện có; không đợi hoặc tự tải thêm11 file. Acquisition plan45/missing ledger và full-source gate được giữ, không đổi partial thành complete.
- Roadmap8 xử lý working snapshot sau QC7; roadmap9/training/evaluation dùng restricted official membership và cardinality thực tế, không ép9/3/3 hoặc36/12/12. Freeze paths/hashes và version snapshot khi thêm nguồn.
- Giữ subject51/Low Vigilance trong provenance/coverage. Thiếu Alert khiến P1 abstain và fold5 P1 validation/test blocked/undefined; không fake baseline hoặc silent P0 fallback. Không đổi calibration config/checkpoint mode.
- Đây là đổi phạm vi tài liệu cho các phase tiếp theo, không training/splits implementation mới và không claim full benchmark. Phase3/hardware verification giữ kết quả đã báo.


## 2026-10-06 — Phase 3

- Thêm sequential `VideoReader`, MediaPipe Tasks VIDEO `FaceLandmarkDetector`, preview chưa mirror và regression tests; giữ shared contracts. Source timestamps strict, target-grid≤20, không upsample, fallback CFR chỉ khi caller đã kiểm chứng; EOF/STOPPED/ERROR và release phân biệt.
- Runtime ba prefix 12s, subject04/cả ba class: 723 face frames, zero invalid landmarks; overlay topology/aspect đúng đã xem trên actual preview HWND. Model no-face/blank và full synthetic EOF đã chạy. Không decode toàn dataset.
- Camera0 sau sửa: 218 decoded/131 emitted, tất cả no-face; capture/model release. Camera99 unavailable: ERROR/exit1/release. Chưa physical disconnect hoặc human face→no-face transition, không claim hardware checklist hoàn chỉnh.
- Sửa clock Python3.12 Windows từ coarse GetTickCount64/monotonic_ns sang high-resolution monotonic perf_counter_ns/QPC. Regression fail-before/pass-after và webcam runtime xác nhận. Contract15 yêu cầu future age/staleness dùng cùng clock.
- Sửa native model initialization failure giữ source error report; regression fail-before/pass-after. Preview resize giữ source aspect ratio.
- Final recheck: **99 passed in 8.89s**; actual video prefix3s:61 face frames, no errors, capture/model release, exit0 (`runs/phase3/final_check/preview_report.json`). Read-only temporal/detector reviews đã hoàn tất; actionable constructor-report finding đã sửa.
- [Phase3_Report.md](Phase3_Report.md) ghi evidence và readiness: đủ bắt đầu Phase4–6 trên clip nhỏ; dataset34/45 và pipeline/training/UI tương lai vẫn chưa nghiệm thu. Đề xuất thảo luận restricted15-subject protocol thay cho full60 targets còn trong roadmap, không silently migrate.


## 2026-10-06 — Phase 0–2

### Tài liệu
- Chuyển 21 tài liệu kỹ thuật Markdown gốc vào `docs/`; root README là lối vào và hướng dẫn chạy thực tế.
- Cập nhật roadmap, cấu trúc project, shared contracts và môi trường đã kiểm chứng. Nguồn, quyền ảnh, selection policy, checksum và giới hạn dữ liệu ghi trong [Dataset_Access.md](Dataset_Access.md).

### Phase 0 — đạt
- Python 3.12 x64 trong `.venv`; requirements và lockfile, config YAML có validation và path resolution độc lập working directory; shared dataclasses/enums.
- Smoke MediaPipe Tasks VIDEO trên hai frame video thật: một mặt/frame, 478 landmarks/mặt. Asset chính thức có URL/SHA256 metadata.
- PyTorch CPU: forward LSTM `[1,100,16] → [1,3]`; cửa sổ Qt thật và screenshot chỉ chứa text; `pip check` không có broken requirements.
- Bằng chứng: `runs/phase0/environment_report.json`, resolved configs và `qt_smoke.png`. Network smoke dùng trọng số ngẫu nhiên, **không phải classifier đã train**. CUDA chưa cài/chưa smoke.

### Phase 1 — code đã chạy, dữ liệu chưa đủ
- Inventory ZIP64 chính thức: 10 archives, 182 physical files / 180 logical recordings; giữ multipart IDs và official fold provenance.
- Kế hoạch 15 complete subjects, ba người/fold, 45 file; chọn dung lượng nhỏ nhất, không phải full/random benchmark. Chỉ stream exact compressed-member Range; từ chối HTTP 200, không tải full ZIP. CRC32/size trước publish, SHA256 và receipts sau xác minh.
- Sửa an toàn path/temporary/receipt để không ghi đè file ngoài raw root qua symlink/hardlink; dọn partial thuộc plan trước disk preflight. Streaming disk reserve là best-effort trước concurrent/external writes.
- **34/45 video verified, 12/15 subjects, 8,206,607,880 bytes (~7.64 GiB)**. 11 file còn thiếu bị Drive quota; cả route chính thức/form và các Range thử nghiệm chưa hoàn tất chúng. Không thay subject/mirror để che phần thiếu. Xóa scaffolding thử nghiệm và inventory prototype lỗi thời.

### Phase 2 — exploration thật, full acceptance bị chặn
- Manifest Parquet giữ leading-zero IDs, nhãn nội bộ nullable, lỗi, hash, official fold source và `split="unassigned"`. Kiểm tra exact plan identity và receipt provenance, không chỉ so số lượng.
- Trên 34 video: SHA256 và decode đầu/giữa/cuối hợp lệ; class counts 11/12/11, subject counts theo fold 3/3/3/2/1, khoảng 5.796 giờ. Duration/FPS là metadata container, chưa khảo sát VFR toàn clip.
- CLI **exit 1 đúng** vì thiếu dữ liệu, có missing IDs/download errors trong `runs/phase2/exploration_report.json`; không claim Phase 1–2 đã đạt.
- Notebook chạy thành công cả ba code cells bằng kernel `.venv` thật, in 34/45 và 12/15. Kernel TCP có cảnh báo không mã hóa; không expose lên mạng công cộng. Bằng chứng: `runs/phase2/notebook_smoke.json`.
- Storage check: không có video ngoài plan, full ZIP hoặc partial; còn 51,347,931,136 bytes trống tại lúc kiểm tra. Bằng chứng: `runs/phase2/storage_check.json`.
- Sửa environment smoke dùng report directory ngoài project; cửa sổ Qt và report đã chạy thành công với directory đó.

### Verification và phạm vi
- Regression suite bao phủ config boundaries, HTTP Range/CRC/SHA, path/partial safety, metadata/labels và plan/source mismatch. Lệnh `.venv/Scripts/python.exe -m pytest -q --tb=short`: **36 passed in 5.72s**.
- Không tải/request NTHU hoặc YawDD; không training, accuracy, checkpoint giả hay runtime Phase 3 trở đi.
