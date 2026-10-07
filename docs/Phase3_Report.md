# Phase 3 — Implementation, verification và readiness

Ngày kiểm tra: **2026-10-06**. Đối chiếu Phase 3 trong [roadmap](16_Development_Roadmap.md), [module contracts](15_Module_Specification.md), [feature topology](06_Feature_Engineering.md) và [test strategy](17_Testing_Strategy.md).

**Snapshot lịch sử trước Phase4/5:** bảng readiness và số test bên dưới mô tả thời điểm nghiệm thu Phase3, không thay thế trạng thái hiện tại. EAR/MAR đã triển khai sau đó; xem [roadmap Phase4/5](16_Development_Roadmap.md#phase-4--ear-extraction) và [changelog](CHANGELOG.md). Human webcam transitions/physical disconnect vẫn chưa quan sát; giữ nguyên bằng chứng Phase3.

## Kết luận
**Code Phase 3 đã triển khai, có runtime proof cho detector/reader/overlay/no-face và camera capture. Đủ prerequisite kỹ thuật để bắt đầu Phase 4–6 trên clip nhỏ. Chưa claim nghiệm thu hardware đầy đủ hoặc codebase hoàn tất mọi yêu cầu trong docs.** Chưa quan sát face→no-face trực tiếp trên webcam hay rút camera vật lý. Dữ liệu vẫn thiếu 11/45 video; không training hoặc classifier đã học.

## Code đã có
- `src/preprocessing/video_reader.py`: sequential decode; finite FPS/count/duration; strict source and integer-ms timestamps; grid sampling target≤20; không upsample/seek mỗi frame. EOF, prefix stop và ERROR khác nhau. Capture ownership dùng session token nên iterator cũ không đóng source mới.
- `src/features/landmarks.py`: constructor nhận resolved config, MediaPipe Tasks VIDEO/num_faces=1, BGR→RGB, source-bound timestamp, float32[478,3] hữu hạn. No-face/malformed output là points null + face_detected false; không bịa confidence/visibility. Context manager/idempotent close và stats.
- `scripts/preview_landmarks.py`: repeated videos hoặc camera, detector mới mỗi source; overlay chưa mirror, giữ aspect ratio và topology mắt theo giải phẫu. Headless và bounded prefix rõ ràng; report JSON, throughput/drop/inference latency. Native model initialization failure giữ source error report.
- Không đổi shared dataclasses/enums; không làm EAR/MAR/pose/quality gates/classifier hay CameraWorker của phase sau.

## Evidence đã chạy
| Scenario | Kết quả quan sát | Artifact |
|---|---|---|
| Subject 04, source `0/5/10`, mỗi prefix 12s | 241 emitted/inference frames mỗi clip, tổng **723 face frames**, zero invalid landmarks. Decode 301/361/301; sampling drops 60/120/60. CAP_PROP_POS_MSEC, không fallback. Capture released và detector closed. | `runs/phase3/three_clips/preview_report.json` |
| Offline throughput ba clip | Khoảng **69–95 processed FPS**, p95 inference **8.97–11.45 ms** ở run này; là xử lý file không pace, không camera FPS hay benchmark 10 phút. | report ba clip |
| Overlay thực | Native PrintWindow chỉ lấy đúng preview HWND; đã xem ảnh: landmark theo mặt, mắt trái giải phẫu cyan/phải orange, chưa mirror, aspect đúng 640×362. Source 04 có quyền công bố ảnh theo CSV tác giả. | `runs/phase3/visual/window_capture.png`, `window_observation.json`, `preview_report.json` |
| Model blank-image | Native model trả `points=None`, `face_detected=False`; close đạt. | `runs/phase3/no_face.json` |
| Full synthetic AVI, 15 FPS, một giây | 15 decoded/emitted, zero duplication/drop; all no-face; **EOF**, capture/model release. File tạm đã dọn. | `runs/phase3/full_eof_smoke.json` |
| Webcam 0, bounded 8s | 218 decoded, 131 emitted, 87 dropped; **131 no-face**, không landmark lỗi. Clock monotonic QPC, capture released/model closed. **Không có bằng chứng mặt người trên webcam trong run này.** Không lưu camera images. | `runs/phase3/webcam_fixed/preview_report.json` |
| Camera 99 không khả dụng | Actual open failure → ERROR, CLI exit1, zero frames, capture/model release. **Không phải physical mid-stream disconnect.** | `runs/phase3/camera_unavailable/preview_report.json` |
| Regression suite, final recheck | `.venv/Scripts/python.exe -m pytest -q --tb=short`: **99 passed in 8.89s**. Source FPS 15/24/30, VFR timestamps, duplicate/backward, fallback, metadata/corruption, lifecycle, schema/input và source mismatches. | terminal verification |
| Final actual-video recheck | Prefix3s, source04/0: 76 decoded,61 face/inference frames,15 sampling drops, zero invalid; source time0→3000ms, no fallback, capture/model release, exit0. | `runs/phase3/final_check/preview_report.json` |

Ảnh overlay chỉ lưu khi opt-in; raw data và runs nằm trong gitignore. Kính râm/low-light có thể vẫn có landmark nhìn hợp lý: face present **không** là quality tuyệt đối. Không lấy các clip nhãn Drowsy làm bằng chứng classifier hoặc trạng thái sinh lý từng frame.

## Lỗi thật đã sửa
1. **Clock camera Windows/Python 3.12:** `monotonic_ns` dùng GetTickCount64, resolution15.625ms; webcam run đầu fail sau 19 decoded vì timestamp trùng. Đo `get_clock_info` xác nhận coarse clock; `perf_counter` là monotonic QPC/resolution100ns. Regression fail-before/pass-after và actual webcam sau sửa đạt bounded capture. Không cộng thời gian giả, không bỏ exception để báo PASS. Failed-before artifact giữ tại `runs/phase3/webcam/`.
2. **Native detector constructor:** trước đây nằm ngoài try nên mất source error report. Regression fail-before/pass-after; chuyển constructor vào lifecycle/report guard. Config/asset path startup errors vẫn không có fallback giả.
3. **Preview aspect ratio:** actual window ban đầu bị ép vuông; resize theo dimensions frame thật rồi quan sát lại surface đúng aspect. Không đổi ảnh inference hoặc mirror.

## So với docs: đã đủ điều kiện nào?
| Gate | Đánh giá |
|---|---|
| Phase 3 prerequisites: Phase0 + vài clip Phase2 | **Đủ**: assets/môi trường hiện chạy native Tasks; đã dùng clip thật của cả ba class. Thiếu 11 video không chặn gate này. |
| Phase 3 core acceptance: đúng overlay, no-face invalid, stats, release | **Có runtime evidence**, cùng strict time/source/session regression. |
| Phase 3 hardware smoke checklist | **Chưa đủ toàn bộ**: có actual camera/no-face và unavailable device; disconnect giữa stream chỉ có deterministic backend regression. Human face→no-face và physical disconnect chưa quan sát, không coi test double là hardware PASS. |
| Phase4/5/6 prerequisites | **Đủ để triển khai trên clip nhỏ**. Cần unit số đúng và live/visual QC riêng của từng phase; chưa có EAR/MAR/pose. |
| Full dataset Phase1/2 | **Chưa đủ**: raw34/45,12/15subjects; subject51 thiếu hai class, fold coverage3/3/3/2/1. Source lỗi gần nhất là Drive quota; lần này không thử tải lại. |
| Phase7/8/9/training | **Chưa đủ**: chưa pipeline/features/QC/profile/splits. Phải hoàn tất tương ứng, không train từ landmark-only hoặc gọi partial manifest là complete. |
| Full system described in docs | **Chưa đủ và đúng theo progression**: Phase4–24 vẫn là thiết kế; chưa classifier/model bundle, realtime worker, UI/audio và safety evaluation. |

## Điểm cần thảo luận/chốt trước các phase dữ liệu tiếp theo
1. **User đã chốt:** trước mắt dùng34 video/12subjects hiện có, không đợi45/15subjects. Roadmap8/9 và training/evaluation docs đã chuyển working-set policy sang snapshot có provenance, subject-disjoint restricted folds, counts thực tế; không áp9/3/3. Full60/36–12–12 chỉ giữ làm reference. Subject51 thiếu Alert phải P1 abstain, fold5 P1 validation/test không có accepted subject phải blocked/undefined; không tự chuyển sang P0 hoặc loại nguồn để báo đủ. Training vẫn phải đợi features/QC/profiles/splits tương ứng.
2. Hardware QC còn thiếu cần người ở trước camera và thao tác disconnect khả thi. Không tự disable thiết bị Windows để giả rút camera, không lấy no-face hiện tại suy rằng camera hỏng hay shutter đang đóng.
3. Webcam packet time dùng **perf_counter**. Future age/staleness checks phải cùng clock; file replay dùng timestamp nguồn, không dùng tốc độ đọc. Điều này đã ghi vào contract15 để tránh trộn epochs.

## Lệnh chạy
```powershell
.venv/Scripts/python.exe -m scripts.preview_landmarks --video data/raw/uta_rldd/04/0.mp4 --seconds 12
.venv/Scripts/python.exe -m scripts.preview_landmarks --camera 0 --seconds 8
.venv/Scripts/python.exe -m scripts.preview_landmarks --video data/raw/uta_rldd/04/0.mp4 --video data/raw/uta_rldd/04/5.mp4 --video data/raw/uta_rldd/04/10.mp4 --seconds 12 --headless --report-dir runs/phase3/three_clips
```
`q`/Esc dừng cửa sổ. `--save-overlay` là opt-in; kiểm tra quyền công bố trước chia sẻ. Không bật `--constant-fps-verified` chỉ từ metadata FPS dương; caller phải khảo sát constant-FPS độc lập. Không truyền `--seconds` nếu muốn đọc toàn file, nhưng full decode không thay thế feature quality/QC.
