# 16 — Development Roadmap

Roadmap dùng để giao từng phase cho coding agent: **“Implement Phase 6 according to 16_Development_Roadmap.md”**. Mọi phase tuân theo contract [15](15_Module_Specification.md), config [14](14_Project_Structure.md) và nguồn [03–04](03_Research_Background.md).

**Phase0 đạt runtime verification; code Phase1–2 đã triển khai nhưng full acceptance chưa đạt vì34/45 video (Drive quota). Phase3–5 đã có reader/detector/EAR/MAR và runtime proof trên clip/camera; confirmed talking, human webcam transitions và physical disconnect chưa quan sát.** Xem kết quả Phase4/5 bên dưới, [báo cáo Phase3 lịch sử](Phase3_Report.md), [changelog](CHANGELOG.md), [Dataset Access](Dataset_Access.md) và reports trong `runs/`. Acceptance Phase4/5 có phần đã chạy/phần blocked được ghi riêng; Phase6–24 vẫn là mục tiêu thiết kế. Chưa có classifier buồn ngủ đã train.

**Phạm vi development user đã chốt sau Phase3:** tiếp tục với **34 video hiện có/12subjects**, không đợi45/180 hoặc tự tải thêm. Acquisition45 và11 missing vẫn là ledger/gate nguồn lịch sử; không đổi chúng thành thành công. Phase8/9 dùng working snapshot đã freeze, cardinality thực tế và protocol eligibility; không ép36/12/12 hay9/3/3. Subject51 thiếu Alert làm P1 abstain; fold5 P1 không có accepted subject, phải báo blocked/undefined. Tham chiếu04/09/10/11.

## Quy tắc làm một phase
- Đọc prerequisites và các tài liệu liên quan; không tự đổi schema/threshold/API.
- Viết test cho công thức, boundary hoặc hành vi có rủi ro; chạy fail trước, implement tối thiểu rồi pass. Smoke thật trên đường dữ liệu thay đổi.
- Không sang bước xử lý toàn dataset trước khi overlay/QC trên clip nhỏ đúng.
- Mỗi phase commit nhỏ kèm config/docs liên quan và log verification; không commit dataset/video khuôn mặt.
- Files “modify” có thể chỉ cần cập nhật section/config; không tạo implementation nằm ngoài objective.
- Phase 9 làm **pure calibration/profile transforms** để dùng offline; Phase 16 chỉ thêm **lifecycle realtime**. Nhờ vậy train trước UI/calibration manager mà không vòng dependency.
- NTHU access có thể chạy song song việc CV; thiếu access không chặn UTA nhưng chặn claim external test đã hoàn thành.

## Phase 0 — Environment setup

- **Objective / Goal:** Khóa môi trường chạy được, chưa train model.
- **Prerequisites / Dependencies:** Không có; xem compatibility ở 03.
- **Files to create:** requirements.txt, requirements-dev.txt, requirements-lock.txt; configs/{preprocessing,training,realtime}.yaml; src/config.py, src/contracts.py; scripts/check_environment.py; .gitignore, docs/CHANGELOG.md
- **Files to modify:** Không có.
- **Functions/classes:** load_config(path), validate_config(config); shared records ở 15.
- **Input → Output:** Candidate versions, paths → venv đã smoke, resolved config, environment report.
- **Technical tasks / Implementation notes:** Python 3.12 x64; chỉ OpenCV contrib; asset .task/canonical từ nguồn chính thức + SHA256. Resolve, pip check; CPU trước, CUDA train sau kiểm tra driver. Không khóa metadata-only candidate như đã tested.
- **Tests:** tests/test_config.py: reject FPS≤0, thresholds ngược, model mode mismatch. Smoke import cv2/mediapipe/torch/PySide6; landmarker một ảnh có consent, forward [1,100,16], mở/đóng Qt window.
- **Acceptance criteria / Definition of Done:** pip check không lỗi; landmark, LSTM, Qt smoke có output/screenshot thật; lockfile và lệnh cài CUDA được lưu.
- **Expected result:** venv đã smoke, resolved config, environment report. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** ABI NumPy, CUDA driver hoặc wheel lỗi: chọn bản compatible và log; không dùng environment global.

## Phase 1 — Dataset acquisition

- **Objective / Goal:** Có dữ liệu hợp pháp và nguồn gốc rõ.
- **Prerequisites / Dependencies:** Phase 0.
- **Files to create:** src/datasets/acquisition.py; scripts/acquire_manifest.py; data/raw/uta_rldd/; data/acquisition/; docs/Dataset_Access.md
- **Files to modify:** README.md phần access.
- **Functions/classes:** record_source(dataset,url,terms), inventory_archive(archive), select_subset(records), download_member(record,root).
- **Input → Output:** Official URL/ZIP index → 45 video gốc được kiểm CRC/SHA256, sidecar receipts, inventory và access ledger.
- **Technical tasks / Implementation notes:** Theo giới hạn user: 15 complete subjects, ba mỗi official fold; chỉ tải 45 members bằng HTTP Range, giữ ≥8 GiB reserve, không tải full ZIP/NTHU/YawDD. Record source IDs/member offsets/folds từ index chính thức, không dùng mirror.
- **Tests:** tests/test_acquisition.py và test_boundaries.py: balanced selection, thiếu/duplicate class, giới hạn subset, giải nén Range thật, CRC lỗi, server bỏ qua Range và unsafe output path.
- **Acceptance criteria / Definition of Done:** Đủ 45 verified receipts cho 15 người, mỗi người ba class; không có video ngoài plan/full ZIP, lỗi tải được ghi rõ; image permissions đối chiếu nguồn tác giả.
- **Expected result:** raw subset, uta_source_inventory.json, uta_subset_plan.json, uta_download_report.json và docs/Dataset_Access.md. NTHU/YawDD deferred, không claim access/external evaluation.
- **Possible problems:** Quota Drive, archive thiếu, quyền tải: liên hệ tác giả; không tự dùng mirror và gọi official.

## Phase 2 — Dataset exploration

- **Objective / Goal:** Hiểu labels, cấu trúc và FPS thật trước extraction.
- **Prerequisites / Dependencies:** Phase 1 có UTA.
- **Files to create:** src/datasets/manifest.py; scripts/explore_dataset.py; notebooks/01_dataset_exploration.ipynb; data/processed/manifest.parquet
- **Files to modify:** docs/Dataset_Access.md.
- **Functions/classes:** build_manifest(root,dataset), validate_manifest(rows).
- **Input → Output:** 45 raw video đã kiểm checksum → manifest schema ở 04 và exploration report.
- **Technical tasks / Implementation notes:** Full UTA có 60 subjects/180 logical recordings nhưng 182 physical files trong ZIP index. Subset hiện tại kiểm **15 subjects/45 unsegmented videos**, mapping 0/5/10 → 0/1/2; fold membership từ official package. Check metadata FPS/duration và decode frame đầu/giữa/cuối từng video; chưa chứng minh constant-FPS hay toàn frame không hỏng.
- **Tests:** tests/test_manifest.py: duplicate ID/path, nhãn lạ, thiếu class, corrupt video giữ error reason; real synthetic AVI và Parquet round trip. YawDD adapter/null-label tests chỉ triển khai khi dataset đó nằm trong phạm vi.
- **Acceptance criteria / Definition of Done:** Mọi raw video có manifest hoặc error reason; folds truy nguyên được. Nếu thiếu membership, custom split gắn nhãn rõ và official comparison bị chặn.
- **Expected result:** manifest schema ở 04 và exploration report. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** FPS container sai hoặc variable; timestamp bất thường, subject ID bị cắt leading zero.

## Phase 3 — Face landmark detection

- **Objective / Goal:** Có detector đúng API và thời gian.
- **Prerequisites / Dependencies:** Phase 0, ít nhất vài clip Phase 2.
- **Files to create:** src/preprocessing/video_reader.py; src/features/landmarks.py; scripts/preview_landmarks.py
- **Files to modify:** src/contracts.py chỉ bổ sung reason enum cần thiết.
- **Functions/classes:** VideoReader.iter_frames/close; FaceLandmarkDetector.detect/close.
- **Input → Output:** Video/FramePacket → LandmarkResult và overlay.
- **Technical tasks / Implementation notes:** Tasks VIDEO, num_faces=1, RGB, increasing ms; reset mỗi video. Resample theo timestamp tối đa20 FPS, không duplicate nguồn thấp.
- **Tests:** tests/test_video_reader.py: source FPS khác nhau, timestamp monotonic. Smoke ba clip và webcam, no-face, disconnect.
- **Acceptance criteria / Definition of Done:** Overlay đúng mặt trên ảnh chưa mirror; no-face output invalid; throughput/drop stats được ghi, close release.
- **Expected result:** LandmarkResult và overlay. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** Landmark vẫn có thể sai khi kính râm; không dùng có-face như quality tuyệt đối.

**Kết quả triển khai Phase 3:** reader/detector/preview đã chạy ba prefix 12 giây của subject 04 (ba class), model blank/no-face, webcam 0 và unavailable-camera error. Overlay đúng topology trên ảnh chưa mirror; có drop/throughput và capture/model release. Regression clock Windows đã được sửa bằng monotonic perf_counter_ns. **Chưa claim nghiệm thu hardware hoàn chỉnh:** webcam hiện không phát hiện mặt trong smoke; chưa quan sát chuyển face→no-face trực tiếp hoặc rút camera vật lý. Bằng chứng và readiness xem [Phase3_Report.md](Phase3_Report.md). Chưa decode toàn video/dataset.


## Phase 4 — EAR extraction

- **Objective / Goal:** Tính EAR hai mắt và mean đúng công thức.
- **Prerequisites / Dependencies:** Phase 3; công thức/mapping ở 06, contract ở 15.
- **Files to create:** src/features/eye.py; tests/test_eye.py
- **Files to modify:** scripts/preview_landmarks.py thêm EAR overlay.
- **Functions/classes:** EyeFeatureExtractor.extract(landmarks) → EyeFeatures.
- **Input → Output:** 478 normalized landmarks + W/H → EAR left/right/mean và validity.
- **Technical tasks / Implementation notes:** Dùng pixel x*W,y*H; mapping giải phẫu ở06, denominator epsilon; mean NaN nếu một mắt invalid. Chưa classifier hay audio ở phase này.
- **Tests:** Synthetic six points: EAR=0.25, scale/translation invariant, ảnh non-square, denominator zero, NaN; mắt mở/nhắm thật trên webcam.
- **Acceptance criteria / Definition of Done:** Unit fixtures đúng số, overlay blink làm EAR giảm; trái/phải không đảo vì mirror.
- **Expected result:** EAR left/right/mean và validity. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** Dùng distance normalized gây sai aspect ratio; số landmark khác schema fail rõ.

**Kết quả Phase4 (2026-10-06):** `eye.py` và41 deterministic tests đã có; pixel/aspect/scale/translation, independent masks, schema/epsilon/closed0 đều qua. Ba prefix12s subject04:723 geometry-valid EAR frames; `runs/phase4/three_clips/preview_report.json`. Trên `04/10.mp4`, ảnh mắt mở0ms EAR mean0.329221 và mắt nhắm6280ms0.022919 đã được xem (`runs/phase4/visual/ear_candidates.json` và ảnh kèm). Actual preview HWND giữ đúng anatomical sides/unmirrored và aspect. Camera có người mở/nhắm mắt chưa được quan sát; recorded transition đạt, **human webcam acceptance vẫn pending**.

## Phase 5 — MAR extraction

- **Objective / Goal:** Đo miệng mà không nhầm mở miệng với nhãn ngủ.
- **Prerequisites / Dependencies:** Phase 3; độc lập Phase 4.
- **Files to create:** src/features/mouth.py; tests/test_mouth.py
- **Files to modify:** scripts/preview_landmarks.py thêm MAR.
- **Functions/classes:** MouthFeatureExtractor.extract(landmarks) → MouthFeatures.
- **Input → Output:** Landmarks → MAR và mouth_valid.
- **Technical tasks / Implementation notes:** Inner lip 78/308, 82/87,13/14,312/317; công thức mean 3 dọc/ ngang ở06. Không hard-code yawn label từ MAR một frame.
- **Tests:** Synthetic MAR=0.5, scaling, zero-width; webcam silent/talk/open mouth.
- **Acceptance criteria / Definition of Done:** Formula thống nhất và N/A khi invalid; overlay tăng khi mở miệng, report talking ambiguity.
- **Expected result:** MAR và mouth_valid. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** Outer/inner lip lẫn; threshold từ công thức khác không áp dụng.

**Kết quả Phase5 (2026-10-06):** `mouth.py` và38 deterministic tests đã có; project formula, non-square/scale/translation, cả3 vertical pairs, invalid/closed0/epsilon và validMAR>1 đều qua. Preview kết hợp EAR/MAR, canonical mouth segments, readable text ở actual640×360/362×640 và finite/null summaries. Full regression **178 passed in6.96s**; geometry và preview reviews không có actionable findings.

Ba prefix12s/cả ba class:723 face/geometry-valid frames (`runs/phase5/three_clips/preview_report.json`), chưa là quality guarantee. Sequential20FPS `04/10.mp4` đến585400ms:11709 face frames, capture/model release; raw lips được xem khép0ms MAR0.005320 → hé mở585400ms0.081579 (`runs/phase5/mouth_sequence/mouth_sequence.json`, `raw_mouth_pair.png`). Đây không phải bằng chứng nói chuyện/ngáp hay nhãn ngủ.

Native blank và15-frame15FPS synthetic AVI full EOF trả tất cả NaN/false → previewN/A/report null (`runs/phase5/no_face/no_face_report.json`), không upsample và release. Webcam0 chạy45s:1326 decoded/821 emitted/505 dropped, cả821 no-face; mọi geometry summary valid0/invalid821/null aggregates (`runs/phase5/webcam/preview_report.json`), không lưu ảnh. **Nói chuyện có xác nhận, human webcam transitions và physical disconnect vẫn blocked** vì không có người/thao tác thiết bị trong session.

QC discovery5FPS decode đủ3 clip subject04, không toàn dataset (`runs/phase5/discovery/discovery.json`). Numeric maximum ở `04/5.mp4` khi quay đầu, không phải observed mouth opening; không gate/classify bằng ratio hoặc có-face. Quality/pose xử lý ở Phase6/7 sau. Prerequisite code để bắt đầu Phase6 đã có; không claim full Phase4/5 hardware acceptance.

## Phase 6 — Head pose estimation

- **Objective / Goal:** Ước lượng pitch/yaw/roll có dấu và quality gate rõ.
- **Prerequisites / Dependencies:** Phase 3, canonical asset Phase 0.
- **Files to create:** src/features/head_pose.py; tests/test_head_pose.py
- **Files to modify:** scripts/preview_landmarks.py thêm pose; configs/preprocessing.yaml K/distortion.
- **Functions/classes:** HeadPoseEstimator.estimate(landmarks) → PoseFeatures.
- **Input → Output:** Six 2D landmarks + canonical3D + K → góc độ, normalized reprojection error.
- **Technical tasks / Implementation notes:** Theo06: solvePnP ITERATIVE, coordinate transform, Rodrigues/Euler; approximate K phải gắn flag. Reject fit fail/reprojection quá cao.
- **Tests:** Synthetic projectPoints với pose known ±10°; resize K; live cúi/quay/nghiêng xác nhận dấu; degenerate points invalid.
- **Acceptance criteria / Definition of Done:** Không đảo dấu hoặc góc 180° khi nhìn thẳng; overlay và synthetic test đạt tolerance 1° không noise; log pose approximation.
- **Expected result:** góc độ, normalized reprojection error. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** Canonical thứ index+1; mouth motion, camera intrinsics sai, Euler discontinuity.

### Thực thi 2026-10-07 — Phase6
Code numerical/preview đã có; sửa canonical reflection thành D=diag(1,−1,−1), signed Euler theo06/15, không offset180. Synthetic neutral/từng trục±10/combined, resize nonuniform, distortion, hull/depth/singular/failure/RMS boundary đã qua regression; Phase6 targeted105 tests green. Native ba subject04 prefixes12s và actual landscape/portrait windows30s đã xem, approximate K và canonical SHA được báo. Webcam45s Phase6 có814 emitted/all no-face, reader/model release. **Live down/up/yaw/roll đủ hai dấu và physical occlusion/transition/disconnect chưa quan sát**, không full hardware PASS. Evidence `runs/phase6/`.


## Phase 7 — Feature extraction pipeline

- **Objective / Goal:** Một pipeline raw dùng chung offline/realtime.
- **Prerequisites / Dependencies:** Phases 3–6.
- **Files to create:** src/features/pipeline.py; src/preprocessing/builder.py; scripts/preprocess.py; tests/test_feature_pipeline.py
- **Files to modify:** configs/preprocessing.yaml quality gates; src/contracts.py storage fields nếu thiếu.
- **Functions/classes:** FeaturePipeline.process/reset/close; FeatureDatasetBuilder.build.
- **Input → Output:** Manifest/video → FeatureSample/Parquet theo14 + metadata/report.
- **Technical tasks / Implementation notes:** Chưa normalize theo subject; label chỉ join builder. Per-video write tạm rồi atomic rename, skip chỉ khi hash/config khớp. Quality gate đo trên sample clip rồi freeze.
- **Tests:** End-to-end một video mỗi class; missing giữ timestamp/null, corrupt video có reason; cùng frame offline/realtime feature tương đương.
- **Acceptance criteria / Definition of Done:** Parquet đọc lại giữ dtype/null, counts/time đúng, hashes/report đầy đủ; không train khi raw extraction chưa qua QC.
- **Expected result:** FeatureSample/Parquet theo14 + metadata/report. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** Schema drift, duplicate cv2 conversion, ghi partial file như complete.

### Thực thi 2026-10-07 — Phase7
`quality.py`, shared `pipeline.py`, working-manifest validator, streaming `builder.py`, measurement/preprocess CLI đã có. Shared contracts/storage columns giữ nguyên; labels chỉ join builder. Working34 rows/12subjects và51_5 partial coverage qua working validation, strict acquisition vẫn reject. Numeric optical profile thật tại `configs/quality_policy_v1.json`:12 reviewed clean giữ12/12,36 degraded controls reject36/36; source/program/asset hashes và exact accepted-report SHA lưu. Không tune drowsiness accuracy.

Native full EOF chọn04_0/04_5/04_10: rows12291/12149/12315, last source timestamps614520/607400/615720ms, tổng36,755. Reopen kiểm schema/null/strict time/index/labels/count/commit hashes; rerun3/3 cached, không load native để extract lại. Native blank15FPS15frames giữ15 null rows, saved5-packet replay/native parity trong1e-6. Actual gated landscape/portrait, manual eyes-only mode, Gaussian blur/dark controls và profile9000ms đã xem. `04_5@416000ms` trong full sequential run mới là no-face/null; không gán việc này riêng cho pose gate hay coi old highMAR là mouth opening.

Final suite **394 passed in12.01s**; hai scoped reviews clean sau sửa preview cleanup isolation (regression RED/GREEN). Windows writable-fsync và numeric35-vs35.0 frozen policy regressions cũng fail-before/pass-after. Webcam Phase7/45s:1327 decoded/806 emitted/521 dropped, all806 no-face, QPC timestamps và release; live human checklist vẫn blocked vì chưa có quan sát người/thiết bị tương tác.

**Coverage risk:** full04_0 left eye2767/12291 vàmouth2873/12291 valid; oblique reason9361 frames dưới raw absolute pose gate. Approximate K/generic face/raw thresholds không là personal calibration; raw extraction/storage QC đạt không chứng minh đủ coverage cho training. Giữ missing/provenance, xem coverage theo source/lớp ởPhase8; không class-driven nới gate. Không chạy all34, download, split/calibrate/train. Evidence `runs/phase7/full_three_clips/`, `resume/`, `native_blank/`, `quality_measurements/`, `quality_surface/`, `visual/`, `webcam/`.


## Phase 8 — Process working dataset snapshot

- **Objective / Goal:** Có processed features đủ để train không đọc raw mỗi epoch.
- **Prerequisites / Dependencies:** Phase 7 và manifest verified.
- **Files to create:** data/processed/raw_features/<video_id>.parquet và companion metadata; data/processed/extraction_status.parquet; runs/<snapshot>/snapshot/{manifest.parquet,preprocessing.yaml,snapshot.json}, report.json, audit.json, coverage.csv.
- **Files to modify:** scripts/preprocess.py resume theo hash chỉ nếu cần; giữ nguyên source manifest `data/processed/manifest.parquet` và `status=ok/error`, ghi extraction/audit status ở sidecar riêng để không trộn acquisition verification với feature quality.
- **Functions/classes:** FeatureDatasetBuilder.build giữ contract Phase7; freeze_snapshot/run_snapshot trong preprocessing/snapshot.py; audit_snapshot trong preprocessing/audit.py; scripts.process_snapshot và scripts.audit_features.
- **Input → Output:** Working manifest snapshot đã freeze (hiện34 verified UTA videos) → feature files + coverage/error report từng source/lớp; không yêu cầu đủ acquisition45 hoặc full180.
- **Technical tasks / Implementation notes:** Sequential trước; checkpoint per video. Freeze source paths/hashes/config, measure storage/time. Snapshot denominator34 và planned-acquisition45 ghi riêng; failure/rejection không bị xóa. Khi thêm nguồn, version snapshot/experiment mới.
- **Tests:** Re-open mỗi output; timestamp strictly increasing, feature bounds/NaN, class counts, spot-check overlay các quality thấp.
- **Acceptance criteria / Definition of Done:** Mỗi video complete hoặc explicit failed; raw không bị sửa; report tỷ lệ valid theo class và subject.
- **Expected result:** feature files + coverage/error report từng video/lớp. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** Disk full, video dài decode chậm, quality lệch lớp gây selection bias.

## Phase 9 — Subject splitting và profile policy

- **Objective / Goal:** Không leakage; khóa P0/P1 trước training.
- **Prerequisites / Dependencies:** Phase 8 và official memberships từ2.
- **Files to create:** src/datasets/splits.py; src/calibration/profile.py; scripts/build_splits.py; data/splits/outer_{0..4}.json; tests/test_splits.py, tests/test_profile.py
- **Files to modify:** configs/training.yaml calibration_mode/fold/split path.
- **Functions/classes:** make_outer_split(folds,test_fold); estimate_profile(samples,mode); transform_sample(sample,profile).
- **Input → Output:** Working manifest/folds/prefix → subject-disjoint train/val/test với cardinality thực tế và P0/P1 profiles/reserved ranges, không hard-code36/12/12 hoặc9/3/3.
- **Technical tasks / Implementation notes:** P0 baseline fit train-only. P1 reserve prefix30–60 giây Alert từng người theo11, failure abstain. Pure profile functions làm tại đây; Phase 16 thêm live manager, tránh dependency vòng.
- **Tests:** Disjoint subject/video; reserved prefix không được sampling; thay test values không đổi population baseline; unknown fold reject.
- **Acceptance criteria / Definition of Done:** 5 outer split descriptions cover toàn bộ working subjects với disjoint roles; eligibility/rejection/blocked folds được ghi riêng, không cần5 valid metrics bằng mọi giá. Profile protocol/schema/hash, snapshot identity và class support lưu rõ. Subject51 thiếu Alert phải abstain P1; fold5 P1 validation/test không có accepted subjects phải blocked/undefined.
- **Expected result:** Subject-disjoint splits theo snapshot hiện có, profiles/reserved ranges và support/coverage report; không claim full60 benchmark. Bằng chứng acceptance trong run log.
- **Possible problems:** Không tìm được official IDs không tự gọi custom là official; không dùng toàn Alert video test làm baseline.

## Phase 10 — Rule-based baseline

- **Objective / Goal:** MVP 1 chạy có thời gian và uncertainty.
- **Prerequisites / Dependencies:** Phases 4–9; webcam có thể dùng P0/P1 profile đã chuẩn.
- **Files to create:** src/features/temporal.py; src/models/rules.py; scripts/webcam_demo.py; tests/test_temporal.py, tests/test_rules.py
- **Files to modify:** configs/preprocessing.yaml event/perclos gates; configs/realtime.yaml rule mode.
- **Functions/classes:** TemporalFeatureExtractor.update/reset; RuleBasedClassifier.predict(temporal_sample).
- **Input → Output:** FeatureSample/profile → causal events/PERCLOS; EAR-only và temporal rule predictions.
- **Technical tasks / Implementation notes:** Áp dụng06 và08; PERCLOS proxy include blinks, dt cap100ms, history60s; no-face không Drowsy. CLI/OpenCV demo tạm, chưa GUI final.
- **Tests:** PERCLOS closed2s/valid10s=0.2; missing excluded; variable FPS; closure event vs blink; webcam chớp nhanh không audio ngay.
- **Acceptance criteria / Definition of Done:** MVP 1: landmark/EAR/MAR và temporal rule status chạy; report hạn chế Low Vigilance và coverage.
- **Expected result:** causal events/PERCLOS; EAR-only và temporal rule predictions. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** Blink FPS thấp; duration bị nối qua mất face; rule thresholds chưa validated.

## Phase 11 — Random Forest baseline

- **Objective / Goal:** MVP 2: ML ba class và subject-independent evaluation.
- **Prerequisites / Dependencies:** Phases 8–10.
- **Files to create:** src/datasets/summaries.py; src/datasets/sequence.py window-index functions; src/models/baseline.py; scripts/train_baseline.py; tests/test_summaries.py
- **Files to modify:** configs/training.yaml RF; src/evaluation/evaluator.py bản metrics core.
- **Functions/classes:** build_window_index(...); summarize_window(...); BaselineClassifier.fit/predict_proba; evaluate_classification(...).
- **Input → Output:** Temporal samples/splits → summary table, RF artifact, val/test metrics.
- **Technical tasks / Implementation notes:** Cùng10s/stride1 và quality gate ở07; train-only median imputer + missing flags. RF300 trees; optional XGBoost sau. Không đưa ID/label vào X.
- **Tests:** No test fit, class order0/1/2, duration/rate đúng; actual train một fold và save/load predictions so khớp.
- **Acceptance criteria / Definition of Done:** MVP 2 có Macro F1/per-class/confusion/support/coverage và split hash; không chỉ accuracy; cold reload chạy.
- **Expected result:** summary table, RF artifact, val/test metrics. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** Overlapping windows tạo thống kê quá lạc quan; class imbalance sau reject.

## Phase 12 — Build sequence dataset

- **Objective / Goal:** Tensor train và webcam cùng semantics.
- **Prerequisites / Dependencies:** Phases 9–11.
- **Files to create:** src/datasets/normalization.py; tests/test_sequence.py, tests/test_normalization.py; data/processed/derived_<hash>/
- **Files to modify:** src/datasets/sequence.py thêm SequenceDataset; configs/training.yaml ordered feature_names.
- **Functions/classes:** fit_scaler(train_unique_timesteps); SequenceDataset.__len__/__getitem__.
- **Input → Output:** Temporal features/window index → float32[100,16], label, metadata riêng.
- **Technical tasks / Implementation notes:** Scaler fit unique train steps; invalid=0 sau scale + mask; không pad/forward-fill qua gap; preserve causal order. P0/P1 transformed features đã tạo bằng profile pure functions Phase 9.
- **Tests:** 100 steps đúng10s; reject >20% missing, gap>1 giây/current invalid; reserved prefix; NaN không vào tensor; future change không đổi past sample.
- **Acceptance criteria / Definition of Done:** Dataset load vài batch đúng shapes/classes/order và training không mở raw video; metadata schema/hash đồng nhất.
- **Expected result:** float32[100,16], label, metadata riêng. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** Lặp window làm fit scaler bias; nhầm timestep với frame native.

## Phase 13 — Train LSTM

- **Objective / Goal:** Train và export model temporal thật.
- **Prerequisites / Dependencies:** Phase 12; RF baseline11.
- **Files to create:** src/models/lstm.py; src/models/bundle.py; src/training/trainer.py; scripts/train_lstm.py; tests/test_model_bundle.py
- **Files to modify:** configs/training.yaml LSTM/loss/optimizer/seed.
- **Functions/classes:** LSTMClassifier.forward; ModelTrainer.fit; ModelBundle.load/validate_schema.
- **Input → Output:** Dataset loaders → best checkpoint/history/scaler/config.
- **Technical tasks / Implementation notes:** Architecture08, Adam/loss09; logits chưa softmax; không giữ hidden state qua window; early stop val Macro F1. Windows loader workers0 trước.
- **Tests:** Synthetic forward/backward loss finite; training một fold thật, loss trace, reload model predicts same within tolerance; schema mismatch reject.
- **Acceptance criteria / Definition of Done:** Checkpoint chạy được trên batch test lạnh, đủ metadata; report val metrics và epoch selected, không hứa hơn RF.
- **Expected result:** best checkpoint/history/scaler/config. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** Overfit60 subjects, silent feature-order mismatch, dropout LSTM1 layer không tác dụng.

## Phase 14 — Model evaluation

- **Objective / Goal:** Bảng kết quả đáng tin trước realtime.
- **Prerequisites / Dependencies:** Phases 11–13.
- **Files to create:** scripts/evaluate.py; runs/<id>/metrics.json, predictions.parquet, confusion_matrix.png; tests/test_evaluation.py
- **Files to modify:** src/evaluation/evaluator.py fold/video/subject metrics.
- **Functions/classes:** ModelEvaluator.evaluate/aggregate_folds.
- **Input → Output:** Frozen RF/rules/LSTM + outer splits → metrics theo10.
- **Technical tasks / Implementation notes:** Chạy5 folds cùng policies; calibration P0/P1 riêng, video aggregation, conditional metrics+abstention; bootstrap theo subject.
- **Tests:** Known confusion fixture, absent-class undefined AUC, unique videos, verify all outer subjects tested once; run đủ 5 folds thật reports.
- **Acceptance criteria / Definition of Done:** Accuracy/Macro/Weighted F1/per-class/recall Low Vigilance/recall Drowsy/balanced accuracy/support/coverage có đủ; mean±std và confusion gộp.
- **Expected result:** metrics theo10. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** Không dùng test để chọn checkpoint; paper video accuracy không so với window accuracy trực tiếp.

## Phase 15 — Ablation experiments

- **Objective / Goal:** Biết feature và temporal order nào có giá trị.
- **Prerequisites / Dependencies:** Phase 14 và frozen experiment matrix18.
- **Files to create:** scripts/run_ablation.py; runs/ablation_<id>/results.csv
- **Files to modify:** configs/training.yaml feature subsets theo experiment; src/datasets/summaries.py subset selection.
- **Functions/classes:** run_experiment(spec), compare_paired_subjects(results).
- **Input → Output:** Same splits/modes → A–E bảng Macro F1/recall Drowsy/recall Low Vigilance/latency.
- **Technical tasks / Implementation notes:** A EAR; B+PERCLOS; C+MAR; D+pose; E D+personalization. A–D P0, E P1; quality masks giữ theo group và cohort chung; retrain scaler/model từng variant.
- **Tests:** No mismatched features, same seeds/folds/cohort; thật chạy experiments đã chọn; report coverage toàn tập riêng.
- **Acceptance criteria / Definition of Done:** Bảng gain/loss và CI, giữ hoặc bỏ feature theo evidence; không tạo kết quả giả cho thử nghiệm chưa chạy.
- **Expected result:** A–E bảng Macro F1/recall Drowsy/recall Low Vigilance/latency. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** Personalization cần calibration data là additional information, không chỉ thêm layer; compare P1 cohort riêng.

## Phase 16 — Personalized calibration realtime

- **Objective / Goal:** Có lifecycle calibration rõ và không fine-tune.
- **Prerequisites / Dependencies:** Profile functions9, checkpoint mode13, policy11.
- **Files to create:** src/calibration/manager.py; tests/test_calibration.py
- **Files to modify:** scripts/webcam_demo.py live calibration; configs/realtime.yaml durations.
- **Functions/classes:** CalibrationManager.start/update/finish/reset.
- **Input → Output:** Stream FeatureSample + confirmed Alert → CalibrationProfile/status.
- **Technical tasks / Implementation notes:** Dùng estimators Phase 9,30 giây / ít nhất 20 giây hợp lệ / timeout 60 giây; state machine, retry, freeze profile, invalidate khi camera/schema đổi. Không đọc model prediction để tự xác nhận người tỉnh.
- **Tests:** Lost face timeout, unstable EAR fail, retry reset, baseline mắt trái/right khác; actual30s webcam và profile applied.
- **Acceptance criteria / Definition of Done:** UI/CLI thấy progress/complete/fail; P1 chỉ inference khi profile valid; P0 chọn đúng model riêng.
- **Expected result:** CalibrationProfile/status. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** Đang buồn ngủ lúc calibration; insufficient blink stats; không auto-adapt baseline vào sleep.

## Phase 17 — Realtime webcam inference

- **Objective / Goal:** MVP 3 phần inference: stream chạy không backlog.
- **Prerequisites / Dependencies:** Phases 13,16 và temporal10.
- **Files to create:** src/realtime/buffer.py; src/realtime/detector.py; tests/test_buffer.py, tests/test_detector.py
- **Files to modify:** scripts/webcam_demo.py model mode; configs/realtime.yaml cadence/device.
- **Functions/classes:** PredictionBuffer.append/get_window/reset; DrowsinessDetector.process/reset_session/close.
- **Input → Output:** FramePackets → DetectionResult mỗi1s sau100 steps, status mỗi sample.
- **Technical tasks / Implementation notes:** Latest capture slot, timestamps monotonic, causal10 Hz; no raw per-frame classify, no repeated model load; reject poor current sample. Smoothing chưa tích hợp đến18.
- **Tests:** Replay variable timestamps, gap/no-face/restart; run webcam10 phút, observe probabilities/cadence/drop/memory.
- **Acceptance criteria / Definition of Done:** Live probabilities3 class có timestamp/quality, warmup/unknown rõ; không backlog lớn dần, correct checkpoint mode.
- **Expected result:** DetectionResult mỗi1s sau100 steps, status mỗi sample. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** Camera read buffering, CPU MediaPipe bottleneck, CUDA copy overhead.

## Phase 18 — Prediction smoothing

- **Objective / Goal:** Giảm rung mà không giấu latency.
- **Prerequisites / Dependencies:** Phase 17; policy12.
- **Files to create:** src/realtime/smoother.py; tests/test_smoother.py
- **Files to modify:** src/realtime/detector.py tích hợp; configs/realtime.yaml smoothing.
- **Functions/classes:** PredictionSmoother.update/reset.
- **Input → Output:** Raw Prediction → mean của 3 probabilities hoặc null.
- **Technical tasks / Implementation notes:** Cadence1s,3 valid mẫu; stale gap≥2 giây clear; UI immediate unavailable khi current bad, không dùng mean cũ.
- **Tests:** Alternating probability arrays có mean đúng, not enough samples, NaN reject, timeout/restart clear; actual replay raw vs smooth.
- **Acceptance criteria / Definition of Done:** Prediction smooth ổn và có raw log; báo cáo độ trễ thêm, no stale result accepted.
- **Expected result:** mean của 3 probabilities hoặc null. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** Mean smoothing che transition ngắn; không chọn threshold bằng test.

## Phase 19 — Warning system

- **Objective / Goal:** MVP 3 hoàn chỉnh có warning nhiều mức.
- **Prerequisites / Dependencies:** Phase 18.
- **Files to create:** src/alerts/manager.py; tests/test_alerts.py; models/assets/warning.wav hoặc UI resources âm thanh có license
- **Files to modify:** scripts/webcam_demo.py audio commands; configs/realtime.yaml enter/exit/dwell/cooldown.
- **Functions/classes:** AlertManager.update/set_muted/reset.
- **Input → Output:** DetectionResult/timestamp → AlertDecision/audio command.
- **Technical tasks / Implementation notes:** State machine ở tài liệu 12, Drowsy precedence, dwell bằng giây valid, strong sau 10 giây, cooldown 15 giây, mute audio-only, technical no-face riêng.
- **Tests:** Fake clock boundary2/3/5/10/15 giây, unknown breaks timer, escalation bypass cooldown thông thường, probability dao động; loa thật smoke trong demo an toàn.
- **Acceptance criteria / Definition of Done:** Low UI/Drowsy audio/strong warning đúng episode; không single prediction alarm; no audio AI từ stale.
- **Expected result:** AlertDecision/audio command. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** Âm thanh driver/Qt plugin; không hứa an toàn chỉ vì beep hoạt động.

## Phase 20 — PySide6 UI

- **Objective / Goal:** Có desktop surface cho người dùng.
- **Prerequisites / Dependencies:** Phases 17–19 và layout ở tài liệu 13.
- **Files to create:** src/realtime/camera_worker.py; src/ui/main_window.py; main.py
- **Files to modify:** configs/realtime.yaml UI/audio; scripts/webcam_demo.py giữ CLI debug không duplicate pipeline.
- **Functions/classes:** CameraWorker.start/request_stop/request_calibration + signals; MainWindow.render/start_session/stop_session.
- **Input → Output:** UiSnapshot/buttons → preview/metrics/probabilities/status/audio.
- **Technical tasks / Implementation notes:** QObject worker, latest snapshot render 10 Hz; QImage owned bytes; preview-only mirror; stop interruption flag, không terminate; audio Qt main thread.
- **Tests:** Cửa sổ thật Start/Stop 5 lần, mute, calibrate, no-face, camera unplug, resize, close; không cần unit GUI chi tiết.
- **Acceptance criteria / Definition of Done:** Tất cả trường UI yêu cầu hiển thị đúng N/A/status; không freeze, camera release khi đóng; ảnh demo có consent.
- **Expected result:** preview/metrics/probabilities/status/audio. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** Signal queue flood, array lifetime, stop slot không chạy trong blocking loop.

## Phase 21 — Integration và generalization

- **Objective / Goal:** Final build có offline/realtime và external report thống nhất.
- **Prerequisites / Dependencies:** Phases 14–20; NTHU access nếu external test.
- **Files to create:** src/evaluation/replay.py; scripts/external_test.py; tests/test_integration.py
- **Files to modify:** main.py startup bundle validation; README.md run instructions; configs paths.
- **Functions/classes:** replay_session(...); evaluate_external(...).
- **Input → Output:** Video đã lưu/checkpoint/NTHU annotations → replay/live result + external binary report.
- **Technical tasks / Implementation notes:** Same extraction/normalization, offline replay equivalence; NTHU frozen P0 binary proxy theo tài liệu 04/10, không tự tạo lớp Low Vigilance. External access blocked không gọi Final generalization complete.
- **Tests:** Cùng clip offline vs replay matching valid features/probabilities within tolerance; cold start main.py; external scenarios đủ subset accessible.
- **Acceptance criteria / Definition of Done:** End-to-end từ raw/file hoặc camera đến warning/UI; external metrics+coverage thật và nêu modality; consent/quyền sử dụng đúng.
- **Expected result:** replay/live result + external binary report. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** NTHU labels test không public; chỉ evaluate subset labeled và ghi giới hạn, không giả kết quả.

## Phase 22 — Performance optimization

- **Objective / Goal:** Đạt realtime bằng đo bottleneck trước sửa.
- **Prerequisites / Dependencies:** Phase 21 có app đã tích hợp; cách đo ở tài liệu 10.
- **Files to create:** scripts/benchmark.py; runs/benchmark_<id>/report.json
- **Files to modify:** Module bottleneck được đo, không sửa mọi nơi; configs device/FPS nếu experiment đã chốt.
- **Functions/classes:** benchmark_session(duration_s,source,device).
- **Input → Output:** 10 phút webcam/replay → FPS,các percentile latency,CPU/GPU/RAM/VRAM/drop.
- **Technical tasks / Implementation notes:** Profile decode/landmarker/UI trước, giảm resolution hoặc giới hạn FPS nếu cần; CPU vs GPU forward bao gồm copy; không đổi feature cadence mà không retrain.
- **Tests:** Before/after cùng source/hardware; chạy 10 phút, repeat Start/Stop, inspect peak memory and UI responsiveness.
- **Acceptance criteria / Definition of Done:** Target ở tài liệu 10 đạt hoặc report rõ phần không đạt và điều chỉnh có evidence; feature/prediction equivalence giữ trong tolerance.
- **Expected result:** FPS,các percentile latency,CPU/GPU/RAM/VRAM/drop. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** Thermal throttle, GPU utilization thấp không là lỗi; capture latency khác model latency.

## Phase 23 — Testing và release checks

- **Objective / Goal:** Có evidence regression đầy đủ, không chỉ demo đẹp.
- **Prerequisites / Dependencies:** Phases 0–22; tests đã viết từ mỗi phase.
- **Files to create:** tests/test_release_scenarios.py; runs/release_<id>/checklist.md
- **Files to modify:** tests còn thiếu behavioral case; README.md limitations; docs/CHANGELOG.md.
- **Functions/classes:** run_release_checks(); interfaces không tự đổi.
- **Input → Output:** Unit/integration fixtures và demo hardware → test report+bằng chứng smoke thủ công.
- **Technical tasks / Implementation notes:** Strategy ở tài liệu 17; không đến phase này mới viết toàn tests. Test seed/split/profile/unknown/audio shutdown; tùy chọn một giờ Alert để mục tiêu cảnh báo giả.
- **Tests:** Run toàn bộ pytest suite once sau integration; actual main.py môi trường mới, video/file + webcam, đường xử lý lỗi, ít nhất một giờ Alert nếu muốn claim không quá hai audio alarm/giờ.
- **Acceptance criteria / Definition of Done:** Không test đang fail; manuals có timestamps/log; chưa đủ thời gian kiểm tra cảnh báo giả ghi chưa xác minh, không tự pass target.
- **Expected result:** test report+bằng chứng smoke thủ công. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** Thiếu hardware/dataset permissions không thay bằng mock rồi claim runtime verified.

## Phase 24 — Final report và demo preparation

- **Objective / Goal:** Bàn giao có thể bảo vệ và tái lập.
- **Prerequisites / Dependencies:** Phase 23; tất cả kết quả thật.
- **Files to create:** reports/final_report.md; reports/demo_checklist.md; presentation có ảnh hợp lệ
- **Files to modify:** README.md run/reproduce; docs/CHANGELOG.md milestone cuối.
- **Functions/classes:** Không thêm model/module.
- **Input → Output:** Metrics/ablation/external/runtime evidence → final report và demo.
- **Technical tasks / Implementation notes:** Trích papers/dataset; phân biệt proposal vs measured; live demo safe, video dự phòng consent; không điều khiển xe. Giải thích Low Vigilance/nhãn yếu/calibration-assisted.
- **Tests:** Một teammate clone mới + lock/config/model từ kênh riêng chạy smoke; verify links/permissions/reproduce một fold.
- **Acceptance criteria / Definition of Done:** MVP 1/2/3 và Final evidence, limitations rõ, instructions agent dùng Phase N không cần thiết kế lại.
- **Expected result:** final report và demo. Lưu bằng chứng acceptance trong run log để teammate kiểm tra lại.
- **Possible problems:** Accuracy thấp hoặc thiếu quyền truy cập external data phải trình bày, không chọn chỉ fold tốt nhất hoặc giả result.

