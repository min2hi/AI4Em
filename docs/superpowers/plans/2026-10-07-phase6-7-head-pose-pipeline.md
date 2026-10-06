# Phase 6–7: Head Pose Estimation & Feature Extraction Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement head-pose estimation (Phase 6) and unify all extractors into a single FeaturePipeline + FeatureDatasetBuilder for offline/realtime use (Phase 7).

**Architecture:** HeadPoseEstimator uses OpenCV solvePnP with 6 MediaPipe landmarks + canonical 3D model to produce pitch/yaw/roll in degrees. FeaturePipeline composes landmarks→eye+mouth+pose into FeatureSample. FeatureDatasetBuilder iterates manifest videos, writes per-video Parquet with atomic rename, produces extraction metadata/report.

**Tech Stack:** Python 3.12, OpenCV (solvePnP/projectPoints/Rodrigues), NumPy, PyArrow Parquet, MediaPipe 478 landmarks.

## Global Constraints

- All thresholds, mappings, and formulas from `docs/06_Feature_Engineering.md` — never invent new values.
- Contracts from `src/contracts.py` — use existing `PoseFeatures`, `FeatureSample`, `LandmarkResult`, etc.
- Config from `configs/preprocessing.yaml` — read `canonical_model_path`, `epsilon`, `max_reprojection_error_norm`.
- Follow existing patterns in `src/features/eye.py` and `src/features/mouth.py` exactly (validation, NaN semantics, epsilon, pixel-space).
- OBJ vertex indexing: MediaPipe landmark index `i` → OBJ vertex line `i+1` (0-indexed in parsed vertex list: `vertices[i]`).
- Coordinate transform: canonical OBJ is y-up; apply `diag(1, -1, 1)` to convert to face coordinate system before solvePnP.
- Sign convention (doc06): pitch>0 = head tilts down, yaw>0 = turns right in image, roll>0 = clockwise.
- Euler decomposition: `R = Rz(roll) @ Ry(yaw) @ Rx(pitch)`, angles in degrees, wrap ±180.
- Approximate K: `fx=fy=max(W,H), cx=W/2, cy=H/2, distortion=0`.
- Quality gate: reject when solvePnP fails or `reprojection_error_norm > 0.03 * diagonal`.
- Parquet storage schema from `docs/14_Project_Structure.md` lines 42-46.
- Feature order from `docs/07_Temporal_Modeling.md`: 16 channels total.
- Never commit video/face images; tests use synthetic fixtures.

---

### Task 1: HeadPoseEstimator — solvePnP + Euler decomposition

**Files:**
- Create: `src/features/head_pose.py`
- Create: `tests/test_head_pose.py`
- Test: `tests/test_head_pose.py`

**Interfaces:**
- Consumes: `LandmarkResult` from `src/contracts.py`, `PoseFeatures` from `src/contracts.py`
- Produces: `HeadPoseEstimator` class with `estimate(result: LandmarkResult) -> PoseFeatures`

**Details:**

The six 2D landmark indices for pose: `[1, 152, 33, 263, 61, 291]` (nose tip, chin, left eye outer, right eye outer, left mouth corner, right mouth corner).

The corresponding 3D points come from the canonical OBJ file (`models/assets/canonical_face_model.obj`). Parse only `v` lines. MediaPipe landmark index `i` maps to the `i`-th vertex in the parsed list (0-indexed), which is OBJ vertex `i+1`. Extract the 6 points for indices `[1, 152, 33, 263, 61, 291]`.

**Coordinate transform:** The canonical model has y pointing up. Doc06 says apply `diag(1, -1, 1)` to get face-space (x right, y down, z forward). So multiply the y-coordinate by -1.

**Camera intrinsics (approximate):**
```python
fx = fy = max(W, H)  # float
cx, cy = W / 2.0, H / 2.0
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
dist_coeffs = np.zeros(4, dtype=np.float64)
```

**solvePnP:**
```python
success, rvec, tvec = cv2.solvePnP(
    canonical_3d,  # (6, 1, 3) float64
    image_2d,      # (6, 1, 2) float64, pixel coords = normalized * (W, H)
    K, dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE
)
```

**Reprojection error:**
```python
projected, _ = cv2.projectPoints(canonical_3d, rvec, tvec, K, dist_coeffs)
error = np.linalg.norm(projected.reshape(-1, 2) - image_2d.reshape(-1, 2), axis=1).mean()
diagonal = math.sqrt(W * W + H * H)
reprojection_error_norm = error / diagonal
```
Reject if `reprojection_error_norm > max_reprojection_error_norm` (0.03 from config).

**Euler extraction:**
```python
R, _ = cv2.Rodrigues(rvec)
# R = Rz(roll) @ Ry(yaw) @ Rx(pitch)
# Extract: pitch = atan2(R[2,1], R[2,2]), yaw = atan2(-R[2,0], sqrt(R[2,1]^2+R[2,2]^2)), roll = atan2(R[1,0], R[0,0])
pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
yaw = math.degrees(math.atan2(-R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2)))
roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
```

**Verify signs** with synthetic test using `cv2.projectPoints` with a known rotation.

**Constructor:** `HeadPoseEstimator(canonical_obj_path: str | Path, max_reprojection_error_norm: float)`.
- Parse canonical OBJ once at init (extract 6 3D points, apply y-flip).
- Store `canonical_3d` as `(6, 1, 3)` float64 array.

**extract method signature:** `estimate(self, result: LandmarkResult) -> PoseFeatures`
- If no face / no points → return `PoseFeatures(NaN, NaN, NaN, False, NaN)`.
- Validate 478×3 float array, positive integer image_size (same as eye.py).
- Extract 6 2D pixel landmarks. If any out-of-range or non-finite → invalid.
- Run solvePnP. If `not success` → invalid.
- Compute reprojection error. If exceeds threshold → invalid.
- Compute Euler angles.
- Return `PoseFeatures(pitch, yaw, roll, True, reprojection_error_norm)`.

**Tests (in `tests/test_head_pose.py`):**

1. **Synthetic known-pose round-trip at ±10°:** Create a known rotation matrix for pitch=10°, yaw=-5°, roll=3°. Use `cv2.projectPoints` with the canonical 3D points and an approximate K to generate 2D points. Feed them through the estimator. Assert recovered angles within 1° tolerance.

2. **Sign convention verification:** Test that pitch>0 corresponds to looking down, yaw>0 to turning right, roll>0 to clockwise rotation. Create projected points for pitch=+15° (down), yaw=+20° (right), roll=+10° (cw) and verify signs.

3. **Scale/resize K:** Same 3D points projected at 640×480 and 320×240 should produce same angles within tolerance.

4. **No face / None points → invalid output.**

5. **Degenerate/coincident points → invalid or solvePnP failure handled.**

6. **High reprojection error → invalid.**

7. **Constructor validation:** Bad path, bad threshold.

8. **Wrong landmark array shape/dtype raises ValueError.**

- [ ] **Step 1:** Write tests in `tests/test_head_pose.py`
- [ ] **Step 2:** Run tests — verify they fail (module doesn't exist)
- [ ] **Step 3:** Implement `src/features/head_pose.py`
- [ ] **Step 4:** Run tests — verify they pass
- [ ] **Step 5:** Commit `git add src/features/head_pose.py tests/test_head_pose.py && git commit -m "feat(phase6): head pose estimator with solvePnP + Euler decomposition"`

---

### Task 2: Integrate head pose into preview_landmarks.py and update config

**Files:**
- Modify: `scripts/preview_landmarks.py` — add pose overlay and pose summary
- Modify: `configs/preprocessing.yaml` — add K/distortion documentation comment if needed (values already present: `canonical_model_path`, `max_reprojection_error_norm`)
- Test: Manual smoke test via script execution

**Interfaces:**
- Consumes: `HeadPoseEstimator` from Task 1, existing `draw_overlay`, `run_source`, `main` from `scripts/preview_landmarks.py`
- Produces: Updated `draw_overlay` that accepts optional `pose: PoseFeatures` param; updated `run_source` that runs pose estimation; report includes pose summaries

**Details:**

1. **Import:** Add `from src.features.head_pose import HeadPoseEstimator` and `from src.contracts import PoseFeatures`.

2. **`draw_overlay`:** Add optional `pose: PoseFeatures | None = None` parameter. When pose is provided and valid, draw a text line showing `P:{pitch:.1f} Y:{yaw:.1f} R:{roll:.1f}` and `reproj:{error:.4f}`. When invalid, show `Pose: N/A`. Optionally draw a small projected nose axis line (just forward direction) using the rvec/tvec — but this is NOT required; text overlay is sufficient.

3. **`run_source`:** Create `HeadPoseEstimator(config["canonical_model_path"], config["max_reprojection_error_norm"])` alongside eye/mouth extractors. After eye/mouth extraction, call `pose = pose_estimator.estimate(result)`. Pass `pose=pose` to `draw_overlay`. Add pose summary stats (pitch, yaw, roll as separate geometry summaries). Update `summaries` dict to include `"pitch"`, `"yaw"`, `"roll"`.

4. **`main`:** Update report `phase` from 5 to 6. Add `"canonical_model_path"` and `"max_reprojection_error_norm"` to the report metadata.

5. **`report_dir` default:** Change from `runs/phase5` to `runs/phase6`.

- [ ] **Step 1:** Modify `scripts/preview_landmarks.py` to integrate pose
- [ ] **Step 2:** Run `python -m scripts.preview_landmarks --video <sample_clip> --seconds 5 --headless --save-overlay --report-dir runs/phase6` on an actual video clip to verify pose values appear in report
- [ ] **Step 3:** Commit `git add scripts/preview_landmarks.py && git commit -m "feat(phase6): integrate head pose into preview overlay and report"`

---

### Task 3: FeaturePipeline — compose extractors into unified per-frame feature extraction

**Files:**
- Create: `src/features/pipeline.py`
- Create: `tests/test_feature_pipeline.py`
- Test: `tests/test_feature_pipeline.py`

**Interfaces:**
- Consumes: `FaceLandmarkDetector` from `src/features/landmarks.py`, `EyeFeatureExtractor` from `src/features/eye.py`, `MouthFeatureExtractor` from `src/features/mouth.py`, `HeadPoseEstimator` from `src/features/head_pose.py`, `FramePacket`/`LandmarkResult`/`EyeFeatures`/`MouthFeatures`/`PoseFeatures`/`FeatureSample` from `src/contracts.py`
- Produces: `FeaturePipeline` class with `process(packet: FramePacket) -> FeatureSample`, `reset()`, `close()`

**Details:**

```python
class FeaturePipeline:
    def __init__(self, config: dict):
        # Create detector, eye_extractor, mouth_extractor, pose_estimator from config
        self._detector = FaceLandmarkDetector(config)
        self._eye = EyeFeatureExtractor(config["epsilon"])
        self._mouth = MouthFeatureExtractor(config["epsilon"])
        self._pose = HeadPoseEstimator(config["canonical_model_path"], config["max_reprojection_error_norm"])

    def process(self, packet: FramePacket) -> FeatureSample:
        result = self._detector.detect(packet)
        eyes = self._eye.extract(result)
        mouth = self._mouth.extract(result)
        pose = self._pose.estimate(result)
        return FeatureSample(
            timestamp_ms=packet.timestamp_ms,
            ear_left=eyes.ear_left,
            ear_right=eyes.ear_right,
            ear_mean=eyes.ear_mean,
            mar=mouth.mar,
            pitch=pose.pitch,
            yaw=pose.yaw,
            roll=pose.roll,
            face_detected=result.face_detected,
            left_eye_valid=eyes.left_eye_valid,
            right_eye_valid=eyes.right_eye_valid,
            mouth_valid=mouth.mouth_valid,
            pose_valid=pose.pose_valid,
            reprojection_error_norm=pose.reprojection_error_norm,
            source_id=packet.source_id,
            frame_index=packet.frame_index,
        )

    def reset(self):
        """Reset detector session (new video). Create fresh detector."""
        self._detector.close()
        # Reconstruct detector for new video session
        self._detector = FaceLandmarkDetector(self._config)

    def close(self):
        self._detector.close()
```

**Note on `reset()`:** The landmark detector needs a new instance per video (MediaPipe VIDEO mode requires increasing timestamps). Store the config dict to recreate the detector on reset.

**Tests:**

1. **Smoke test with synthetic frame:** Create a 100×100 black image packet, run through pipeline, verify FeatureSample has expected invalid/NaN fields (no face in black image).

2. **Contract check:** Verify FeatureSample fields match contracts.py schema.

3. **Reset creates fresh detector session** (timestamps can restart from 0 after reset without error).

4. **Close releases resources** (no error on double-close; detector stats available).

- [ ] **Step 1:** Write tests in `tests/test_feature_pipeline.py`
- [ ] **Step 2:** Run tests — verify they fail
- [ ] **Step 3:** Implement `src/features/pipeline.py`
- [ ] **Step 4:** Run tests — verify they pass
- [ ] **Step 5:** Commit `git add src/features/pipeline.py tests/test_feature_pipeline.py && git commit -m "feat(phase7): unified FeaturePipeline composing all extractors"`

---

### Task 4: FeatureDatasetBuilder — offline batch extraction to Parquet

**Files:**
- Create: `src/preprocessing/builder.py`
- Create: `scripts/preprocess.py`
- Test: `tests/test_feature_pipeline.py` (extend with builder tests)

**Interfaces:**
- Consumes: `FeaturePipeline`, `VideoReader`, manifest Parquet (from `data/processed/manifest.parquet`), config
- Produces: `FeatureDatasetBuilder` class with `build(manifest_path, output_dir) -> report_dict`; `scripts/preprocess.py` CLI entry point

**Details:**

`FeatureDatasetBuilder.build(manifest_path, output_dir, config)`:

1. Read manifest Parquet → list of video rows with `video_id, path, label_id, subject_id, dataset_name, fold, status`.
2. Filter only `status == "verified"` rows.
3. For each video:
   a. Check if `output_dir/<video_id>.parquet` exists with matching metadata (source hash + config hash → skip).
   b. Create `FeaturePipeline(config)`.
   c. Create `VideoReader()`.
   d. Iterate frames via `iter_frames(path, config["landmark_target_fps"])`.
   e. For each frame, `pipeline.process(packet)` → collect `FeatureSample` list.
   f. Convert to PyArrow table with schema: `timestamp_ms:int64, frame_index:int32, ear_left:float32, ear_right:float32, ear_mean:float32, mar:float32, pitch:float32, yaw:float32, roll:float32, face_detected:bool, left_eye_valid:bool, right_eye_valid:bool, mouth_valid:bool, pose_valid:bool, reprojection_error_norm:float32, label_id:int8, label_source:string`.
   g. Write to temp file `<video_id>.parquet.tmp`, then atomic rename to `<video_id>.parquet`.
   h. Write metadata companion `<video_id>.metadata.json` with schema_version, source_hash, config_hash, sampling stats, failure reasons.
   i. Close pipeline and reader.
   j. Record per-video result in report.
4. Return report dict with per-video status, counts, timing, coverage.

**Atomic write:** Write to `.tmp` then `os.replace()` for crash safety. Never mark partial file as complete.

**Skip logic:** If output Parquet exists and its companion `.metadata.json` has matching `source_hash` (SHA256 of source video) and `config_hash` (SHA256 of serialized config), skip extraction. Otherwise re-extract.

**Error handling:** Per-video try/except; record error in report; continue to next video.

**`scripts/preprocess.py`:**
```
python -m scripts.preprocess [--config configs/preprocessing.yaml] [--video-id <specific>] [--report-dir runs/preprocessing_<timestamp>]
```

**Tests (added to `tests/test_feature_pipeline.py` or separate file):**

1. **End-to-end with synthetic AVI:** Create a 2-second synthetic AVI (solid color frames), write a minimal manifest Parquet pointing to it, run builder, verify output Parquet exists with correct schema/dtypes, timestamps strictly increasing.

2. **Missing values preserved:** No-face frames produce NaN features with validity=False.

3. **Atomic write:** If extraction is interrupted (simulate), no partial `.parquet` file exists (only `.tmp`).

4. **Skip on matching hash:** Running builder twice with same config/source skips second run.

5. **Corrupt video:** Builder records error and continues.

- [ ] **Step 1:** Write builder tests
- [ ] **Step 2:** Run tests — verify they fail
- [ ] **Step 3:** Implement `src/preprocessing/builder.py`
- [ ] **Step 4:** Implement `scripts/preprocess.py`
- [ ] **Step 5:** Run tests — verify they pass
- [ ] **Step 6:** Commit `git add src/preprocessing/builder.py scripts/preprocess.py tests/test_feature_pipeline.py && git commit -m "feat(phase7): FeatureDatasetBuilder with Parquet output and preprocess CLI"`

---

### Task 5: Update contracts.py if needed, update configs, update CHANGELOG

**Files:**
- Modify: `src/contracts.py` — verify FeatureSample has all fields; add if any missing
- Modify: `configs/preprocessing.yaml` — ensure all Phase 6/7 config keys present
- Modify: `docs/CHANGELOG.md` — add Phase 6 and 7 entries

**Interfaces:**
- Consumes: All modules from Tasks 1-4
- Produces: Updated config/contracts/docs

**Details:**

1. Verify `FeatureSample` in `src/contracts.py` has fields: `timestamp_ms, ear_left, ear_right, ear_mean, mar, pitch, yaw, roll, face_detected, left_eye_valid, right_eye_valid, mouth_valid, pose_valid, reprojection_error_norm, source_id, frame_index`. Check against doc14 raw row. Add any missing fields.

2. Verify `configs/preprocessing.yaml` has: `canonical_model_path`, `max_reprojection_error_norm`, `epsilon`, `output_dir`, `manifest_path`. All present in current config already.

3. CHANGELOG entries for Phase 6 (head pose) and Phase 7 (pipeline + builder).

- [ ] **Step 1:** Read current contracts.py FeatureSample fields and compare with schema
- [ ] **Step 2:** Make any needed additions
- [ ] **Step 3:** Update CHANGELOG.md
- [ ] **Step 4:** Run full test suite: `pytest tests/ -v`
- [ ] **Step 5:** Commit `git add src/contracts.py configs/preprocessing.yaml docs/CHANGELOG.md && git commit -m "docs: update contracts and changelog for Phase 6-7"`
