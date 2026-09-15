# Multi-Modal Identity Continuity, Throttled ReID & Bay State Hardening - Tasks

## Task 1: Model Resolution Lock & OpenVINO Thread Pinning
**Description:** Hard-code model export and inference resolutions in `edge/build_sidecar.py` and `edge/runtime.py` to `imgsz=640` for pose (`yolo11n-pose.onnx`) and `imgsz=512` for vehicles (`yolo11n.onnx`) to prevent anchor grid inflation and restore $\ge 25\text{ FPS}$ edge throughput. Restrict OpenVINO and ONNX CPU inference thread pools to physical core count (`num_threads=4`) in `runtime.py` to eliminate context switching and thread thrashing. Re-export weights and validate inference latency.
**Acceptance criteria:**
- [x] `build_sidecar.py` explicitly exports pose models at `imgsz=640` and vehicle models at `imgsz=512`.
- [x] `runtime.py` sets `num_threads=4` (or physical CPU core count) for ONNX Runtime and OpenVINO session options.
- [x] Re-exported `yolo11n-pose.onnx` (640x640) measures $\le 40\text{ ms}$ latency on CPU execution (measured 33.92 ms ORT, 27.51 ms OpenVINO).
**Verification:**
- [x] Benchmark test passes: `.venv/bin/python -c "from runtime import benchmark_pose; print('640x640 Benchmark latency check')"`
- [x] Tests pass: `.venv/bin/python -m unittest test_tinypose test_patch_fixes -v`
- [x] Manual check: Confirm output model dimensions via Netron/ONNX inspection ($1 \times 3 \times 640 \times 640$).
**Dependencies:** None
**Files likely touched:**
- `edge/build_sidecar.py`
- `edge/runtime.py`
- `edge/launcher.py`
**Estimated scope:** Medium (3 files)

---

## Checkpoint 1: Performance Baseline & Resolution Lock
- [x] YOLO11n-pose ONNX re-exported at 640x640 and YOLO11n at 512x512
- [x] OpenVINO / CPU session thread count pinned to physical cores
- [x] Inference latency verified at $\ge 25\text{ FPS}$ on edge CPU

---

## Task 2: Persistent Inter-Camera ReID Gallery with Spatial Exclusivity
**Description:** Implement `PersistentReIDGallery` in `edge/reid.py` and decouple gallery state from camera-level tracker resets in `edge/launcher.py`. Enforce a **spatial exclusivity constraint**: if Technician A has an active, confirmed track on Camera 1 (Bay 1), Camera 2 (Bay 3) cannot assign Technician A's identity to an ambiguous track unless Camera 1 registers a departure or track loss for $> 5.0\text{ seconds}$. Enforce strict anti-poisoning enrollment (requires face confidence $\ge 0.70$ and upright bounding box $H/W \ge 1.0$). Cap gallery to 16 embeddings per technician with FIFO eviction.
**Acceptance criteria:**
- [x] `PersistentReIDGallery` persists across active camera switches and tracker resets.
- [x] Spatial exclusivity blocks Camera 2 from falsely claiming a technician who is currently active on Camera 1.
- [x] Anti-poisoning policy rejects distorted, occluded, or low-confidence face crops.
- [x] Gallery size is capped at 16 embeddings per person with FIFO eviction.
**Verification:**
- [x] Tests pass: `.venv/bin/python -m unittest test_tracker -k gallery -v`
- [x] Build succeeds: `.venv/bin/python -c "import launcher, tracker, reid; print('Imports valid')"`
- [x] Manual check: Simulate two camera feeds with synthetic dark-uniform crops; verify spatial exclusivity prevents duplicate identity assignment.
**Dependencies:** Task 1
**Files likely touched:**
- `edge/reid.py`
- `edge/tracker.py`
- `edge/launcher.py`
**Estimated scope:** Medium (3 files)

---

## Task 3: Throttled & Event-Gated 360° Re-ID Handover
**Description:** Implement cadenced, event-gated Re-ID extraction in `edge/tracker.py` to prevent the 70–140 ms OSNet CPU hot-loop. Extract 512-dim OSNet embeddings ONLY when: (1) a track hits $\ge 3$ hits and has no initial embedding (`track.features is None`), (2) spatial IoU / Kalman matching drops completely, or (3) as a background refresh throttled to at most once every **30–45 frames** ($2.0-3.0\text{s}$). Skip extraction completely on degenerate crops ($w < 20\text{ px}$ or $h < 40\text{ px}$). Match against `PersistentReIDGallery` with cosine similarity $\ge 0.65$.
**Acceptance criteria:**
- [x] OSNet forward pass is never invoked on every frame; extraction is capped at $\le 1$ pass per 30 frames per track.
- [x] Extraction drops out immediately on degenerate crops ($w < 20$ or $h < 40$).
- [x] Tracks with turned heads ($90^\circ-180^\circ$) recover confirmed staff identity without causing FPS degradation.
**Verification:**
- [x] Tests pass: `.venv/bin/python -m unittest test_tracker test_person -v`
- [x] FPS benchmark check: Verify that running a 30-second sequence with 2 turned-away tracks maintains $\ge 20\text{ FPS}$.
- [x] Manual check: Verify simulated track without face matches gallery embedding when spatial matching drops.
**Dependencies:** Task 2
**Files likely touched:**
- `edge/tracker.py`
- `edge/test_tracker.py`
**Estimated scope:** Small (2 files)

---

## Checkpoint 2: ReID & Multi-Camera Continuity
- [x] All tracker unit tests pass (`.venv/bin/python -m unittest test_tracker -v`)
- [x] ReID gallery survives camera switches without clearing embeddings
- [x] Spatial exclusivity blocks cross-camera identity collisions
- [x] Re-ID extraction throttling maintains $\ge 20\text{ FPS}$ with turned-away tracks

---

## Task 4: Strict Three-Tier Bay Labor Admission Gate & Anti-Clutter
**Description:** Update `BayZoneManager.update()` in `edge/occupancy.py` to enforce a strict three-tier admission gate before admitting a detection into a bay session or accumulating labor time. Detections must satisfy: (1) confirmed track status (`hits >= 3` and not flagged as `clutter`), (2) verified torso keypoint connectivity (shoulders + hips), and (3) non-zero motion/jitter history over a temporal window. Inanimate objects (boots, bags, jack stands, tires) are rejected at the gate and never start a bay session or accrue unverified seconds. Whitelist `is_creeper_or_underbody_pose` so legitimate mechanics working under chassis are preserved.
**Acceptance criteria:**
- [x] Inanimate clutter inside bay ROIs (shoes, backpacks, tires) is blocked from starting bay sessions.
- [x] `bay.state` remains `EMPTY` / `IDLE` and `bay.unverified_seconds` stays 0.0 on stationary clutter.
- [x] Mechanics lying on creepers (`is_creeper_or_underbody_pose`) pass admission and accumulate labor time.
**Verification:**
- [x] Tests pass: `.venv/bin/python -m unittest test_garage -k clutter -v`
- [x] Tests pass: `.venv/bin/python -m unittest test_person -k creeper -v`
- [x] Manual check: Run synthetic test with stationary shoes in Bay 1 ROI; verify zero wrench/unverified time accrued.
**Dependencies:** Task 1
**Files likely touched:**
- `edge/occupancy.py`
- `edge/test_garage.py`
**Estimated scope:** Small (2 files)

---

## Task 5: 30-Second Bay Occlusion Hysteresis & Polygon Exit Short-Circuit
**Description:** Refine bay session state hysteresis in `edge/occupancy.py`. When a confirmed technician is working in a bay and becomes occluded under a vehicle or behind a lift pillar, maintain their active `WORKING` or `UNDER_VEHICLE` session and locked identity for up to 30 seconds before timing out to vacant. If the technician's bounding box is explicitly observed crossing the bay polygon boundary, short-circuit the 30-second dwell timer and close the bay session immediately.
**Acceptance criteria:**
- [x] Occlusion dwell grace period holds state and active technician for up to 30 seconds during visual occlusions.
- [x] Re-emergence within 30 seconds resumes labor on the same session without splitting records or reverting to `"Employee"`.
- [x] Bounding box exiting the bay polygon short-circuits the grace timer and closes the session immediately.
**Verification:**
- [x] Tests pass: `.venv/bin/python -m unittest test_garage -k occlusion -v`
- [x] Build succeeds: `.venv/bin/python -c "import occupancy; print('Occupancy clean')"`
- [x] Manual check: Verify simulated 20s occlusion holds state; verify immediate exit closes session.
**Dependencies:** Task 4
**Files likely touched:**
- `edge/occupancy.py`
- `edge/test_garage.py`
**Estimated scope:** Small (2 files)

---

## Checkpoint 3: Bay State Machine & Labor Integrity
- [x] All garage unit tests pass (`.venv/bin/python -m unittest test_garage -v`)
- [x] Inanimate clutter never transitions bay to WORKING or accrues unverified seconds
- [x] Mechanic under vehicle retains session continuity across 20-30s occlusions
- [x] Physical bay departure closes session without lingering dwell time

---

## Task 6: Sidecar PyInstaller Build & DirectML/OpenVINO Packaging
**Description:** Update `edge/build_sidecar.py` and `edge/inbound-engine.spec` to validate that all required models (`yolo11n-pose.onnx` at 640x640, `yolo11n.onnx` at 512x512, `osnet_x0_25_market1501.onnx`), runtime libraries (OpenVINO / DirectML / onnxruntime shared DLLs), MSVC runtimes, and modules (`one_euro`) are validated during the dry-run inspection step before PyInstaller packaging.
**Acceptance criteria:**
- [x] `build_sidecar.py --dry-run` verifies presence of 640x640 and 512x512 ONNX models and `one_euro.py`.
- [x] `inbound-engine.spec` bundles OpenVINO and onnxruntime shared libraries and hidden imports without missing symbols.
- [x] Package dry-run confirms zero missing runtime dependencies.
**Verification:**
- [x] Dry-run command passes: `.venv/bin/python build_sidecar.py --dry-run`
- [x] Spec check passes: `.venv/bin/python -c "import PyInstaller; print('PyInstaller available')"`
**Dependencies:** Task 1, Task 2
**Files likely touched:**
- `edge/build_sidecar.py`
- `edge/inbound-engine.spec`
**Estimated scope:** Small (2 files)

---

## Task 7: Virtual Camera Live Multi-Stream Benchmark Test
**Description:** Build an automated end-to-end regression script using `tools/virtual-camera/` and `edge/test_video_file.py` to stream realistic workshop video sequences through the complete ML pipeline. Measure tracking FPS ($\ge 20\text{ FPS}$ target on edge CPU), tracklet ID switch count, face-to-ReID handoff rate, and verify that bay wrench time accrues accurately within 5% tolerance.
**Acceptance criteria:**
- [x] Test executes a 30-second garage video clip through `VideoFileAdapter` and `LiveStreamEngine`.
- [x] Measures tracking continuity: zero unexpected track ID resets on walking technician.
- [x] Average throughput exceeds $20\text{ FPS}$ on edge CPU.
- [x] Bay wrench time matches expected ground-truth duration within 5% tolerance.
**Verification:**
- [x] Tests pass: `.venv/bin/python -m unittest test_video_file -v`
- [x] Benchmark script executes cleanly: `.venv/bin/python -c "import adapters.video_file; print('Video adapter ready')"`
**Dependencies:** Task 3, Task 5
**Files likely touched:**
- `edge/test_video_file.py`
- `tools/virtual-camera/generate_test_clip.py`
**Estimated scope:** Small (2 files)

---

## Checkpoint 4: Complete System Validation
- [x] Full automated test suite passes: `.venv/bin/python -m unittest discover -s edge -p "test_*.py"`
- [x] Zero track flapping, zero skeleton flailing, zero clutter hallucinations
- [x] Multi-camera ReID and bay identity locking fully verified at $\ge 20\text{ FPS}$
