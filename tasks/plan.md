# Implementation Plan: Machine Learning Worker Detection, Stream Mirroring & Multi-Camera Grid Hardening

## Overview
This implementation plan diagnoses, isolates, and resolves the three critical functional regressions reported after the Edge Pose Tracking rollout, and installs unbreakable invariant guardrails to protect against future modifications:
1. **Worker Detection & Tracking Failure**:
   - The kinematic pose acceptance gate (`is_human_pose` in `edge/person.py`) and three-tier bay admission gate (`is_admissible_bay_occupant` in `edge/occupancy.py`) over-indexed on full-standing skeletons with connected hips/legs. When a worker is framed by a laptop webcam, desk, toolbench, or hood line, hips/legs are occluded, causing YOLO person boxes ($\text{conf} \ge 0.50-0.70$) to be rejected as "blobs" and excluded from tracking.
   - `tracker.update()` hides unconfirmed tracks (`hits < 3`), creating an initial 2-frame blind spot which, compounded by the 3.0s idle cadence, causes the system to drop newly appearing workers before they ever become visible.
   - The UI "AI Person & Vehicle Detection" switch (`#camera-ml-input`) lacked an `onchange` event listener and was omitted from `save_camera` and `/api/connect-stream` payloads, meaning toggling it in the form never updated the engine.
2. **Mirror / Flip View Disconnect**:
   - `edge/hub.html`'s `toggleFlip(which)` only mutated the DOM ROI polygon coordinates, leaving the video element and live WebRTC/JPEG player unflipped without CSS transforms or stream reloads.
   - Dynamic `/api/orient` on the backend updated `cfg["flip"]` in memory and disk, but never propagated `output_flip` to the active `AsyncFrameGrabber` or `CameraStreamWorker` instances, causing pre-encoded server JPEGs to remain unflipped.
3. **Main Laptop Camera Disappearing in Grid Mode**:
   - The multi-camera grid rendered camera tiles using standard HTML `<img>` elements pointing to `/api/camera/${cam.id}/frame.jpeg`. In browsers, receiving HTTP 204 (No Content) during startup or frame intervals triggers `img.onerror`, which permanently hid the image element and displayed the fallback placeholder.
   - When launching directly into grid mode or before clicking "Connect Stream", `self.is_streaming` was `False`, meaning `CameraStreamPool` was never initialized with active workers and `self.current_frame_jpeg` was `None`, resulting in immediate HTTP 204 responses for the main webcam.

---

## Architecture Decisions & Guardrails

### 1. Robust Pose Acceptance for Upper-Body & Seated Workers (`edge/person.py`)
- **Root Cause**: `is_face_closeup` disqualifies anyone when both shoulders are visible (`_count_visible >= 2 -> False`), forcing desk/webcam workers into `is_human_pose`, which demands $\ge 2$ kinematic bones (requiring hips/legs). Furthermore, `is_admissible_bay_occupant` strictly demands `has_shoulder and has_hip`.
- **Solution**:
  - Add an explicit `is_upper_body_pose` path in `is_human_pose`: if a head (nose/eyes/ears) and both shoulders form a valid shoulder girdle (`L_SHOULDER` to `R_SHOULDER`) with the head positioned anatomically above the shoulders and $\text{conf} \ge 0.35$, the detection is accepted as human even if hips/legs are below the desk or vehicle bumper.
  - Update `is_admissible_bay_occupant`: accept workers satisfying `(has_shoulder and has_hip) or (has_head and has_shoulder and upper_bones >= 1)`.
  - In `PersonTracker.update()`: render high-confidence unconfirmed tracks (`hits < 3`) with a distinct tentative style (e.g. dashed bounding box / acquiring state) rather than discarding them completely from the frame's `last_accepted` rendering list.

### 2. End-to-End Stream Mirroring Architecture (`edge/hub.html` & `edge/launcher.py`)
- **Root Cause**: `toggleFlip()` in `hub.html` transformed ROI geometries but never styled `#live-camera-feed` / `#live-webrtc`, and `/api/orient` never updated `grabber.output_flip`.
- **Solution**:
  - In `edge/launcher.py`'s `set_orient`: explicitly update `self.grabber.output_flip = self.cfg["flip"]`, and iterate over all `self.camera_pool._workers` to update `worker.grabber.output_flip` and `worker.cfg["flip"]`.
  - In `edge/capture.py`: ensure `AsyncFrameGrabber._capture_loop` applies `output_flip` to both the encoded JPEG and the frame metadata.
  - In `edge/hub.html`: apply CSS transform matrix (`transform: scaleX(-1)` for `h`, `scaleY(-1)` for `v`, or `scale(-1, -1)` for both) to `#live-camera-feed`, `#live-webrtc`, and corresponding grid tile images so visual feedback is instantaneous, and call `startLivePlayer()` to align stream state.

### 3. Grid Stream Resiliency & Unified Webcam Ingest (`edge/hub.html` & `edge/launcher.py`)
- **Root Cause**: Grid tiles use raw `<img>` tags that collapse on HTTP 204. In addition, when the engine is in `STANDBY` (`is_streaming=False`), `get_camera_frame` returns `None` for the main camera because background workers are not started.
- **Solution**:
  - In `edge/hub.html`: replace brittle raw `<img src>` polling with robust fetch-and-blob-URL pipeline (identical to `startJpegPump`) or prevent `img.onerror` from hiding the video element on HTTP 204 / transient empty responses.
  - In `edge/launcher.py`: ensure `get_camera_frame(cid)` can spin up on-demand preview capture for local webcams or serve the fallback grabber even if `is_streaming` has not been formally engaged. When in grid mode, ensure the main camera worker cleanly provisions frames without device busy locks (`EBUSY`).

### 4. Regression Shielding & Invariant Guardrails (`.agents/rules/camera_detection_invariants.md`)
- Create dedicated invariant rules documenting that:
  - Upper-body / webcam workers must never be gated on hip or ankle keypoints.
  - Orientation (`rotate` / `flip`) must always synchronize simultaneously across backend grabber encoding, DOM overlay geometries, and frontend video transforms.
  - Multi-camera grid tiles must never hide the main camera feed on transient 204 responses.
  - Unit tests in `test_person.py`, `test_tracker.py`, `test_multicam_roi.py`, and `test_patch_fixes.py` will enforce these contracts on every CI/test run.

---

## Task List

### Phase 1: Machine Learning Detection & Tracking Fixes
- [ ] Task 1: Relax upper-body & webcam pose acceptance in `edge/person.py` and `edge/occupancy.py`
- [ ] Task 2: Fix unconfirmed track visibility and dynamic cadence in `edge/tracker.py` and `edge/launcher.py`
- [ ] Task 3: Wire frontend ML toggle checkbox (`#camera-ml-input`) to backend `/api/cameras/toggle-ml` and payload handlers

### Checkpoint: Detection & Tracking Verified
- [ ] Laptop webcam detects user immediately with bounding box, skeleton, and face tracking
- [ ] `test_person.py`, `test_tracker.py`, and `test_garage.py` pass without regression

### Phase 2: Stream Mirroring & Orientation Synchronization
- [ ] Task 4: Propagate dynamic flip to grabbers in `edge/launcher.py` and `edge/capture.py`
- [ ] Task 5: Implement synchronized CSS mirror transforms and feed restart in `edge/hub.html`

### Checkpoint: Mirror Options Verified
- [ ] Clicking "Left-right" and "Up-down" mirrors the live video feed and ROI box simultaneously in both single and grid view

### Phase 3: Multi-Camera Grid Resilience & Main Camera Ingest
- [ ] Task 6: Resilient grid tile polling (prevent HTTP 204 error collapse) in `edge/hub.html`
- [ ] Task 7: Ensure main webcam frames are consistently served via `get_camera_frame` in `edge/launcher.py`

### Checkpoint: Grid System Verified
- [ ] Main laptop camera streams continuously in 2x2, 3x3, auto grid, and 1x1 full screen

### Phase 4: Invariant Rules & Regression Shielding
- [ ] Task 8: Add regression test suite and system invariant documentation (`.agents/rules/camera_detection_invariants.md`)

### Checkpoint: Complete System Validation
- [ ] All 172+ automated tests pass clean with zero errors
