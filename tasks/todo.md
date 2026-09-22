# Worker Detection, Stream Mirroring & Multi-Camera Grid Resilience - Tasks

## Task 1: Relax Upper-Body & Webcam Pose Acceptance Gate
**Description:** In `edge/person.py`, add `is_upper_body_pose` to `is_human_pose` to ensure workers framed from chest-up by a laptop webcam, desk, tool bench, or engine hood are accepted as humans rather than rejected as inanimate clutter. Require head visibility (nose, eyes, or ears) plus a connected shoulder girdle (`L_SHOULDER` to `R_SHOULDER`) with the head positioned above the shoulders and bounding box confidence $\ge 0.35$. In `edge/occupancy.py`, update `is_admissible_bay_occupant` so bay occupancy admission accepts legitimate upper-body mechanics when hips/ankles are occluded by vehicle components.
**Acceptance criteria:**
- [ ] `is_human_pose` accepts webcam, seated, and desk workers who have head and shoulder visibility without requiring hip or leg joints.
- [ ] `is_admissible_bay_occupant` in `edge/occupancy.py` accepts upper-body mechanics working over vehicle hoods or at service benches.
- [ ] Hallucinated bike/engine frames with clustered joints or inverted anatomy continue to be rejected.
**Verification:**
- [ ] Tests pass: `.venv/bin/python -m unittest test_person -k test_upper_body -v`
- [ ] Tests pass: `.venv/bin/python -m unittest test_garage -k clutter -v`
- [ ] Manual check: Run pose inference on webcam capture frame; verify person detection changes from rejected to accepted.
**Dependencies:** None
**Files likely touched:**
- `edge/person.py`
- `edge/occupancy.py`
- `edge/test_person.py`
**Estimated scope:** Small (3 files)

---

## Task 2: Fix Unconfirmed Track Visibility & Dynamic Cadence Blind Spot
**Description:** In `edge/tracker.py`, update `PersonTracker.update()` so that high-confidence detections (`conf >= 0.35`) in their initial confirmation window (`hits < 3`) are returned with an acquiring flag (`det.confirmed = False`) rather than being discarded into an empty list. In `edge/person.py`, update `draw_detection()` to render tentative tracks (e.g. dashed border or acquiring tag) so workers are immediately visible to the user from frame 1. In `edge/launcher.py`, ensure that when an acquiring person is present in the frame, `dynamic_interval` immediately switches to high-cadence detection ($1.0 / \text{detect\_fps}$) rather than sleeping for 3.0s.
**Acceptance criteria:**
- [ ] A person entering the frame is rendered on screen on frame 1 rather than being completely invisible until frame 3.
- [ ] Motion/person presence immediately kicks `dynamic_interval` into high cadence ($8\text{ FPS}$).
- [ ] Downstream bay wrench-time accumulation and Face ID enrollment continue to require confirmed tracks (`hits >= 3`) to prevent clutter accumulation.
**Verification:**
- [ ] Tests pass: `.venv/bin/python -m unittest test_tracker -v`
- [ ] Tests pass: `.venv/bin/python -m unittest test_video_file -v`
- [ ] Manual check: Verify new track appears immediately on screen during live stream ingest.
**Dependencies:** Task 1
**Files likely touched:**
- `edge/tracker.py`
- `edge/person.py`
- `edge/launcher.py`
**Estimated scope:** Medium (3 files)

---

## Task 3: Wire Frontend ML Toggle Checkbox to Backend State & API
**Description:** In `edge/hub.html`, attach an `onchange` event listener to `#camera-ml-input` so toggling the switch invokes `/api/cameras/toggle-ml` immediately for the active camera. In `cameraPayload()` and `connectAndStreamCamera()`, include `ml_enabled: document.getElementById('camera-ml-input').checked` in the request body. In `edge/launcher.py`, update `save_camera()` to map and persist `ml_enabled` into `self.cfg["cameras"]`.
**Acceptance criteria:**
- [ ] Toggling the "AI Person & Vehicle Detection" switch immediately fires an API request to toggle ML on the active camera without needing to click "Save Camera".
- [ ] Connecting a stream via `connectAndStreamCamera()` preserves the user's ML enabled/disabled state.
- [ ] `save_camera()` persists `ml_enabled` in `config.yaml`.
**Verification:**
- [ ] Tests pass: `.venv/bin/python -m unittest test_background_ml -v`
- [ ] Manual check: Toggle switch in UI; verify backend returns updated camera list with matching `ml_enabled`.
**Dependencies:** None
**Files likely touched:**
- `edge/hub.html`
- `edge/launcher.py`
**Estimated scope:** Small (2 files)

---

## Checkpoint 1: Detection & Tracking Verified
- [ ] Laptop webcam detects user immediately with bounding box, skeleton, and face tracking
- [ ] Unit tests pass clean: `.venv/bin/python -m unittest test_person test_tracker test_garage`

---

## Task 4: Propagate Dynamic Flip to Grabbers and Ingest Pipeline
**Description:** In `edge/launcher.py`, update `set_orient()` so that when a flip mode (`'h'`, `'v'`, or `'none'`) is applied via `/api/orient`, it immediately sets `self.grabber.output_flip = new_flip` and updates `worker.grabber.output_flip = new_flip` and `worker.cfg["flip"] = new_flip` across all active camera pool workers. In `edge/capture.py`, ensure `AsyncFrameGrabber` applies `output_flip` reliably during JPEG generation and frame publication so all served frames match the requested orientation.
**Acceptance criteria:**
- [ ] Calling `/api/orient` with `flip: "h"` or `flip: "v"` dynamically changes the orientation of JPEGs served from `/api/frame.jpeg` and `/api/camera/<id>/frame.jpeg`.
- [ ] Both the active grabber and background camera workers immediately reflect the updated flip mode without requiring a server restart.
**Verification:**
- [ ] Tests pass: `.venv/bin/python -m unittest test_patch_fixes -k orient -v`
- [ ] Manual check: Call `/api/orient` with `flip: "h"`; verify fetched `/api/frame.jpeg` has horizontally mirrored pixels.
**Dependencies:** None
**Files likely touched:**
- `edge/launcher.py`
- `edge/capture.py`
**Estimated scope:** Small (2 files)

---

## Task 5: Implement Synchronized CSS Mirror Transforms and Feed Restart in UI
**Description:** In `edge/hub.html`, update `toggleFlip(which)` and `paintOrient()` to apply a responsive CSS transform (`transform: scaleX(-1)` for horizontal, `scaleY(-1)` for vertical, or `scale(-1, -1)` for both) to `#live-camera-feed`, `#live-webrtc`, and active grid tile elements. Call `startLivePlayer(lastLiveStreamId, lastLiveMedia)` and `requestAnimationFrame(layoutVideoStage)` on flip toggles to ensure the video canvas and DOM ROI overlays remain in perfect pixel alignment.
**Acceptance criteria:**
- [ ] Clicking "Left-right" or "Up-down" immediately mirrors the camera feed display in the DOM.
- [ ] The ROI box and camera view flip synchronously so bounding boxes never become inverted relative to the video image.
- [ ] The transform applies cleanly in both single view and multi-camera grid tiles.
**Verification:**
- [ ] Tests pass: Inspect CSS styling rules and verify DOM element style assignment in `edge/hub.html`.
- [ ] Manual check: Click "Left-right"; verify webcam feed flips horizontally alongside ROI boxes.
**Dependencies:** Task 4
**Files likely touched:**
- `edge/hub.html`
**Estimated scope:** Small (1 file)

---

## Checkpoint 2: Mirror Options Verified
- [ ] Clicking mirror buttons flips both the camera view and the ROI overlay simultaneously in single and grid mode
- [ ] Grabbers encode flipped frames on backend; UI mirrors video feed in frontend

---

## Task 6: Resilient Grid Tile Polling (Prevent HTTP 204 Error Collapse)
**Description:** In `edge/hub.html`, overhaul the grid tile rendering in `renderMultiCamGrid()`. Rather than assigning `<img src>` which permanently triggers `onerror` on HTTP 204 (No Content) during startup, implement a resilient blob-based fetch loop with retry backoff (mirroring `startJpegPump`) or explicitly ignore HTTP 204 without toggling `img.style.display = 'none'`. When existing tiles are refreshed in `renderMultiCamGrid()`, ensure polling timers are restarted if paused.
**Acceptance criteria:**
- [ ] Receiving an HTTP 204 response from `/api/camera/<id>/frame.jpeg` does NOT permanently hide the grid tile image or trigger the disconnected fallback.
- [ ] Grid tiles smoothly reconnect and display frames as soon as the camera stream produces data.
- [ ] Re-rendering the grid dimensions (e.g. switching between Auto, 2x2, 3x3) maintains active frame polling without freezing tiles.
**Verification:**
- [ ] Manual check: Switch between 1x1 and 2x2 grid; verify laptop camera tile streams continuously.
- [ ] Tests pass: Frontend integration check in `edge/hub.html`.
**Dependencies:** None
**Files likely touched:**
- `edge/hub.html`
**Estimated scope:** Small (1 file)

---

## Task 7: Ensure Main Webcam Frames Are Consistently Served in Grid Mode
**Description:** In `edge/launcher.py`, update `get_camera_frame(cid)` and `CameraStreamPool`. When the active camera is a local webcam (`source: 0` or protocol `webcam`), ensure that grid requests for `/api/camera/<active_id>/frame.jpeg` cleanly share the active frame grabber's latest JPEG without competing for exclusive device access (`EBUSY`). If `is_streaming` is False when entering grid mode, auto-provision background worker grabbers so the webcam streams frames without requiring manual single-mode connection.
**Acceptance criteria:**
- [ ] `/api/camera/<active_id>/frame.jpeg` reliably returns the latest frame for the laptop camera in grid mode.
- [ ] No `EBUSY` or "busy device" conflicts occur between `CameraStreamPool` workers and `LiveStreamEngine`.
- [ ] Switching between single-mode full screen (1x1) and multi-camera grid maintains uninterrupted video flow.
**Verification:**
- [ ] Tests pass: `.venv/bin/python -m unittest test_video_file -k CameraStreamPool -v`
- [ ] Tests pass: `.venv/bin/python -m unittest test_multicam_roi -v`
- [ ] Manual check: Run grid mode with main camera; verify frame rate is steady and preview never drops.
**Dependencies:** Task 6
**Files likely touched:**
- `edge/launcher.py`
- `edge/test_multicam_roi.py`
**Estimated scope:** Medium (2 files)

---

## Checkpoint 3: Grid System Verified
- [ ] Main laptop camera displays cleanly in 2x2, 3x3, auto grid, and 1x1 full screen
- [ ] No race conditions, port locks, or HTTP 204 tile collapses

---

## Task 8: Invariant Rules & Regression Shielding
**Description:** Create `.agents/rules/camera_detection_invariants.md` and automated regression tests in `edge/test_patch_fixes.py` and `edge/test_person.py` to freeze the core operational laws: (1) upper-body pose whitelist law, (2) orientation transform synchronization law, (3) grid stream resiliency law, and (4) UI ML toggle synchronization law. Ensure that any future refactor attempting to add strict hip requirements or un-mirrored flip handlers immediately fails the automated test suite.
**Acceptance criteria:**
- [ ] `.agents/rules/camera_detection_invariants.md` documents non-negotiable architectural rules.
- [ ] Regression unit tests in `edge/test_person.py` and `edge/test_patch_fixes.py` enforce:
  - Upper-body worker acceptance with occluded lower body.
  - Grabber `output_flip` synchronization upon `/api/orient`.
  - Non-collapsing grid tile response handlers.
- [ ] 100% of repository test suite passes cleanly.
**Verification:**
- [ ] Run full test suite: `PYTHONPATH=edge edge/.venv/bin/python -m unittest discover -s edge -p "test_*.py"`
**Dependencies:** Tasks 1–7
**Files likely touched:**
- `.agents/rules/camera_detection_invariants.md`
- `edge/test_person.py`
- `edge/test_patch_fixes.py`
**Estimated scope:** Small (3 files)

---

## Checkpoint 4: Complete System Validation
- [ ] Full automated test suite passes: `PYTHONPATH=edge edge/.venv/bin/python -m unittest discover -s edge -p "test_*.py"`
- [ ] All 3 user issues resolved and guarded by regression tests
