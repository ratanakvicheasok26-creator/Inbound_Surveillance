# Machine Learning Architecture, Tracking Mechanics & Problem Diagnostics Report

**Project:** Inbound Surveillance Engine  
**Version:** V2 Architecture  
**Key Codebase Locations:** `edge/` (`launcher.py`, `person.py`, `tracker.py`, `tinypose.py`, `face_id.py`, `reid.py`, `occupancy.py`, `liveness.py`, `bay_zoom.py`)  
**Status:** In-Depth Diagnostic & Research Guide  

---

## 1. Executive Summary

This document provides a comprehensive technical breakdown of the machine learning architecture in the **Inbound Surveillance** system. It details:
1. The **end-to-end system architecture** and multi-stage computer vision pipeline.
2. The **exact machine learning models** used, their architectural design, and where they were acquired.
3. How the system performs **person detection, pose estimation, temporal tracking, identity association, and bay occupancy analysis**.
4. A **deep root-cause analysis of all ML failures and defects currently observed in production**, specifically:
   - Why the system constantly **tracks and then untracks people** (flickering, track loss, ID switching).
   - Why the **skeleton moves and flails wildly while the person is standing completely still**.
   - Why the skeleton coordinates frequently **do not match the physical body posture**.
   - Why identity drops from named technicians to unknown `"Employee"` on head/body turns.
   - Why inanimate objects (shoes, backpacks, engine blocks) are hallucinated as human workers.

---

## 2. System Architecture & Pipeline Flow

The edge surveillance engine runs a multi-threaded, hierarchical vision pipeline that processes live camera streams (RTSP, USB V4L2 webcams, IP Webcam HTTP streams, or go2rtc WebRTC gateway feeds) to monitor automotive repair bays.

```
                    ┌────────────────────────────────────────┐
                    │       Video Ingestion Layer            │
                    │   (RTSP / WebRTC / USB / Phone HTTP)   │
                    └───────────────────┬────────────────────┘
                                        │ BGR Video Frames (15–30 FPS)
                                        ▼
                    ┌────────────────────────────────────────┐
                    │      Vehicle Detection (YOLO11n)       │
                    │   Identifies cars, bikes, bays, hoists │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │                Tier 1: Person & Pose Estimation Engine                 │
    │   Configured via config.yaml: `pose_engine: yolo` vs `tinypose`        │
    │                                                                        │
    │  [Backend A: Ultralytics YOLO11n-pose]   [Backend B: Paddle TinyPose] │
    │  - Single-stage detector + pose head     - Stage 1: PicoDet (320x320)  │
    │  - 640x640 input, direct 17 keypoints    - Stage 2: TinyPose (192x256) │
    │                                            - 64x48 subpixel heatmap    │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │ Raw Detections + 17 COCO Keypoints
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │          Tier 2: Kinematic Verification & Anti-Object Filtering        │
    │                           (`edge/person.py`)                           │
    │  - Biologically connected bone graph validation (Torso / Limb chains)  │
    │  - Inanimate clutter rejection (eliminates floating shoe/tool points)  │
    │  - Posture classifiers: Creeper pose, Hood lean, Squat/Crouch          │
    │  - Suppression of nested/duplicate bounding boxes                      │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │ Verified Human Bounding Boxes & Keypoints
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │           Tier 3: Multi-Object Tracking & Temporal Filtering           │
    │                          (`edge/tracker.py`)                           │
    │  - ByteTrack-style Kalman filter on bounding boxes (cx, cy, w, h)      │
    │  - Multi-frame confirmation (min_hits = 3)                             │
    │  - Liveness probe: Pixel motion energy + keypoint jitter analysis      │
    │  - Spatial IoU + Re-ID feature matching + Center distance fallback     │
    │  - Static clutter suppression & coasting prediction during occlusions  │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │ Tracked Person Objects (Persistent track_id)
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │            Tier 4: Multi-Modal Identity Binding & Re-ID                │
    │                   (`edge/face_id.py`, `edge/reid.py`)                  │
    │  - Face Detection: OpenCV YuNet (320x320)                              │
    │  - Face Recognition: OpenCV SFace (128-dim embeddings vs faces/ dir)   │
    │  - Body Re-ID: OSNet (512-dim) / 8x8x8 HSV Appearance Histogram        │
    │  - Tracklet Identity Locking (binds recognized staff name to track_id) │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │ Annotated Detections (Name, Time, Pose)
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │            Tier 5: Bay Occupancy & Labor State Machine                 │
    │                         (`edge/occupancy.py`)                          │
    │  - Multi-Bay ROI mapping (polygon & bounding box intersection)         │
    │  - Work Classification: WORKING (wrenching), UNDER_VEHICLE, IDLE       │
    │  - Temporal hysteresis & unverified labor holding                      │
    │  - Optional Tier 2 Cloud Vision AI Auditor (VLM behavior verification) │
    │  - SQLite logging, MJPEG overlay streaming, and Telegram alerts        │
    └────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Machine Learning Models: Inventory & Provenance

The table below catalogs every machine learning model in the repository, its architecture, role, and where it was acquired.

| Model File | Architecture / Backbone | Task | Size / Parameters | Source & Provenance |
|---|---|---|---|---|
| **`yolo11n-pose.pt`** | YOLO11 Nano Pose (Ultralytics) | Full-frame human detection and 17-keypoint pose estimation | ~6.2 MB (~2.6M params) | **Ultralytics Inc.** ([github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)). Pre-trained on the COCO Keypoint Dataset. |
| **`yolo11n.pt`** | YOLO11 Nano Detection (Ultralytics) | Vehicle detection (classes: car, motorcycle, bus, truck) | ~5.6 MB (~2.6M params) | **Ultralytics Inc.** Pre-trained on COCO Object Detection. |
| **`yolo11n_improved.pt`** | YOLO11 Nano (fine-tuned) | Domain-adapted vehicle / garage asset detection | ~5.4 MB | Fine-tuned variant of Ultralytics YOLO11n for garage surveillance. |
| **`picodet_s_320_lcnet_pedestrian.onnx`** | PP-PicoDet-S (LCNet backbone, SimOTA) | Lightweight pedestrian bounding box detection (320x320 input) | ~4.8 MB (~1.1M params) | **Baidu PaddlePaddle / PaddleDetection** ([github.com/PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)), converted to ONNX via `guojin-yan/Csharp_and_OpenVINO_deploy_PP-TinyPose`. |
| **`tinypose_256_192.onnx`** | PP-TinyPose (Modified Lite-HRNet / ShuffleNetV2) | Top-down 17-keypoint pose estimation on cropped persons (192x256) | ~5.7 MB (~1.3M params) | **Baidu PaddlePaddle / PaddleDetection**. Trained on COCO + AI Challenger keypoint benchmarks. |
| **`face_detection_yunet_2023mar.onnx`** | YuNet (Lightweight Feature Pyramid Network) | Fast face detection & 5 facial landmark extraction | ~232 KB | **OpenCV Zoo / Shiqi Yu** ([github.com/opencv/opencv_zoo](https://github.com/opencv/opencv_zoo)). |
| **`face_recognition_sface_2021dec.onnx`** | SFace (ResNet-like with SphereFace2 Hypersphere loss) | 128-dimensional facial embedding extraction for staff identification | ~38.7 MB | **OpenCV Zoo / Zhong et al.** ([github.com/opencv/opencv_zoo](https://github.com/opencv/opencv_zoo)). |
| **`osnet_x0_25_market1501.onnx`** *(Referenced in `edge/reid.py`)* | OSNet-x0.25 (Omni-Scale Feature Learning Network) | 512-dimensional full-body appearance embedding for 360° Re-ID | ~1.1 MB | **Kaiyang Zhou / Torchreid** ([github.com/KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)). *Note: Currently missing from `edge/models/` in this install, causing runtime fallback to HSV color histogram.* |

---

## 4. How the Machine Learning & Tracking Operates

### 4.1. Pose Inference Engines (`yolo` vs `tinypose`)
The system supports two swappable pose backends configured in `config.yaml` (`pose_engine: tinypose` or `yolo`):

1. **Ultralytics YOLO11n-pose (1-stage)**:
   - Takes the 640x640 resized full video frame.
   - Concurrently outputs bounding boxes, class scores, and 17 COCO keypoint coordinates $(x, y, \text{conf})$ in a single forward pass.
   - Fast on modern GPUs (2–5 ms), but heavier on edge CPUs (45–90 ms).

2. **Paddle PP-TinyPose + PP-PicoDet (2-stage)**:
   - **Stage 1 (PicoDet)**: Resizes frame to 320x320 and predicts pedestrian boxes. Boxes are scaled back to frame dimensions.
   - **Stage 2 (TinyPose)**: For each detected person box, crops the frame with a 15% margin to fit the 3:4 aspect ratio (192x256). Normalizes using ImageNet mean/std and runs ONNX inference.
   - **Heatmap Output**: Outputs a $(17 \times 64 \times 48)$ heatmap. Finds the maximum point (`argmax`) for each keypoint slice, applies subpixel refinement, and projects coordinates back to the original full-frame space.

### 4.2. Kinematic Skeleton Filtering (`edge/person.py`)
Before passing detections to the tracker, the system attempts to filter out false positives:
- **Bone Graph Check (`KINEMATIC_EDGES`)**: Checks if keypoints form valid biological bones (shoulders, elbows, wrists, hips, knees, ankles) of plausible length relative to the bounding box diagonal.
- **Crossing Bone Rejection (`bones_cross`)**: Rejects detections where limb segments cross each other.
- **Cluster Rejection (`keypoints_are_clustered`)**: Discards boxes where keypoints are bunched up into a tight ball (typical of textures on engine parts).
- **Posture Shortcuts**: Specialized logic for garage poses:
  - `is_creeper_or_underbody_pose`: Worker lying horizontally on a creeper under a vehicle.
  - `is_hood_lean_pose`: Mechanic leaning over a bumper into an engine bay.
  - `is_crouch_or_sit_pose`: Kneeling or squatting by wheel wells.

### 4.3. Multi-Object Tracking (`edge/tracker.py`)
- Tracks individuals across frames using a Kalman filter predicting $(cx, cy, w, h)$ with damped velocity.
- Matches detections using:
  1. **IoU Matching**: Greedy matching against Kalman predicted bounding boxes (`iou_threshold = 0.30`).
  2. **Re-ID Feature Matching**: Compares 512-dim embeddings if IoU fails.
  3. **Center-Distance Matching**: Fallback when boxes shrink during hood-leaning.
- Requires $\ge 3$ consecutive hits (`min_hits = 3`) before a track is considered "confirmed" and shown in the output.
- Coasts lost tracks for up to 30 frames (`max_age = 30`) during brief occlusions.

### 4.4. Identity Association (`edge/face_id.py` & `edge/reid.py`)
- When a person is detected, `face_id.py` crops the person's upper body and runs YuNet face detection.
- If a face is found, SFace extracts a 128-dim embedding and computes cosine similarity against pre-enrolled images in `faces/<StaffName>/`.
- If similarity $\ge 0.60$, the detection is labeled with the staff member's name (`is_staff = True`).
- `tracker.py` locks this identity to the `track_id`. When the technician turns away and the face disappears, the track preserves their name.

---

## 5. Comprehensive Analysis of Current Machine Learning Problems

Below is the detailed technical breakdown of every issue and failure mode currently affecting the vision and tracking pipeline.

---

### Issue 1: Constant Tracking and Untracking (Flickering & Track Loss)

#### Symptom:
A person walking or standing in the camera view is detected for 1–2 seconds, then the bounding box and skeleton vanish (untracked), then reappear with a new Track ID a second later, causing flickering and fragmented analytics.

#### Root Causes:

1. **Over-Aggressive Inanimate/Clutter Killing in `edge/tracker.py`**:
   In `tracker.py`, lines 395–404:
   ```python
   if anatomy_is_weak(getattr(det, "keypoints", []) or [], 0.35):
       trk.weak_anatomy_hits += 1
   else:
       trk.weak_anatomy_hits = max(0, trk.weak_anatomy_hits - 1)
   static_long_enough = trk.hits >= self.static_hits and trk.motion < self.static_px
   weak_clutter = static_long_enough and trk.weak_anatomy_hits >= self.static_hits // 2
   if not trk.is_staff and weak_clutter:
       trk.clutter = True
   ```
   - If a person stands relatively still (`motion < 3.0` pixels) for just 20 frames (~0.8 seconds at 24fps) and has not yet been recognized by Face ID (`not trk.is_staff`), any dip in keypoint confidence triggers `weak_anatomy_hits`.
   - `anatomy_is_weak` returns `True` whenever head keypoints are hidden (e.g. wearing a cap, head tilted down looking at a car, back turned) and torso keypoints have $<2$ points above 0.35 confidence.
   - Once `weak_anatomy_hits >= 10`, the track is marked as `clutter = True`.
   - In `tracker.py` line 254:
     ```python
     confirmed_ids = {t.track_id for t in self.tracks if t.hits >= self.min_hits and not t.clutter}
     ```
     The person is **immediately stripped from confirmed tracks and disappears from the video feed!**
   - When the person takes a step, their motion exceeds $6.0$ pixels, unmarking them as clutter, but the Kalman filter has already lost synchronization, so the tracker instantiates a **new Track ID**.

2. **Dead-Zone Latency (`min_hits = 3`)**:
   - Every newly created track is suppressed until it has been detected for 3 consecutive frames.
   - If a person is temporarily occluded or the detector misses them for a single frame during initialization, the counter resets. The person flickers between visible and invisible.

3. **Bounding Box Aspect Ratio Collapse During Posture Transitions**:
   - When a technician transitions from standing upright (tall box, e.g., $150 \times 450$ px) to crouching or bending over a fender (wide, short box, e.g., $300 \times 200$ px), the spatial intersection-over-union (IoU) between the Kalman predicted standing box and the new crouched box drops to near $0.0$.
   - The greedy IoU matcher fails because $\text{IoU} < 0.30$.
   - The tracker assumes the standing person vanished and generates a new track for the crouching person.

4. **Single-Frame Confidence Thresholding**:
   - The detector confidence floor is set to a hard threshold (e.g., $0.25$ or $0.35$).
   - Real-world camera noise causes raw detector scores to fluctuate (e.g., $0.28 \rightarrow 0.23 \rightarrow 0.27$).
   - The frame with $0.23$ produces zero detections. If this coincides with low Kalman velocity, the track rapidly degrades.

---

### Issue 2: Skeleton Flailing and Wild Movement While Person is Standing Still

#### Symptom:
A technician is standing completely still, but their rendered skeleton is twitching, vibrating, jumping across the screen, or flailing violently as if moving rapidly.

#### Root Causes:

1. **Complete Absence of Temporal Keypoint Filtering**:
   - The system feeds raw, single-frame keypoints straight from the neural network into the display renderer (`draw_detection` in `edge/person.py`).
   - Deep neural networks exhibit inherent per-frame inference noise: pixel gradients, H.264 compression artifacts, and minor lighting changes cause the output regression heads to wobble by 3–15 pixels every frame.
   - Without an **Exponential Moving Average (EMA)**, **One-Euro Filter**, or **Joint Kalman Filter**, this high-frequency neural network noise is drawn directly onto the screen as violent vibrating joints.

2. **Severe Heatmap Quantization and Pseudo-DarkPose in PP-TinyPose (`edge/tinypose.py`)**:
   - PP-TinyPose uses an extremely small heatmap resolution: **$64 \times 48$ pixels** for the entire body.
   - In a 1080p surveillance frame where a person is 500 pixels tall, **one single pixel on the $64 \times 48$ heatmap represents 8 to 12 pixels in the video frame!**
   - If the network's peak activation wobbles by just one cell, the joint snaps 10 pixels instantly!
   - Furthermore, the subpixel refinement code in `tinypose.py` lines 252–260 is an improper approximation:
     ```python
     # Fake subpixel refinement:
     refined_px = px + 0.25 * float(np.sign(dx))
     refined_py = py + 0.25 * float(np.sign(dy))
     ```
     True DarkPose subpixel decoding computes the continuous Taylor expansion of the 2D Gaussian surface ($x^* = x - \mathcal{H}^{-1} \nabla$). Instead, this code simply adds $\pm 0.25$ depending on the sign of the gradient.
   - This restricts coordinates to discrete step increments ($0.0, +0.25, -0.25$), creating jarring, unnatural step-jumping on stationary joints.

3. **Detector Crop Box Jitter Amplification**:
   - In the two-stage TinyPose pipeline, PicoDet detects the person box first.
   - PicoDet's bounding box moves by 2–6 pixels on every frame even on a stationary person.
   - In `tinypose.py` lines 225–228:
     ```python
     crop_x1 = max(0, int(round(cx - cw / 2.0)))
     crop_y1 = max(0, int(round(cy - ch / 2.0)))
     ```
   - The crop boundary shifts by integer rounding.
   - When TinyPose keypoints are projected back to full-frame space:
     ```python
     frame_coord_x = crop_x1 + (crop_coord_x / float(TINYPOSE_INPUT_W)) * crop_w
     ```
     The jitter of `crop_x1` is added to the jitter of `crop_coord_x`, **doubling the visual jitter**.

4. **Flawed Assumption in the Liveness Probe (`edge/liveness.py`)**:
   - In `edge/liveness.py`, the system explicitly assumes that real human skeletons *must* wobble:
     > *"A person breathes, shifts weight, and makes the pose regression head wobble; a jack stand under fixed shop lighting reprojects to the same pixels and the same joints every frame."*
   - Because the codebase treats keypoint jitter as proof of human life, previous developers intentionally avoided smoothing keypoints, allowing the wobble to propagate directly into the user interface.

---

### Issue 3: Skeleton Inversion and Severe Misalignment with the Body

#### Symptom:
The skeleton is displayed at the wrong angle, limbs cross the body unnaturally, or the skeleton appears detached or misaligned with the actual physical person.

#### Root Causes:

1. **Left/Right Symmetry Confusion (Bilateral Inversion)**:
   - In low-contrast lighting or when staff wear dark uniforms, pose estimation models struggle to differentiate the left arm/leg from the right arm/leg.
   - The model flips left and right joints (e.g. mapping the left wrist coordinate to the right wrist).
   - The rendering engine connects Left Shoulder $\rightarrow$ Left Elbow and Right Shoulder $\rightarrow$ Right Elbow. When coordinates flip, the skeleton lines cross into an "X" over the torso.
   - Because this flips back and forth every few frames, the arms look like they are rapidly thrashing.

2. **Severe Foreshortening from Elevated CCTV Camera Angles**:
   - COCO dataset models are trained on photos taken at eye level (0° to 15° elevation).
   - Garage surveillance cameras are mounted 3–4 meters high on walls or ceilings, pointing down at 35°–60° angles.
   - From this vantage point, a person's head overlaps their chest, the torso appears compressed, and legs are visually truncated. The neural network's spatial priors fail, predicting joints outside the body silhouette.

3. **Background Texture Snapping (Clutter Entanglement)**:
   - Automotive shops contain high-contrast geometric objects: hydraulic lift arms, exhaust pipes, tires, engine belts, wire harnesses, and shadows.
   - When a technician's limb is partially shadowed or occluded by a vehicle, the keypoint regression head latches onto nearby high-contrast edges in the background, anchoring wrists or feet to tools or car parts.

4. **Overly Restrictive `bones_cross` Gate Causing False Rejections**:
   - In `edge/person.py` lines 268–283, `bones_cross()` checks if any arm bones intersect.
   - When a mechanic naturally crosses their arms, reaches across their chest to turn a bolt, or leans on a fender, their arm bones physically intersect in the 2D projection.
   - `bones_cross()` flags this as an inanimate engine hallucination and rejects the detection, causing the skeleton to vanish until the worker uncrosses their arms.

---

### Issue 4: Identity Drops and Timer Fragmentation (George $\rightarrow$ "Employee")

#### Symptom:
A registered technician ("George") is recognized and begins accumulating labor time in a bay. As soon as George turns his back to the camera or leans into an engine bay, his label changes to `"Employee"`, and the bay timer splits into two separate entries.

#### Root Causes:

1. **Single-Frame 2D Facial Recognition Dependence**:
   - OpenCV YuNet (`face_id.py`) requires a relatively clear facial view (yaw/pitch within $\pm 30^\circ$).
   - When a technician bends over, faces a car hood, or walks away, their face is completely invisible.
   - Single-frame recognition immediately returns `UNKNOWN_LABEL` (`"Employee"`).

2. **Missing Full-Body Appearance Model (`OSNet`)**:
   - `edge/reid.py` was architected to use `osnet_x0_25_market1501.onnx` to extract 512-dimensional clothing and appearance vectors that remain identifiable from 360° angles (even when the face is turned away).
   - However, `osnet_x0_25_market1501.onnx` is **not installed in `edge/models/`**.
   - As a result, the system falls back to an **8×8×8 HSV color histogram**.
   - HSV color histograms are extremely fragile: changes in ambient lighting, walking into shadows, or wearing dark clothing with low saturation completely alter the histogram, causing Re-ID matching to fail.

3. **Absence of Bay-Level Identity Hysteresis**:
   - In `edge/occupancy.py`, `_pick_technician` looks at the immediate detections inside the bay on that exact frame.
   - If a track's identity drops to `"Employee"` for even 3 seconds, `occupancy.py` begins billing time to `"Employee"`.
   - The bay manager lacks an "identity lock" that holds the primary technician assigned to the bay until they physically leave the zone.

---

### Issue 5: Inanimate Object Hallucinations (Shoes, Bags, and Engines Detected as People)

#### Symptom:
A workbench with tools, a pile of clothes on a chair, a pair of boots on the floor, or an open vehicle engine bay is detected as a person and accumulates hours of "active labor" in an empty bay.

#### Root Causes:

1. **Heuristic Weaknesses in `edge/person.py`**:
   - YOLO-pose and PicoDet generate proposals on high-contrast textures (e.g. shoes, engine manifolds).
   - Because pose models will output 17 keypoint coordinates for *any* box fed to them, textured objects produce noisy keypoints with low-to-moderate confidence (0.35–0.50).
   - Early versions of `person.py` accepted detections with merely 2 leg or torso points. While recent revisions added kinematic bone length checks, stationary clutter under bright workshop lights can still satisfy minimal bone count heuristics.

2. **Dead Liveness Integration in `edge/tracker.py`**:
   - In `edge/tracker.py`, the function `_is_inanimate(trk)` was designed to test for zero pixel motion energy and zero keypoint jitter over time.
   - However, code inspection reveals that **`_is_inanimate(trk)` is never actually called inside `update()`**!
   - Instead, the tracker relies on `weak_anatomy_hits`, which fails to eliminate objects that happened to trigger a plausible kinematic graph.

---

## 6. Comprehensive Problem & Fix Roadmap

To resolve these issues systematically, the following engineering solutions should be researched and implemented:

| Problem Domain | Root Cause | Recommended Engineering Solution |
|---|---|---|
| **Skeleton Flailing & Jitter** | No temporal filter on predicted keypoints; noisy single-frame regression. | **Implement a 1EuroFilter or Kalman filter on keypoints**: A 1EuroFilter provides adaptive low-pass filtering that eliminates high-frequency jitter when stationary while preserving zero-lag response during rapid motion. |
| **TinyPose Coordinate Snapping** | 64x48 coarse heatmap quantization; discrete `0.25 * sign(dx)` subpixel step. | **Implement True DarkPose Gaussian subpixel decoding**: Compute the continuous 2D Taylor series expansion of the heatmap peak. Alternatively, switch to **YOLO11s-pose with TensorRT/OpenVINO**, which directly regresses continuous floating-point coordinates. |
| **Tracking / Untracking Flapping** | Over-aggressive clutter killing (`weak_anatomy_hits`); strict IoU matching on changing aspect ratios. | **1. Remove the weak anatomy clutter kill for confirmed tracks**.<br>**2. Upgrade tracker to ByteTrack with BoT-SORT / Re-ID association**: Allow bounding box association based on spatial distance and feature embeddings when IoU drops during crouching/bending.<br>**3. Increase coasting persistence (`max_age`)** from 30 to 90 frames (3 seconds). |
| **Skeleton Misalignment & Crossing** | Out-of-distribution camera elevation; bilateral left/right limb swapping. | **1. Replace `bones_cross` hard-rejection with temporal limb smoothing**.<br>**2. Fine-tune pose estimator on top-down / CCTV surveillance datasets** (e.g. Overhead Pose / CrowdPose / CCTV-Pose). |
| **Identity Drops on Head Turns** | Reliance on 2D face recognition alone; missing OSNet model; brittle HSV histogram fallback. | **1. Download and deploy `osnet_x0_25_market1501.onnx`** to `edge/models/` for 360° appearance embedding.<br>**2. Implement Bay Identity Locking**: Once a technician is identified in a bay, lock their identity to that bay session until the track physically exits the polygon. |
| **Inanimate Object False Positives** | Textured objects triggering weak heuristics; dead `_is_inanimate` probe in tracker. | **1. Wire `_is_inanimate()` into `tracker.py`**: Kill tracks that have zero pixel displacement and zero motion energy over 100+ frames.<br>**2. Require minimum torso-to-head kinematic connectivity** before any box is admitted to the bay state machine. |

---

## 7. File & Code Reference Map

For ongoing research and debugging, the relevant source files and their responsibilities are:

- **[`edge/person.py`](file:///home/george/Documents/Inbound-Surveillance/edge/person.py)**:
  - `is_human_pose()`: Gatekeeper function for accepting/rejecting detector proposals.
  - `draw_skeleton()` & `draw_detection()`: Renders bounding boxes and skeleton lines on video frames.
  - `count_valid_bones()`, `skeleton_is_plausible()`: Kinematic graph and crossing-bone checks.
- **[`edge/tinypose.py`](file:///home/george/Documents/Inbound-Surveillance/edge/tinypose.py)**:
  - `PicoDetDetector.detect()`: Stage 1 pedestrian detector.
  - `TinyPoseEstimator.estimate()`: Stage 2 crop extraction, heatmap argmax, and subpixel coordinate decoding.
  - `PaddlePoseEngine`: Duck-typed wrapper matching Ultralytics YOLO interface.
- **[`edge/tracker.py`](file:///home/george/Documents/Inbound-Surveillance/edge/tracker.py)**:
  - `PersonTracker.update()`: ByteTrack association, Kalman filtering, track creation, and coasting.
  - `Track`: Data structure holding trajectory, keypoints, Re-ID features, and clutter flags.
  - `_match_reid()` & `_match_center()`: Secondary association passes when IoU fails.
- **[`edge/face_id.py`](file:///home/george/Documents/Inbound-Surveillance/edge/face_id.py)**:
  - `FaceRecognizer`: YuNet detection and SFace feature extraction.
  - `annotate_detections()`: Matches detected faces against `faces/<StaffName>/`.
- **[`edge/reid.py`](file:///home/george/Documents/Inbound-Surveillance/edge/reid.py)**:
  - `BodyReIDExtractor`: Extracts 512-dim OSNet embeddings or falls back to 8x8x8 HSV histograms.
  - `ReIDGallery`: Tracks appearance profiles for persistent identity matching.
- **[`edge/occupancy.py`](file:///home/george/Documents/Inbound-Surveillance/edge/occupancy.py)**:
  - `BayZoneManager.update()`: Bay state machine (`WORKING`, `UNDER_VEHICLE`, `IDLE`).
  - `is_working_pose()`, `is_under_vehicle_pose()`: Keypoint geometry heuristics for mechanic labor.
- **[`edge/launcher.py`](file:///home/george/Documents/Inbound-Surveillance/edge/launcher.py)**:
  - `LiveStreamEngine._run_infer_loop()`: Main background thread capturing video, executing inference, and broadcasting telemetry/MJPEG.
