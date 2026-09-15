# Implementation Plan: Multi-Modal Identity Continuity, Throttled 360° ReID, and Bay State Machine Hardening

## Overview
Building upon the verified foundation of the **Edge Pose Tracking Plan** (8D size-aware Kalman filter, ByteTrack high/low association, OneEuro keypoint stabilization, and bilateral limb swap), this phase delivers production-grade identity continuity and bay labor integrity.

This plan incorporates critical operational adjustments to eliminate performance traps and cross-camera collision hazards:
1. **Model Resolution Lock (640x640 pose / 512x512 vehicle)** and OpenVINO thread-pinning are prioritized upfront to secure a $\ge 25\text{ FPS}$ foundation.
2. **Throttled, Event-Gated OSNet Extraction** (capped to at most once per 30–45 frames, or strictly on spatial match drops) prevents the 70–140 ms edge CPU hot-loop bottleneck.
3. **Spatial Exclusivity Constraints** in `PersistentReIDGallery` prevent dark-uniform cross-camera identity misattributions across simultaneous bay camera streams.
4. **Three-Tier Bay Labor Admission & Whitelisted Creeper Support** blocks inanimate clutter (jack stands, tires, shoes) while preserving mechanics under chassis.
5. **30-Second Occlusion Dwell with Boundary Exit Short-Circuit** provides seamless labor continuity while immediately terminating sessions when a technician physically walks out.

---

## Architecture Decisions

### 1. Upfront Model Resolution Lock & OpenVINO Thread Pinning
- **Hard-lock Export & Inference Resolutions**:
  - `yolo11n-pose`: $640 \times 640$ (reduces proposal grid from 18,900 down to 8,400 proposals vs $960$).
  - `yolo11n` (vehicle): $512 \times 512$ at decoupled 1.0s cadence.
- **Thread Pinning**:
  - Restrict OpenVINO / ONNX CPU inference threads to physical core count (e.g. `num_threads=4`) in `runtime.py` to eliminate thread thrashing and context-switching overhead.
- **Execution Order Rationale**:
  - Performance must be stabilized first. Testing tracking continuity, Re-ID, and bay state hysteresis on an engine running below 2 FPS generates timing artifacts that invalidate test metrics.

### 2. Throttled & Event-Gated Re-ID (Preventing the OSNet Hot-Loop)
- **Problem**: Running 512-dim OSNet on edge CPU takes 70–140 ms per crop. Continuous extraction drops frame rate to 3–6 FPS.
- **Solution**:
  - **Never extract on every frame**.
  - Extract appearance embeddings *only* when:
    1. A new track confirms (`hits >= 3`) and has no initial embedding (`track.features is None`), OR
    2. Spatial IoU and Kalman matching fail completely and track recovery is required, OR
    3. As a background refresh throttled to at most once every **30–45 frames** (2.0–3.0 seconds).
  - Degenerate crop guard: Skip extraction if $w < 20\text{ px}$ or $h < 40\text{ px}$.

### 3. Cross-Camera Spatial Exclusivity in Shared Gallery
- **Problem**: Mechanics wearing identical dark-blue/black shop uniforms can trigger false cross-camera Re-ID matches when matching solely against appearance embeddings ($\ge 0.65$).
- **Solution**:
  - `PersistentReIDGallery` tracks which camera stream currently holds a confirmed active technician.
  - If Technician A has an active, confirmed track in Camera 1 (Bay 1), Camera 2 (Bay 3) **cannot** assign Technician A's identity to an ambiguous track unless Camera 1 registers a departure or total track loss for $> 5.0\text{ seconds}$.
  - Gallery enrollment is strictly anti-poisoned: Requires face confidence $\ge 0.70$ and upright aspect ratio ($H/W \ge 1.0$).
  - FIFO eviction caps each technician profile to 16 embeddings.

### 4. Three-Tier Bay Admission Gating & Creeper Whitelisting
- In `BayZoneManager.update()`, admissions must pass:
  1. **Track Confirmation**: Track must be confirmed (`hits >= 3`) and not flagged as clutter.
  2. **Kinematic Torso Connectivity**: Shoulders and hips must form a connected graph (isolated shoe/ankle pairs are rejected).
  3. **Motion / Jitter Proof**: Non-zero motion energy or joint variance over a temporal window.
- **Whitelist**: `is_creeper_or_underbody_pose` is explicitly whitelisted to preserve legitimate floor mechanics.

### 5. 30-Second Bay Occlusion Hysteresis with Polygon Exit Short-Circuit
- Mechanics crawling under chassis or occluded behind vehicle pillars retain active `WORKING` / `UNDER_VEHICLE` state and locked technician name for up to 30 seconds.
- **Boundary Exit Short-Circuit**: If the track's bounding box is observed leaving the bay polygon boundary, the session closes immediately without waiting for the 30s timeout.

---

## Dependency Graph & Execution Order

```
Phase 1: Performance Baseline & Resolution Lock
  │   - Task 1: Model Resolution Lock (640x640 / 512x512) & OpenVINO Thread Pinning
  ▼
Checkpoint 1: Performance Foundation (>= 25 FPS verified)
  │
Phase 2: Persistent ReID with Spatial Exclusivity & Throttling
  │   - Task 2: Persistent Inter-Camera ReID Gallery with Spatial Exclusivity
  │   - Task 3: Throttled & Event-Gated 360° Re-ID Handover (30-frame cap, crop gate)
  ▼
Checkpoint 2: ReID & Multi-Camera Continuity
  │
Phase 3: Bay State Machine Hardening & Occlusion Hysteresis
  │   - Task 4: Strict Three-Tier Bay Labor Admission Gate & Anti-Clutter
  │   - Task 5: 30-Second Bay Occlusion Hysteresis & Polygon Exit Short-Circuit
  ▼
Checkpoint 3: Bay State Machine & Labor Integrity
  │
Phase 4: Sidecar Packaging & End-to-End Verification
  │   - Task 6: Sidecar PyInstaller Build & Dependency Verification (DirectML/OpenVINO)
  │   - Task 7: Virtual Camera Multi-Stream Regression Benchmark (FPS, ID switches)
  ▼
Checkpoint 4: Complete System Validation
```

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| OSNet Hot-Loop Latency Collapse | Critical | Throttled extraction capped at 30–45 frames; trigger on-demand only during spatial drop; reject degenerate crops ($w < 20, h < 40$). |
| Cross-Camera Uniform False ReID | High | Enforce spatial exclusivity: camera 2 cannot claim an identity already confirmed active on camera 1 without 5s departure. |
| Inadvertent Veto of Underbody Mechanics | High | Whitelist `is_creeper_or_underbody_pose` and preserve `tracker.protected_ids`. |
| Excessive Occlusion Dwell Delaying Bay Vacancy | Medium | Short-circuit the 30-second dwell if the technician's track is explicitly observed crossing out of the bay polygon. |
| OpenVINO Thread Over-Subscription | Medium | Explicitly configure `INFERENCE_NUM_THREADS = 4` in `runtime.py`. |

---

## Open Questions
- None blocking. All constraints, guardrails, and architectural priorities have been aligned with production requirements.
