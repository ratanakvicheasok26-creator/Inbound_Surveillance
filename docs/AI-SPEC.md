# AI-SPEC: Robust Multi-Person Biometrics & Single-Camera Physical Exclusivity

## Executive Summary
This design contract establishes the architectural and mathematical specification for the **Multi-Person Vision & Tracking Pipeline** in Inbound Surveillance. It resolves the "weak vision" problem and identity collapse (where multiple people or objects are assigned duplicate visitor IDs simultaneously) by enforcing the **Physical Invariant of Single-Camera Exclusivity** and a **Hierarchical Dual-Tier Biometric Pipeline**.

---

## 1. Problem Statement & Mathematical Diagnosis

### 1.1 The "Weak Vision" Phenomenon
In retail surveillance environments (e.g., convenience stores, supermarkets, workshops):
1. **Camera Angle Degradation**: Ceiling-mounted cameras view subjects at steep overhead angles ($\sim 35^\circ\text{--}50^\circ$). Faces are often occluded by tilted heads, hats, or backs turned to aisles.
2. **Background Dominance**: In full-body bounding boxes, store shelving, price tags, and linoleum floors account for up to $60\%\text{--}70\%$ of the image crop.
3. **Clothing Clustering**: Under standard retail fluorescent lighting, common dark jackets and denim produce very narrow feature separation in full-body Re-ID models (OSNet cosine similarity between distinct humans is often $0.60\text{--}0.75$).

### 1.2 The Single-Camera Physical Invariant
$$\forall \text{ detections } i, j \text{ at timestamp } t \text{ in camera } C: \quad i \neq j \implies \text{ID}(i) \neq \text{ID}(j)$$

In physical space, **one human being cannot occupy two coordinates in the same video frame**. A surveillance system must treat this as a non-negotiable hard constraint: **No single camera frame may ever output duplicate visitor IDs.**

---

## 2. Multi-Model Architecture & Pipeline Flow

The vision system operates in **three hierarchical layers**:

```
Raw Camera Frame (RGB)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ Layer 1: Person Detection & Keypoint Estimation           │
│ - YOLOv11-Nano / RTMPose                                  │
│ - Output: Bounding Boxes (x1, y1, x2, y2) + 17 Keypoints   │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ Layer 2: Real-Time Multi-Object Tracking (ByteTrack)      │
│ - Motion Estimation: Kalman Filter                        │
│ - Association: Bipartite IoU Matching                     │
│ - Clutter Gate: Liveness & Keypoint Jitter Verification   │
│ - Output: Stable Tracklets with persistent `track_id`     │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ Layer 3: Dual-Tier Identity Resolution                     │
│                                                           │
│  [Tier 1: Facial Biometrics (Ground Truth)]               │
│  - YuNet (~300 KB): Face & Landmark Detection             │
│  - SFace (37 MB): 128-d Clothing-Invariant Embedding      │
│  - Accuracy: Cross-person distance ~0.005                 │
│                                                           │
│  [Tier 2: Body Re-ID & Spatial-Temporal Stitching]        │
│  - OSNet-x0.25: 512-d Appearance Representation           │
│  - Active Only When Face Is Occluded                      │
│  - 1-to-1 Bipartite Frame Mutex (No Duplicate IDs)        │
└───────────────────────────────────────────────────────────┘
```

---

## 3. Core Operating Laws for Identity Matching

### Law 1: Frame-Exclusivity Mutex (No Co-Occurrence Collisions)
When resolving visitor identities in frame $t$:
1. A frame-level registry `claimed_ids` tracks all identities assigned in the current frame.
2. If candidate identity $K$ is already in `claimed_ids`, it is **strictly barred** from all subsequent detections in frame $t$.
3. If an existing tracklet collision is detected (two active tracks holding the same cached ID), the tracklet with higher spatial-temporal confidence retains the ID; the conflicting tracklet is immediately assigned a new unique visitor identity.

### Law 2: Anti-Poisoning Gallery Enrollment
1. Gallery profiles must maintain **appearance centroid integrity**.
2. New embeddings may only be appended to an existing subject's template gallery if cosine similarity to the existing prototype clears a conservative threshold:
   $$\text{sim}(\mathbf{v}_{\text{new}}, \mathbf{v}_{\text{proto}}) \ge 0.70$$
3. Marginal matches ($< 0.70$) must **never** be used to update or mutate the gallery templates, preventing the formation of "black-hole" clusters that swallow disparate people.

### Law 3: Inanimate Clutter Rejection
1. Static objects (hanging jackets on racks, display stands, cardboard shelves) that trigger false YOLO person detections must be filtered before identity assignment.
2. Detections flagged as `clutter=True` by the tracker's liveness probe (motion energy $\le 0.05$, keypoint jitter $\le 0.001$) or unconfirmed detections (`hits < 2`) are excluded from visitor minting.

---

## 4. Evaluation Strategy & Verification Contract

### Automated Verification Gates
1. **Multi-Subject Co-Occurrence Test**:
   - Present $N$ simultaneous people in frame $t$.
   - Assert $N$ distinct unique IDs are generated: $|\{\text{ID}_1, \dots, \text{ID}_N\}| = N$.
2. **Tracklet Continuity Test**:
   - Maintain tracklet identities across $> 60$ frames under camera noise and jitter.
3. **Cross-Day Clothing Change Test**:
   - Verify that YuNet + SFace correctly links a returning customer wearing different outfits across days.
4. **Clutter Suppression Test**:
   - Verify that hanging clothes and inanimate displays never produce visitor session rows in SQLite.
