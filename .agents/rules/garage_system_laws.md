# Inbound Garage Core Operating Rules & Architecture Laws

> **SUPREME LAW REFERENCE:** All work on this system is governed by [SYSTEM_ARCHITECTURE_LAWS.md](file:///home/george/Documents/Inbound-Surveillance/.agents/rules/SYSTEM_ARCHITECTURE_LAWS.md).  
> **LAW 0 (STOP AND REPORT):** If an edit risks frame drops, tracking loss, or regressions, STOP and warn the user before writing code.

These architectural laws prevent regressions and ensure the system accurately mirrors real-world garage operations:

---

## 1. ⏱️ Timer & Bay State Machine Law
- **Inside Bay (Active Wrenching / Underbody)**:
  - Must continuously accumulate `wrench_seconds` (active repair time).
- **Stepping Outside Bay (Break / Tool Fetching)**:
  - Must immediately freeze `wrench_seconds` and transition state to `ON_BREAK`.
  - Must accumulate `break_seconds` while outside the bay.
- **Re-entering Bay**:
  - Must seamlessly resume `wrench_seconds` from where it left off (never reset back to 0 during an active job).

---

## 2. 👥 Multi-Mechanic per Bay Law
- Never assume a 1:1 relationship between bays and mechanics.
- A single car/bay can have **2 or 3 mechanics** working simultaneously.
- Always track each technician independently in `technicians_times: dict[str, float]` and display individual durations: e.g. `Hour-Meng (4m 03s), Sothun (2m 15s)`.

---

## 3. 🚗 Dual Neural Network Inference Law
- **`yolo11n-pose.pt`**: Specialized solely for human pose keypoints (Class 0 = `person`). Does NOT detect vehicles.
- **`yolo11n.pt`**: Full COCO detector loaded specifically for `car` (2), `motorcycle` (3), `bus` (5), `truck` (7).
- Both models must run together so mechanics and vehicles are recognized simultaneously.
- Use sensitive vehicle threshold (`conf=0.18`) so cars on phone screens and in low light are instantly captured.

---

## 4. 🛡️ Vehicle Departure Grace Period Law
- When a mechanic stands in front of a car, the car may be momentarily occluded for 1–2 seconds.
- The `VehicleTracker` must enforce a minimum **15-second departure grace window** before declaring a car departed and auto-closing the repair job.

---

## 5. 🎨 "Digital Overwatch" Brand Theme Law
All interface components, glass HUD overlays, and OpenCV video annotations must strictly follow:
- **Absolute Black (`#000000`)**: Deep infinite background.
- **Surface Charcoal (`#121212`)**: Cards, panels, modal windows, glass HUD.
- **Surveillance Green (`#00FF66`)**: Primary action color, active mechanic badges, live dots, buttons.
- **Stealth Green (`#0B833A`)**: Secondary darker green, subtle borders, paused break indicators.
- **Crisp White (`#FAFAFA`)**: High-contrast typography and core lettering.

---

## 6. 👤 Terminology Law
- Unassigned / unknown persons must always be labeled **`Employee`** (never `Customer`).
