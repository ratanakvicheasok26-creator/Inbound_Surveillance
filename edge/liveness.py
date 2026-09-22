"""Per-detection proof that a box belongs to something animate.

Single-frame pose geometry cannot separate a mechanic leaning into an engine
bay from the engine itself — both project a plausible head-and-shoulders graph
onto a COCO skeleton. Time can. A person breathes, shifts weight, and makes the
pose regression head wobble; a jack stand under fixed shop lighting reprojects
to the same pixels and the same joints every frame.

Two independent signals, neither of which an inanimate object can fake:

- ``box_motion_energy`` — mean absolute grayscale change inside the box.
- ``keypoint_jitter`` — mean joint displacement as a fraction of box diagonal.
"""

from __future__ import annotations

import cv2
import numpy as np

# Mean abs grayscale delta below this is sensor/compression noise, not a body.
DEAD_MOTION_ENERGY = 2.0
# Joint wobble below this fraction of the box diagonal is a rigid reprojection.
DEAD_KEYPOINT_JITTER = 0.015
DEFAULT_SCALE = 0.5
_MIN_PROBE_PX = 6


def _clip_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(int(width), int(box[2]))
    y2 = min(int(height), int(box[3]))
    if x2 - x1 < _MIN_PROBE_PX or y2 - y1 < _MIN_PROBE_PX:
        return None
    return x1, y1, x2, y2


def to_probe_gray(frame, scale: float = DEFAULT_SCALE):
    """Downscaled grayscale plane. One conversion per frame serves every box."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if scale >= 0.999:
        return gray
    h, w = gray.shape[:2]
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)


def box_motion_energy(
    gray,
    prev_gray,
    box: tuple[float, float, float, float],
    scale: float = DEFAULT_SCALE,
) -> float | None:
    """Mean absolute grayscale delta inside the box. None when unmeasurable."""
    if gray is None or prev_gray is None:
        return None
    if gray.shape != prev_gray.shape:
        return None
    h, w = gray.shape[:2]
    scaled = (box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale)
    clipped = _clip_box(scaled, w, h)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    diff = cv2.absdiff(gray[y1:y2, x1:x2], prev_gray[y1:y2, x1:x2])
    return float(np.mean(diff))


def keypoint_jitter(
    prev_keypoints: list,
    keypoints: list,
    diag: float,
    kpt_conf: float = 0.35,
) -> float | None:
    """Mean displacement of jointly visible joints, as a fraction of the diagonal."""
    if not prev_keypoints or not keypoints or diag <= 0:
        return None
    n = min(len(prev_keypoints), len(keypoints))
    dists: list[float] = []
    for i in range(n):
        px, py, pc = prev_keypoints[i]
        cx, cy, cc = keypoints[i]
        if pc < kpt_conf or cc < kpt_conf:
            continue
        dists.append(float(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5))
    if not dists:
        return None
    return float(sum(dists) / len(dists) / diag)


class LivenessProbe:
    """Holds the previous frame so callers only pass the current one."""

    def __init__(self, scale: float = DEFAULT_SCALE) -> None:
        self.scale = float(scale)
        self._prev_gray = None

    def reset(self) -> None:
        self._prev_gray = None

    def annotate(self, frame, detections: list) -> None:
        """Set ``det.liveness`` on each detection, then remember the frame."""
        gray = to_probe_gray(frame, self.scale)
        for det in detections or []:
            try:
                det.liveness = box_motion_energy(gray, self._prev_gray, det.box(), self.scale)
            except Exception:
                det.liveness = None
        if gray is not None:
            self._prev_gray = gray
