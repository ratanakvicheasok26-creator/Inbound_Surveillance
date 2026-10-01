"""Role and Uniform Classifier for Inbound Surveillance.

Classifies person detections into:
- 'customer': Real visiting customers / guests.
- 'security_guard': Bodyguards, security guards, door guards on duty.
- 'spa_staff': Spa therapists, receptionists, cleaners.
- 'delivery': Couriers, delivery drivers.

Combines visual appearance (uniform/clothing features) and behavioral duty
tracking (entrance guard dwell) so on-duty bodyguards are never falsely counted
as customers.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("role_classifier")

ROLE_CUSTOMER = "customer"
ROLE_GUARD = "security_guard"
ROLE_STAFF = "spa_staff"
ROLE_DELIVERY = "delivery"

# Guard duty thresholds
DEFAULT_GUARD_DWELL_SECONDS = 25.0  # Persistent entrance dwell indicates guard on duty
DEFAULT_GUARD_CONF_THRESHOLD = 0.55


@dataclass
class RoleVerdict:
    role: str
    confidence: float
    is_staff: bool
    is_guard: bool
    is_customer: bool
    details: Dict[str, Any] = field(default_factory=dict)


class GuardDwellTracker:
    """Tracks stationary or pacing behavior at entrance/parking areas.
    
    Customers walk through the entrance towards reception (transit time < 15s).
    Bodyguards and security personnel stay stationed at the entrance/perimeter
    for extended periods. Once a person exceeds the guard dwell threshold, they
    are automatically classified as an on-duty security guard.
    """

    def __init__(self, guard_dwell_s: float = DEFAULT_GUARD_DWELL_SECONDS) -> None:
        self.guard_dwell_s = guard_dwell_s
        # track_id -> {'first_seen': float, 'last_seen': float, 'positions': list, 'is_guard': bool, 'bounds': [min_x, max_x, min_y, max_y]}
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._update_counter = 0

    def update(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        now: float | None = None,
        in_guard_zone: bool = True,
    ) -> bool:
        """Update track position and return True if this person is confirmed as guard/staff."""
        t = time.time() if now is None else float(now)
        if track_id <= 0:
            return False

        self._update_counter += 1
        if self._update_counter % 200 == 0 and len(self._tracks) > 50:
            self.prune(t, max_age=180.0)

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        if track_id not in self._tracks:
            self._tracks[track_id] = {
                "first_seen": t,
                "last_seen": t,
                "positions": [(cx, cy, t)],
                "bounds": [cx, cx, cy, cy],
                "is_guard": False,
            }
            return False

        data = self._tracks[track_id]
        data["last_seen"] = t
        positions = data["positions"]
        positions.append((cx, cy, t))

        bounds = data.setdefault("bounds", [cx, cx, cy, cy])
        bounds[0] = min(bounds[0], cx)
        bounds[1] = max(bounds[1], cx)
        bounds[2] = min(bounds[2], cy)
        bounds[3] = max(bounds[3], cy)

        # Keep position history bounded to prevent memory leaks
        if len(positions) > 60:
            del positions[0 : len(positions) - 60]

        if data["is_guard"]:
            return True

        dwell = t - data["first_seen"]
        if dwell >= self.guard_dwell_s and in_guard_zone:
            spread_x = bounds[1] - bounds[0]
            spread_y = bounds[3] - bounds[2]
            total_spread = math.sqrt(spread_x * spread_x + spread_y * spread_y)

            # Stationary or localized pacing at the entrance/guard post
            if total_spread < 350.0:  # within ~350px radius over extended time
                data["is_guard"] = True
                return True

        return data["is_guard"]

    def prune(self, current_time: float, max_age: float = 300.0) -> None:
        """Prune tracks that have disappeared."""
        stale = [
            tid
            for tid, d in self._tracks.items()
            if (current_time - d["last_seen"]) > max_age
        ]
        for tid in stale:
            del self._tracks[tid]



class RoleClassifier:
    """Visual appearance and attribute classifier for person roles."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        guard_dwell_s: float = DEFAULT_GUARD_DWELL_SECONDS,
    ) -> None:
        self.dwell_tracker = GuardDwellTracker(guard_dwell_s=guard_dwell_s)
        self.model = None
        self.model_path = Path(model_path) if model_path else None
        self._init_model()

    def _init_model(self) -> None:
        if self.model_path is None:
            # Check default trained model paths
            for candidate in (
                Path("models/yolo11n-role-cls.pt"),
                Path("models/yolo11n-role-cls_openvino_model"),
                Path("models/yolo11n-role-cls.onnx"),
            ):
                if candidate.exists():
                    self.model_path = candidate
                    break

        if self.model_path and self.model_path.exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(self.model_path))
                logger.info(f"[RoleClassifier] Loaded trained role model: {self.model_path.name}")
            except Exception as exc:
                logger.warning(f"[RoleClassifier] Failed to load {self.model_path}: {exc}")

    def classify_crop(self, crop: np.ndarray) -> Tuple[str, float]:
        """Classify a single person image crop into role and confidence."""
        if crop is None or crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 10:
            return ROLE_CUSTOMER, 0.50

        # 1. If trained YOLO-cls model is available, use it
        if self.model is not None:
            try:
                res = self.model(crop, verbose=False)[0]
                top1_idx = int(res.probs.top1)
                top1_conf = float(res.probs.top1conf.cpu().numpy())
                top1_name = res.names.get(top1_idx, ROLE_CUSTOMER)
                return top1_name, top1_conf
            except Exception as exc:
                logger.debug(f"[RoleClassifier] Inference error: {exc}")

        # 2. Heuristic appearance analysis (Color/Tone & Uniform features)
        # Guards typically wear dark/black uniform shirts, dark trousers, tactical vests
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, w = crop.shape[:2]
        torso_crop = hsv[int(h * 0.15) : int(h * 0.55), int(w * 0.2) : int(w * 0.8)]

        if torso_crop.size > 0:
            # Value (brightness) and Saturation in torso area
            v_channel = torso_crop[:, :, 2]
            s_channel = torso_crop[:, :, 1]
            dark_ratio = float(np.mean(v_channel < 60))
            muted_ratio = float(np.mean(s_channel < 50))

            # Dark uniform (black shirt/polo + dark attire)
            if dark_ratio > 0.55 and muted_ratio > 0.60:
                return ROLE_GUARD, 0.72

        return ROLE_CUSTOMER, 0.60

    def evaluate(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        track_id: int = 0,
        now: float | None = None,
        in_guard_zone: bool = True,
    ) -> RoleVerdict:
        """Evaluate a person detection and return a comprehensive RoleVerdict."""
        t = time.time() if now is None else float(now)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_f, w_f = frame.shape[:2]
        crop = frame[max(0, y1) : min(h_f, y2), max(0, x1) : min(w_f, x2)]

        visual_role, visual_conf = self.classify_crop(crop)

        # Update guard dwell tracker
        is_dwell_guard = self.dwell_tracker.update(
            track_id=track_id,
            bbox=bbox,
            now=t,
            in_guard_zone=in_guard_zone,
        )

        # Decide final role
        final_role = visual_role
        final_conf = visual_conf

        if is_dwell_guard:
            final_role = ROLE_GUARD
            final_conf = max(final_conf, 0.90)
        elif visual_role == ROLE_GUARD and visual_conf >= DEFAULT_GUARD_CONF_THRESHOLD:
            final_role = ROLE_GUARD

        is_staff = final_role in (ROLE_GUARD, ROLE_STAFF)
        is_guard = final_role == ROLE_GUARD
        is_customer = final_role == ROLE_CUSTOMER

        return RoleVerdict(
            role=final_role,
            confidence=final_conf,
            is_staff=is_staff,
            is_guard=is_guard,
            is_customer=is_customer,
            details={
                "visual_role": visual_role,
                "visual_conf": round(visual_conf, 2),
                "is_dwell_guard": is_dwell_guard,
            },
        )
