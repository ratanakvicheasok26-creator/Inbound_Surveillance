"""Shoe-change / flip-flop station monitor.

At the spa entrance visitors take off their street shoes and put on indoor
flip-flops before going in. That reach-down-to-the-feet motion at the shoe rack
is a near-unique *customer* signal: staff walk in and out of the lobby without
changing footwear at the rack.

The monitor combines:

* a spatial prior -- the person's bbox center must be inside a configurable
  ``zone`` (the shoe rack / bench area); and
* a kinematic *reach-down* posture -- the closest visible *wrist* to any
  visible *ankle* must be within ``wrist_ankle_dist`` of the frame diagonal,
  with the hand below the hip. That is exactly the motion of pulling off a
  shoe or strapping on a flip-flop while bent forward.

Reach-down time spent inside the zone is accumulated per tracker ID. Once a
track has spent >= ``confirm_seconds`` in the posture the monitor fires one
``CUSTOMER`` event (subject to a per-track cooldown), persists a
``customer_shoe_change`` DB event, and optionally saves an annotated proof.

Low-volume oriented (a single shoe bench), mirroring the door traffic monitor.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import cv2

from db import insert_event
from proof import save_proof

from person import (
    NOSE,
    L_EYE,
    R_EYE,
    L_EAR,
    R_EAR,
    L_SHOULDER,
    R_SHOULDER,
    L_WRIST,
    R_WRIST,
    L_HIP,
    R_HIP,
    L_ANKLE,
    R_ANKLE,
)

HEAD_POINTS = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)
SHOULDER_POINTS = (L_SHOULDER, R_SHOULDER)
WRIST_POINTS = (L_WRIST, R_WRIST)
ANKLE_POINTS = (L_ANKLE, R_ANKLE)
HIP_POINTS = (L_HIP, R_HIP)


def _point_at(pts: list, index: int, conf: float) -> tuple[float, float] | None:
    try:
        x, y, c = pts[index]
    except (IndexError, ValueError, TypeError):
        return None
    if c >= conf:
        return x, y
    return None


def _person_visible(det: Any, conf: float) -> bool:
    """A real person at the rack: at least one head point and one shoulder.

    Deep forward bends (pulling off a shoe) often drop the nose below the
    shoulder line, so we deliberately do NOT require head-above-shoulders here.
    The wrist-to-ankle reach already separates hands-at-feet from clutter.
    """
    pts = list(getattr(det, "keypoints", None) or [])
    head = any(_point_at(pts, i, conf) is not None for i in HEAD_POINTS)
    shoulder = any(_point_at(pts, i, conf) is not None for i in SHOULDER_POINTS)
    return head and shoulder


def reach_down_distance(det: Any, frame_h: int, frame_w: int, conf: float = 0.30) -> float | None:
    """Normalized minimum wrist-to-ankle distance, or None if not a reach-down.

    Measures how far the visitor's hands are from their feet. A value at or
    below ``wrist_ankle_dist`` signals bending to put on / take off footwear.
    """
    pts = list(getattr(det, "keypoints", None) or [])
    wrists = [p for i in WRIST_POINTS if (p := _point_at(pts, i, conf)) is not None]
    ankles = [p for i in ANKLE_POINTS if (p := _point_at(pts, i, conf)) is not None]
    hips = [p for i in HIP_POINTS if (p := _point_at(pts, i, conf)) is not None]
    if not wrists or not ankles:
        return None
    diag = max(1.0, float(((frame_w ** 2) + (frame_h ** 2)) ** 0.5))
    best = None
    for wx, wy in wrists:
        for ax, ay in ankles:
            if hips:
                # hand should reach below the hip toward the foot
                lowest_hip = max(hy for _, hy in hips)
                if wy < lowest_hip:
                    continue
            dist = (((wx - ax) ** 2 + (wy - ay) ** 2) ** 0.5) / diag
            if best is None or dist < best:
                best = dist
    return best


def _cx(det: Any, w: int) -> float:
    x1 = float(getattr(det, "x1", 0.0))
    x2 = float(getattr(det, "x2", 0.0))
    if w <= 0:
        return 0.5
    if x2 > x1:
        return (x1 + x2) / 2.0 / w
    return (x1 + 0.05) / w


def _cy(det: Any, h: int) -> float:
    y1 = float(getattr(det, "y1", 0.0))
    y2 = float(getattr(det, "y2", 0.0))
    if h <= 0:
        return 0.5
    if y2 > y1:
        return (y1 + y2) / 2.0 / h
    return (y1 + 0.05) / h


def _tid(det: Any) -> int:
    return int(getattr(det, "track_id", 0) or 0)


@dataclass
class ShoeChangeEvent:
    kind: str                       # "customer"
    via: str                        # "shoe_change"
    ts: float                       # monotonic seconds
    stamp: datetime                 # wall clock
    cx: float
    cy: float
    track_id: int | None = None
    reach: float | None = None
    proof_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "via": self.via,
            "ts": self.stamp.isoformat(timespec="seconds"),
            "cx": round(self.cx, 3),
            "cy": round(self.cy, 3),
            "track_id": self.track_id,
            "reach": round(self.reach, 3) if self.reach is not None else None,
            "proof": self.proof_path,
        }


@dataclass
class _Track:
    last_seen: float
    last_cx: float
    last_cy: float
    hits: int = 0
    in_zone: bool = False
    reach_accum: float = 0.0
    last_reach_at: float = 0.0
    signaled: bool = False
    last_signal_at: float = 0.0
    customer: bool = False


class ShoeChangeMonitor:
    """Detect visitors changing shoes at the rack and flag them as customers."""

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        *,
        conn: Any = None,
        proofs_root=None,
    ) -> None:
        cfg = cfg or {}
        zone = cfg.get("zone") or [0.40, 0.40, 0.70, 1.0]
        self.zone = (
            float(zone[0]),
            float(zone[1]),
            float(zone[2]),
            float(zone[3]),
        )
        self.wrist_ankle_dist = float(cfg.get("wrist_ankle_dist", 0.06))
        self.confirm_seconds = float(cfg.get("confirm_seconds", 1.5))
        self.cooldown_seconds = float(cfg.get("cooldown_seconds", 30.0))
        self.relink_dist = float(cfg.get("relink_distance", 0.10))
        self.relink_seconds = float(cfg.get("relink_seconds", 8.0))
        self.kpt_conf = float(cfg.get("kpt_conf", 0.30))
        self.leave_seconds = float(cfg.get("leave_seconds", 4.0))
        self.require_person = bool(cfg.get("require_person", True))
        self.save_proofs = bool(cfg.get("proofs", True))
        self.conn = conn
        self.proofs_root = proofs_root

        self.tracks: dict[int, _Track] = {}
        self.events: deque[ShoeChangeEvent] = deque(maxlen=64)
        self._zone_roi_px: tuple[int, int, int, int] | None = None
        # recent (cx, cy, now) fires, to swallow tracker-ID re-label flicker
        self._fired_recent: deque[tuple[float, float, float]] = deque(maxlen=32)

    def _recently_fired(self, cx: float, cy: float, now: float) -> bool:
        """True if another track already signaled close to this spot recently.

        Swallows the tracker flickering a fresh track ID onto the same person
        still crouched at the rack (door_gate relink philosophy), while letting
        a second visitor at a different spot of the bench signal independently.
        """
        for fcx, fcy, fts in self._fired_recent:
            if (now - fts) > self.relink_seconds:
                continue
            if abs(cx - fcx) <= self.relink_dist and abs(cy - fcy) <= self.relink_dist:
                return True
        return False

    # ------------------------------------------------------------------ utils
    def _zone_roi(self, w: int, h: int) -> tuple[int, int, int, int]:
        x1 = max(0, int(self.zone[0] * w))
        y1 = max(0, int(self.zone[1] * h))
        x2 = min(w, int(self.zone[2] * w))
        y2 = min(h, int(self.zone[3] * h))
        return (x1, y1, x2, y2)

    def _in_zone(self, cx: float, cy: float) -> bool:
        zx1, zy1, zx2, zy2 = self.zone
        return zx1 <= cx <= zx2 and zy1 <= cy <= zy2

    def _save_proof(self, frame: Any, w: int, h: int, stamp: datetime) -> str | None:
        if not self.save_proofs or frame is None:
            return None
        if self._zone_roi_px is None:
            self._zone_roi_px = self._zone_roi(w, h)
        try:
            p = save_proof(frame, self._zone_roi_px, stamp, self.proofs_root, kind="customer_shoe")
            return str(p)
        except Exception as ex:
            print(f"[ShoeGate] proof save failed: {ex}")
            return None

    def _persist(self, event: ShoeChangeEvent) -> None:
        if self.conn is None:
            return
        try:
            insert_event(self.conn, "customer_shoe_change", event.stamp, event.proof_path)
        except Exception as ex:
            print(f"[ShoeGate] db insert failed: {ex}")

    # ------------------------------------------------------------- main update
    def update(
        self,
        detections: list[Any],
        w: int,
        h: int,
        now: float,
        *,
        frame: Any = None,
        stamp: datetime | None = None,
    ) -> list[ShoeChangeEvent]:
        stamp = stamp or datetime.now()
        fired: list[ShoeChangeEvent] = []
        if self._zone_roi_px is None:
            self._zone_roi_px = self._zone_roi(w, h)

        self._prune_tracks(now)

        if not detections:
            return fired

        for det in detections:
            tid = _tid(det)
            if tid <= 0:
                continue
            cx = _cx(det, w)
            cy = _cy(det, h)
            if not (-0.05 <= cx <= 1.05) or not (-0.05 <= cy <= 1.05):
                continue
            trk = self.tracks.get(tid)
            if trk is None:
                trk = _Track(last_seen=now, last_cx=cx, last_cy=cy)
                self.tracks[tid] = trk
            trk.in_zone = self._in_zone(cx, cy)

            if trk.in_zone and not trk.signaled:
                if self.require_person and not _person_visible(det, self.kpt_conf):
                    trk.hits += 1
                    continue
                reach = reach_down_distance(det, h, w, self.kpt_conf)
                if reach is not None and reach <= self.wrist_ankle_dist:
                    delta = max(0.0, min(now - trk.last_reach_at, 1.0)) if trk.last_reach_at else 0.0
                    trk.reach_accum += delta
                else:
                    trk.reach_accum = 0.0
                trk.last_reach_at = now

            if (
                trk.in_zone
                and not trk.signaled
                and trk.reach_accum >= self.confirm_seconds
            ):
                cooldown_ok = (now - trk.last_signal_at) >= self.cooldown_seconds
                not_relink = not self._recently_fired(cx, cy, now)
                if cooldown_ok and not_relink:
                    self._fire(det, trk, tid, cx, cy, now, stamp, frame, w, h, fired)
            trk.hits += 1
            trk.last_cx = cx
            trk.last_cy = cy
            trk.last_seen = now

        return fired

    def _fire(
        self,
        det: Any,
        trk: _Track,
        tid: int,
        cx: float,
        cy: float,
        now: float,
        stamp: datetime,
        frame: Any,
        w: int,
        h: int,
        fired: list[ShoeChangeEvent],
    ) -> None:
        trk.signaled = True
        trk.last_signal_at = now
        trk.customer = True
        self._fired_recent.append((cx, cy, now))
        try:
            det.is_customer = True
        except Exception:
            pass
        proof = self._save_proof(frame, w, h, stamp)
        ev = ShoeChangeEvent(
            "customer",
            "shoe_change",
            now,
            stamp,
            cx,
            cy,
            track_id=tid,
            reach=trk.reach_accum,
            proof_path=proof,
        )
        fired.append(ev)
        self.events.append(ev)
        self._persist(ev)
        print(f"[ShoeGate] CUSTOMER  {stamp.strftime('%H:%M:%S')}  via=shoe_change  track={tid}  cx={cx:.2f}")

    def _prune_tracks(self, now: float) -> None:
        for tid, trk in list(self.tracks.items()):
            if now - trk.last_seen < self.leave_seconds:
                continue
            del self.tracks[tid]

    # ------------------------------------------------------------------ drawing
    def annotate(self, frame: Any, w: int, h: int, now: float | None = None) -> None:
        if frame is None:
            return
        now = now or 0.0
        x1, y1, x2, y2 = self._zone_roi(w, h)
        ov = frame[y1:y2, x1:x2].copy()
        cv2.addWeighted(ov, 0.12, frame[y1:y2, x1:x2], 0.88, 0, frame[y1:y2, x1:x2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
        active = [t for t in self.tracks.values() if t.in_zone and t.reach_accum >= self.confirm_seconds]
        for t in active:
            px = int(t.last_cx * w)
            py = int(t.last_cy * h)
            cv2.circle(frame, (px, py), 26, (0, 220, 255), 2)
        total = sum(1 for t in self.tracks.values() if t.signaled)
        status = f"Customers flagged: {total}" if now else "ShoeChange monitor"
        cv2.putText(frame, status, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------- state
    def recent(self) -> list[dict[str, Any]]:
        return [e.as_dict() for e in self.events]

    def flagged_count(self) -> int:
        return sum(1 for t in self.tracks.values() if t.signaled)