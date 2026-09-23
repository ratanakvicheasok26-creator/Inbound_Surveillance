"""Door traffic monitor: times when customers enter/exit the spa lobby door.

The door is at the RIGHT edge of the frame by default. The monitor combines:

* a vertical *tripwire line* (``line_x``) where a tracked person's direction of
  travel decides ENTER (right -> left, walking in from the door) vs EXIT
  (left -> right, walking out toward the door); and
* *anchor events* for doors that sit just at the edge of the frame:
  a new track appearing inside the door-side ``zone_x`` counts as an ENTER,
  and an entered track that was last seen inside the zone and then vanishes
  counts as an EXIT.

Pairing an ENTER with its later EXIT produces a per-visit session with a dwell
time. Low-volume oriented (a spa lobby), with a hard cooldown to swallow the
tracker's ID re-assignment flicker.
"""

from __future__ import annotations

import cv2
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from db import insert_event
from proof import save_proof

MOTION_BLUR = 0.5


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
class DoorEvent:
    kind: str                       # "enter" | "exit"
    via: str                        # "tripwire" | "appear" | "disappear"
    ts: float                       # monotonic seconds
    stamp: datetime                 # wall clock
    cx: float
    cy: float
    track_id: int | None = None
    session_id: int | None = None
    dwell_s: float | None = None
    proof_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "via": self.via,
            "ts": self.stamp.isoformat(timespec="seconds"),
            "cx": round(self.cx, 3),
            "cy": round(self.cy, 3),
            "track_id": self.track_id,
            "session_id": self.session_id,
            "dwell_s": self.dwell_s,
            "proof": self.proof_path,
        }


@dataclass
class _Track:
    last_seen: float
    last_cx: float
    last_cy: float
    side: str | None = None          # side of the tripwire line seen last
    hits: int = 0
    in_zone: bool = False
    entered: bool = False
    entered_at: float = 0.0
    session_id: int | None = None
    heading: str | None = None       # dominant recent direction ("L"/"R")
    dir_streak: int = 0
    pending: str | None = None       # "enter" | "exit" awaiting confirmation
    pending_dir: str | None = None   # heading that confirms it
    pending_at: float = 0.0
    last_stamp: datetime | None = None
    proof_enter: str | None = None


@dataclass
class _Session:
    session_id: int
    entered_at: float
    entered_stamp: datetime
    entered_cx: float
    entered_cy: float
    exited_at: float | None = None
    exited_stamp: datetime | None = None
    exit_cx: float | None = None
    dwell_s: float | None = None
    track_ids: set[int] = field(default_factory=set)
    closed: bool = False
    proof: str | None = None
    via: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "entered": self.entered_stamp.isoformat(timespec="seconds"),
            "exited": self.exited_stamp.isoformat(timespec="seconds") if self.exited_stamp else None,
            "dwell_s": round(self.dwell_s, 1) if self.dwell_s is not None else None,
            "entered_cx": round(self.entered_cx, 3),
            "exited_cx": round(self.exit_cx, 3) if self.exit_cx is not None else None,
            "proof": self.proof,
        }


class DoorTrafficMonitor:
    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        *,
        conn: Any = None,
        proofs_root=None,
    ) -> None:
        cfg = cfg or {}
        self.line_x = float(cfg.get("line_x", 0.55))
        zone = cfg.get("zone") or [0.50, 1.0]
        self.zone = (float(zone[0]), float(zone[1]))
        self.enter_direction = str(cfg.get("enter_direction", "right_to_left"))
        self.confirm_obs = max(2, int(cfg.get("confirm_obs", 3)))
        self.pending_timeout = float(cfg.get("pending_timeout", 6.0))
        self.cooldown_s = float(cfg.get("cooldown_seconds", 12.0))
        self.leave_s = float(cfg.get("leave_seconds", 5.0))
        self.relink_s = float(cfg.get("relink_seconds", 30.0))
        self.relink_dist = float(cfg.get("relink_distance", 0.28))
        self.motion_confirm = bool(cfg.get("motion_confirm", False))
        self.motion_thresh = float(cfg.get("motion_thresh", 2.5))
        self.max_sessions = max(1, int(cfg.get("max_sessions", 2)))
        self.save_proofs = bool(cfg.get("proofs", True))
        self.conn = conn
        self.proofs_root = proofs_root

        self.tracks: dict[int, _Track] = {}
        self.sessions: list[_Session] = []
        self._next_session = 1
        self.events: deque[DoorEvent] = deque(maxlen=64)
        self._last_enter_ts = 0.0
        self._last_exit_ts = 0.0
        self._prev_gray: Any = None
        self.door_motion = 0.0
        self._zone_roi_px: tuple[int, int, int, int] | None = None

    # ------------------------------------------------------------------ utils
    def _zone_roi(self, w: int, h: int) -> tuple[int, int, int, int]:
        zx1, zx2 = self.zone
        x1 = max(0, int(zx1 * w))
        x2 = min(w, int(zx2 * w))
        return (x1, 0, x2, h)

    def _in_zone(self, cx: float) -> bool:
        return self.zone[0] <= cx <= self.zone[1]

    def _cooldown_ok(self, kind: str, now: float) -> bool:
        last = self._last_enter_ts if kind == "enter" else self._last_exit_ts
        return (now - last) >= self.cooldown_s

    def _mark(self, kind: str, now: float) -> None:
        if kind == "enter":
            self._last_enter_ts = now
        else:
            self._last_exit_ts = now

    def _save_proof(self, frame: Any, w: int, h: int, stamp: datetime, kind: str) -> str | None:
        if not self.save_proofs or frame is None:
            return None
        if self._zone_roi_px is None:
            self._zone_roi_px = self._zone_roi(w, h)
        try:
            p = save_proof(frame, self._zone_roi_px, stamp, self.proofs_root, kind=kind)
            return str(p)
        except Exception as ex:
            print(f"[DoorGate] proof save failed: {ex}")
            return None

    def _persist(self, event: DoorEvent) -> None:
        if self.conn is None:
            return
        try:
            insert_event(self.conn, f"customer_{event.kind}", event.stamp, event.proof_path)
        except Exception as ex:
            print(f"[DoorGate] db insert failed: {ex}")

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
    ) -> list[DoorEvent]:
        stamp = stamp or datetime.now()
        fired: list[DoorEvent] = []
        self._update_door_motion(frame, w, h)
        self._prune_tracks(now, stamp, fired)

        if not detections:
            return fired

        live: set[int] = set()
        for det in detections:
            tid = _tid(det)
            if tid <= 0:
                continue
            live.add(tid)
            cx = _cx(det, w)
            cy = _cy(det, h)
            if not (-0.05 <= cx <= 1.05) or not (-0.05 <= cy <= 1.05):
                continue  # ignore boxes extrapolated outside the frame (partly off-frame / coasting)
            trk = self.tracks.get(tid)
            if trk is None:
                trk = _Track(last_seen=now, last_cx=cx, last_cy=cy, last_stamp=stamp)
                self.tracks[tid] = trk
            side = "R" if cx >= self.line_x else "L"

            if trk.hits == 1 and self._in_zone(cx):
                self._maybe_enter(trk, tid, cx, cy, now, stamp, frame, w, h, fired)

            if trk.hits >= 1:
                new_heading = "R" if cx > trk.last_cx else "L"
                if new_heading == trk.heading:
                    trk.dir_streak += 1
                else:
                    trk.heading = new_heading
                    trk.dir_streak = 1

            # a track flipping across the tripwire line starts a confirmed
            # crossing; it only fires once it keeps moving in that direction
            if trk.hits >= 2 and trk.side is not None and trk.side != side:
                if trk.side == "R" and side == "L":
                    cand = "enter" if self.enter_direction == "right_to_left" else "exit"
                else:
                    cand = "exit" if self.enter_direction == "right_to_left" else "enter"
                trk.pending = cand
                trk.pending_dir = "L" if cand == "enter" else "R"
                trk.pending_at = now

            if trk.pending is not None:
                confirmed_heading = (
                    trk.heading == trk.pending_dir and trk.dir_streak >= self.confirm_obs
                )
                turned_back = trk.heading is not None and trk.heading != trk.pending_dir
                timed_out = now - trk.pending_at > self.pending_timeout
                if confirmed_heading:
                    if trk.pending == "enter":
                        self._fire_enter(trk, tid, cx, cy, now, stamp, frame, w, h, fired, via="tripwire", force=False)
                    else:
                        self._fire_exit(trk, tid, cx, cy, now, stamp, frame, w, h, fired, via="tripwire")
                    trk.pending = None
                elif turned_back or timed_out:
                    trk.pending = None

            trk.side = side
            trk.last_cx = cx
            trk.last_cy = cy
            trk.last_seen = now
            trk.last_stamp = stamp
            trk.in_zone = self._in_zone(cx)
            trk.hits += 1

        # force-close sessions held by tracks that no longer report
        for tid, trk in list(self.tracks.items()):
            if tid not in live:
                continue
        return fired

    def _maybe_enter(self, trk: _Track, tid: int, cx: float, cy: float, now: float, stamp: datetime,
                     frame: Any, w: int, h: int, fired: list[DoorEvent]) -> None:
        if trk.entered:
            return
        if not self._cooldown_ok("enter", now):
            return
        if self.motion_confirm and self.door_motion < self.motion_thresh:
            return
        self._fire_enter(trk, tid, cx, cy, now, stamp, frame, w, h, fired, via="appear", force=False)

    def _fire_enter(self, trk: _Track, tid: int, cx: float, cy: float, now: float, stamp: datetime,
                    frame: Any, w: int, h: int, fired: list[DoorEvent], *, via: str, force: bool) -> None:
        if trk.entered and not force:
            return
        if not force and not self._cooldown_ok("enter", now):
            return
        # try to bind to a recent open session instead of opening a brand-new one
        bound = self._relink(tid, cx, cy, now, stamp)
        if bound is not None:
            trk.entered = True
            trk.session_id = bound.session_id
            bound.track_ids.add(tid)
            trk.proof_enter = bound.proof
            return
        if sum(1 for s in self.sessions if not s.closed) >= self.max_sessions:
            self._force_close_oldest(now, stamp, frame, w, h, fired)
        sid = self._next_session
        self._next_session += 1
        proof = self._save_proof(frame, w, h, stamp, "enter")
        sess = _Session(
            session_id=sid,
            entered_at=now,
            entered_stamp=stamp,
            entered_cx=cx,
            entered_cy=cy,
            track_ids={tid},
            proof=proof,
            via=via,
        )
        self.sessions.append(sess)
        trk.entered = True
        trk.entered_at = now
        trk.session_id = sid
        trk.proof_enter = proof
        ev = DoorEvent("enter", via, now, stamp, cx, cy, track_id=tid, session_id=sid, proof_path=proof)
        fired.append(ev)
        self.events.append(ev)
        self._mark("enter", now)
        self._persist(ev)
        print(f"[DoorGate] CUSTOMER ENTER #{sid}  {stamp.strftime('%H:%M:%S')}  via={via} cx={cx:.2f}")

    def _fire_exit(self, trk: _Track, tid: int, cx: float, cy: float, now: float, stamp: datetime,
                   frame: Any, w: int, h: int, fired: list[DoorEvent], *, via: str) -> None:
        if not trk.entered and self._cooldown_ok("enter", now) and via == "tripwire":
            pass
        if not self._cooldown_ok("exit", now):
            return
        sess = next((s for s in self.sessions if not s.closed and s.session_id == trk.session_id), None)
        sid = None
        dwell = None
        proof = None
        if sess is not None:
            sess.closed = True
            sess.exited_at = now
            sess.exited_stamp = stamp
            sess.exit_cx = cx
            sess.dwell_s = max(0.0, now - sess.entered_at)
            sess.track_ids.add(tid)
            sess.proof = self._save_proof(frame, w, h, stamp, "exit") or sess.proof
            sid = sess.session_id
            dwell = sess.dwell_s
            proof = sess.proof
        trk.entered = False
        trk.session_id = None
        ev = DoorEvent("exit", via, now, stamp, cx, cy, track_id=tid, session_id=sid, dwell_s=dwell, proof_path=proof)
        fired.append(ev)
        self.events.append(ev)
        self._mark("exit", now)
        self._persist(ev)
        if dwell is not None:
            print(f"[DoorGate] CUSTOMER EXIT  #{sid}  {stamp.strftime('%H:%M:%S')}  dwell={dwell:.0f}s  via={via}")
        else:
            print(f"[DoorGate] CUSTOMER EXIT  {stamp.strftime('%H:%M:%S')}  via={via}")

    def _relink(self, tid: int, cx: float, cy: float, now: float, stamp: datetime) -> _Session | None:
        """Fold tracker-ID flicker into a recent open session: same person
        re-appeared near where the previous track of the session was seen."""
        if not self.sessions:
            return None
        best = None
        for s in reversed(self.sessions):
            if s.closed:
                continue
            if now - s.entered_at > self.relink_s:
                continue
            # find the last known position from the session's tracks
            lcx, lcy, lseen = s.entered_cx, s.entered_cy, s.entered_at
            for tid_s, trk_s in self.tracks.items():
                if trk_s.session_id == s.session_id:
                    if trk_s.last_seen > lseen:
                        lseen, lcx, lcy = trk_s.last_seen, trk_s.last_cx, trk_s.last_cy
            dx = abs(cx - lcx)
            dy = abs(cy - lcy) * 2.0
            if dx + dy <= self.relink_dist and (now - lseen) < self.leave_s * 3.0:
                best = s
                break
        return best

    def _force_close_oldest(self, now: float, stamp: datetime, frame: Any, w: int, h: int,
                            fired: list[DoorEvent]) -> None:
        for s in (s for s in self.sessions if not s.closed):
            s.closed = True
            s.exited_at = now
            s.exited_stamp = stamp
            s.dwell_s = max(0.0, now - s.entered_at)
            print(f"[DoorGate] SESSION #{s.session_id} force-closed, dwell={s.dwell_s:.0f}s")
            break

    def _prune_tracks(self, now: float, stamp: datetime, fired: list[DoorEvent]) -> None:
        for tid, trk in list(self.tracks.items()):
            if now - trk.last_seen < self.leave_s:
                continue
            if trk.entered and trk.in_zone:
                sess = next((s for s in self.sessions if not s.closed and s.session_id == trk.session_id), None)
                if sess is not None:
                    sess.closed = True
                    sess.exited_at = now
                    sess.exited_stamp = stamp
                    sess.exit_cx = trk.last_cx
                    sess.dwell_s = max(0.0, now - sess.entered_at)
                    sess.proof = self._save_proof(None, 0, 0, stamp, "exit") or sess.proof
                    ev = DoorEvent(
                        "exit", "disappear", now, stamp, trk.last_cx, trk.last_cy,
                        track_id=tid, session_id=sess.session_id, dwell_s=sess.dwell_s,
                        proof_path=sess.proof,
                    )
                    fired.append(ev)
                    self.events.append(ev)
                    self._mark("exit", now)
                    self._persist(ev)
                    print(f"[DoorGate] CUSTOMER EXIT #{sess.session_id}  {stamp.strftime('%H:%M:%S')}  dwell={sess.dwell_s:.0f}s  via=disappear")
            del self.tracks[tid]

    # --------------------------------------------------------------- door motion
    def _update_door_motion(self, frame: Any, w: int, h: int) -> None:
        if frame is None:
            return
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            return
        x1 = max(0, int(self.zone[0] * w))
        x2 = min(w, int(self.zone[1] * w))
        strip = gray[:, x1:x2]
        scale = 8
        small = cv2.resize(strip, (max(1, strip.shape[1] // scale), max(1, strip.shape[0] // scale)))
        if self._prev_gray is not None:
            if self._prev_gray.shape != small.shape:
                self._prev_gray = small
            else:
                diff = cv2.absdiff(small, self._prev_gray)
                cur = float(diff.mean()) if diff.size else 0.0
                self.door_motion = MOTION_BLUR * cur + (1 - MOTION_BLUR) * self.door_motion
        self._prev_gray = small

    # ------------------------------------------------------------------ drawing
    def annotate(self, frame: Any, w: int, h: int) -> None:
        if frame is None:
            return
        lx = int(self.line_x * w)
        cv2.line(frame, (lx, 0), (lx, h), (0, 255, 90), 2)
        zx1 = int(self.zone[0] * w)
        ov = frame[:, zx1:].copy()
        cv2.addWeighted(ov, 0.14, frame[:, zx1:], 0.86, 0, frame[:, zx1:])
        cv2.rectangle(frame, (zx1, 0), (w, h), (0, 255, 90), 1)
        open_n = sum(1 for s in self.sessions if not s.closed)
        over = next((s for s in reversed(self.sessions) if not s.closed), None)
        if over is not None:
            text = f"Customer inside {max(0, float(datetime.now().timestamp()) - over.entered_at):.0f}s"
        else:
            text = "Lobby empty"
        cv2.putText(frame, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 90), 2, cv2.LINE_AA)
        if open_n > 1:
            cv2.putText(frame, f"{open_n} open", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------- state
    def recent(self) -> list[dict[str, Any]]:
        return [e.as_dict() for e in self.events]

    def sessions_meta(self) -> list[dict[str, Any]]:
        return [s.as_dict() for s in self.sessions]