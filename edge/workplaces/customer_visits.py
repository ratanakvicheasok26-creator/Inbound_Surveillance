"""Anonymous customer-visit monitor for massage (and similar) workplaces.

Does not use garage wrench-time or Face ID names. Appearance embeddings are
matched to opaque visitor ids; zone enter/leave uses occupancy hysteresis.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from occupancy import OccupancyGate, point_in_roi
from workplaces import parse_zone_kind

VISIT_CONFIRM_SECONDS = 1.0
VISIT_CLEAR_SECONDS = 4.0
VISIT_GRACE_SECONDS = 8.0
REID_MATCH_THRESHOLD = 0.72


def _iso(ts: float | datetime) -> str:
    if isinstance(ts, datetime):
        return ts.isoformat(timespec="seconds")
    return datetime.fromtimestamp(float(ts)).isoformat(timespec="seconds")


def _feat_hash(feat: np.ndarray) -> str:
    raw = np.asarray(feat, dtype=np.float32).tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass
class VisitSnapshot:
    zone_id: str
    name: str
    zone_kind: str
    occupied: bool
    visitor_ids: list[str]
    open_visit_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.zone_id,
            "bay_id": self.zone_id,
            "name": self.name,
            "type": self.zone_kind,
            "zone_kind": self.zone_kind,
            "state": "OCCUPIED" if self.occupied else "EMPTY",
            "visitor_ids": list(self.visitor_ids),
            "open_visit_count": self.open_visit_count,
            "person_present": self.occupied,
        }


@dataclass
class _OpenVisit:
    visit_id: str
    subject_id: str
    zone_id: str
    started_at: float
    last_seen: float


@dataclass
class _GalleryEntry:
    subject_id: str
    embeddings: list[np.ndarray] = field(default_factory=list)


class AnonymousVisitorGallery:
    """Match appearance embeddings to opaque visitor ids. No names, no faces."""

    def __init__(self, threshold: float = REID_MATCH_THRESHOLD, max_per_id: int = 12) -> None:
        self.threshold = threshold
        self.max_per_id = max_per_id
        self.entries: dict[str, _GalleryEntry] = {}

    def match_or_enroll(self, feat: np.ndarray | None) -> str:
        vec = np.asarray(feat if feat is not None else [], dtype=np.float32).flatten()
        if vec.size == 0 or float(np.linalg.norm(vec)) <= 1e-6:
            return f"visitor_{secrets.token_hex(4)}"
        vec = vec / max(float(np.linalg.norm(vec)), 1e-6)
        best_id: str | None = None
        best = self.threshold
        for subject_id, entry in self.entries.items():
            if not entry.embeddings:
                continue
            proto = np.mean(np.stack(entry.embeddings, axis=0), axis=0)
            proto = proto / max(float(np.linalg.norm(proto)), 1e-6)
            score = float(np.dot(proto, vec))
            if score > best:
                best = score
                best_id = subject_id
        if best_id is None:
            best_id = f"visitor_{secrets.token_hex(4)}"
            self.entries[best_id] = _GalleryEntry(subject_id=best_id, embeddings=[vec])
            return best_id
        bucket = self.entries[best_id].embeddings
        bucket.append(vec)
        if len(bucket) > self.max_per_id:
            del bucket[0 : len(bucket) - self.max_per_id]
        return best_id


class CustomerVisitMonitor:
    def __init__(
        self,
        zones: object | None = None,
        *,
        confirm_seconds: float = VISIT_CONFIRM_SECONDS,
        clear_seconds: float = VISIT_CLEAR_SECONDS,
        grace_seconds: float = VISIT_GRACE_SECONDS,
        match_threshold: float = REID_MATCH_THRESHOLD,
        conn: Any | None = None,
    ) -> None:
        self.confirm_seconds = confirm_seconds
        self.clear_seconds = clear_seconds
        self.grace_seconds = grace_seconds
        self.conn = conn
        self.gallery = AnonymousVisitorGallery(threshold=match_threshold)
        self._gates: dict[str, OccupancyGate] = {}
        self._open: dict[tuple[str, str], _OpenVisit] = {}
        self._last_snapshots: list[VisitSnapshot] = []
        self.set_zones(zones)

    def set_zones(self, zones: object | None) -> None:
        from workplaces import normalize_workplace_zones

        self.zones = normalize_workplace_zones("massage", zones, seed_if_empty=True)
        keep = {str(z["id"]) for z in self.zones}
        self._gates = {
            zid: self._gates.get(zid) or OccupancyGate(self.confirm_seconds, self.clear_seconds)
            for zid in keep
        }
        self._open = {key: visit for key, visit in self._open.items() if key[1] in keep}

    def configs(self) -> list[dict[str, Any]]:
        return list(self.zones)

    def update(
        self,
        detections: list[Any],
        frame_w: int,
        frame_h: int,
        now: float | None = None,
        embeddings: dict[int, np.ndarray] | None = None,
    ) -> list[VisitSnapshot]:
        now = time.time() if now is None else float(now)
        visitors_in_zone: dict[str, list[str]] = {str(z["id"]): [] for z in self.zones}
        for det in detections or []:
            if hasattr(det, "x1"):
                x1, y1, x2, y2 = float(det.x1), float(det.y1), float(det.x2), float(det.y2)
            else:
                bbox = getattr(det, "bbox", None) or getattr(det, "xyxy", None)
                if bbox is None or len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = (float(v) for v in bbox[:4])
            cx = ((x1 + x2) / 2.0) / max(frame_w, 1)
            cy = ((y1 + y2) / 2.0) / max(frame_h, 1)
            track_id = int(getattr(det, "track_id", 0) or 0)
            feat = None
            if embeddings and track_id in embeddings:
                feat = embeddings[track_id]
            else:
                feat = getattr(det, "reid_feat", None)
            subject_id = self.gallery.match_or_enroll(feat)
            try:
                det.identity = subject_id
                det.is_staff = False
            except Exception:
                pass
            for zone in self.zones:
                if point_in_roi(cx, cy, zone["roi"]):
                    visitors_in_zone[str(zone["id"])].append(subject_id)
                    self._touch_visit(subject_id, str(zone["id"]), now)

        snapshots: list[VisitSnapshot] = []
        present_keys = set()
        for zone in self.zones:
            zid = str(zone["id"])
            ids = list(dict.fromkeys(visitors_in_zone.get(zid) or []))
            occupied = self._gates[zid].update(bool(ids), now)
            for sid in ids:
                present_keys.add((sid, zid))
            snapshots.append(
                VisitSnapshot(
                    zone_id=zid,
                    name=str(zone["name"]),
                    zone_kind=parse_zone_kind(zone.get("type"), "massage"),
                    occupied=occupied,
                    visitor_ids=ids,
                    open_visit_count=sum(1 for key in self._open if key[1] == zid),
                )
            )
        self._close_stale(now, present_keys)
        self._last_snapshots = snapshots
        return snapshots

    def telemetry(self) -> list[dict[str, Any]]:
        if self._last_snapshots:
            return [s.as_dict() for s in self._last_snapshots]
        return [
            VisitSnapshot(
                zone_id=str(z["id"]),
                name=str(z["name"]),
                zone_kind=parse_zone_kind(z.get("type"), "massage"),
                occupied=False,
                visitor_ids=[],
                open_visit_count=0,
            ).as_dict()
            for z in self.zones
        ]

    def open_sessions(self) -> list[dict[str, Any]]:
        names = {str(z["id"]): str(z["name"]) for z in self.zones}
        return [
            {
                "subject_id": visit.subject_id,
                "zone_id": visit.zone_id,
                "zone_name": names.get(visit.zone_id, visit.zone_id),
                "started_at": _iso(visit.started_at),
            }
            for visit in self._open.values()
        ]

    def _touch_visit(self, subject_id: str, zone_id: str, now: float) -> None:
        key = (subject_id, zone_id)
        current = self._open.get(key)
        if current is not None:
            current.last_seen = now
            return
        visit_id = secrets.token_hex(8)
        self._open[key] = _OpenVisit(
            visit_id=visit_id,
            subject_id=subject_id,
            zone_id=zone_id,
            started_at=now,
            last_seen=now,
        )
        if self.conn is not None:
            from db import start_customer_visit, upsert_anonymous_subject

            upsert_anonymous_subject(self.conn, subject_id, now)
            start_customer_visit(self.conn, visit_id, subject_id, zone_id, now)

    def _close_stale(self, now: float, present: set[tuple[str, str]]) -> None:
        stale = [
            key
            for key, visit in self._open.items()
            if key not in present and (now - visit.last_seen) >= self.grace_seconds
        ]
        for key in stale:
            visit = self._open.pop(key)
            if self.conn is not None:
                from db import end_customer_visit

                end_customer_visit(self.conn, visit.visit_id, now)


def visit_counts(conn: Any, now: datetime | None = None) -> dict[str, Any]:
    from db import customer_visit_counts

    return customer_visit_counts(conn, now)
