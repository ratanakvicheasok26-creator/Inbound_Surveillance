"""Anonymous customer-visit monitor for massage (and similar) workplaces.

Does not use garage wrench-time or Face ID names. Appearance embeddings are
matched to opaque visitor ids; zone enter/leave uses occupancy hysteresis.

Visit identity rules (anti-churn):
- Sticky bind visitor_id to tracker track_id while the track lives.
- Never mint a new visitor_* on a missing/empty embedding (reid is throttled).
- Match gallery by best embedding in the bucket, not only the mean prototype.
- Persist a visit only after zone confirm hysteresis for that subject+zone.
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
from workplaces import VISIT_ZONE_KINDS, is_staff_id, parse_workplace_id, parse_zone_kind

VISIT_CONFIRM_SECONDS = 1.0
VISIT_CLEAR_SECONDS = 4.0
VISIT_GRACE_SECONDS = 8.0
# Cosine match; slightly soft so coat / angle drift still rematches same day.
REID_MATCH_THRESHOLD = 0.62
TRACK_STALE_SECONDS = 4.0
WAIT_SLA_SECONDS = 180.0
WAIT_SLA_COOLDOWN_SECONDS = 600.0
WAIT_SLA_ZONE_KINDS = frozenset({"waiting", "reception"})


def _iso(ts: float | datetime) -> str:
    if isinstance(ts, datetime):
        return ts.isoformat(timespec="seconds")
    return datetime.fromtimestamp(float(ts)).isoformat(timespec="seconds")


def _feat_hash(feat: np.ndarray) -> str:
    raw = np.asarray(feat, dtype=np.float32).tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def _normalize_feat(feat: np.ndarray | None) -> np.ndarray | None:
    vec = np.asarray(feat if feat is not None else [], dtype=np.float32).flatten()
    if vec.size == 0:
        return None
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return None
    return vec / norm


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


@dataclass
class _TrackBinding:
    subject_id: str
    last_seen: float
    cx: float = 0.5
    cy: float = 0.5
    zone_ids: list[str] = field(default_factory=list)
    last_feat: np.ndarray | None = None


@dataclass
class _RecentLostTrack:
    subject_id: str
    zone_ids: list[str]
    cx: float
    cy: float
    last_seen: float
    feat: np.ndarray | None = None


class AnonymousVisitorGallery:
    """Match appearance embeddings to opaque visitor ids. No names, no faces."""

    def __init__(self, threshold: float = REID_MATCH_THRESHOLD, max_per_id: int = 12) -> None:
        self.threshold = threshold
        self.max_per_id = max_per_id
        self.entries: dict[str, _GalleryEntry] = {}

    def match(
        self,
        feat: np.ndarray | None,
        threshold: float | None = None,
        allowed_ids: set[str] | None = None,
        excluded_ids: set[str] | None = None,
    ) -> str | None:
        """Return an existing subject_id if cosine match clears threshold, else None."""
        vec = _normalize_feat(feat)
        if vec is None:
            return None
        best_id: str | None = None
        best = threshold if threshold is not None else self.threshold
        candidates = list(self.entries.items())
        if allowed_ids is not None:
            candidates = [(sid, entry) for sid, entry in candidates if sid in allowed_ids]
        if excluded_ids is not None:
            candidates = [(sid, entry) for sid, entry in candidates if sid not in excluded_ids]
        for subject_id, entry in candidates:
            if not entry.embeddings:
                continue
            # Best-of-bucket: coat/angle views often diverge from the mean proto.
            for emb in entry.embeddings:
                score = float(np.dot(emb, vec))
                if score > best:
                    best = score
                    best_id = subject_id
            proto = np.mean(np.stack(entry.embeddings, axis=0), axis=0)
            proto = proto / max(float(np.linalg.norm(proto)), 1e-6)
            score = float(np.dot(proto, vec))
            if score > best:
                best = score
                best_id = subject_id
        return best_id

    def enroll(self, feat: np.ndarray | None, subject_id: str | None = None) -> str | None:
        """Enroll a normalized embedding with angle-diversity clustering."""
        vec = _normalize_feat(feat)
        if vec is None:
            return None
        sid = subject_id or f"visitor_{secrets.token_hex(4)}"
        entry = self.entries.get(sid)
        if entry is None:
            self.entries[sid] = _GalleryEntry(subject_id=sid, embeddings=[vec])
            return sid
        # Angle diversity check
        sims = [float(np.dot(e, vec)) for e in entry.embeddings]
        max_sim = max(sims) if sims else 0.0
        if max_sim > 0.95:
            idx = int(np.argmax(sims))
            updated = _normalize_feat(0.85 * entry.embeddings[idx] + 0.15 * vec)
            if updated is not None:
                entry.embeddings[idx] = updated
        else:
            entry.embeddings.append(vec)
            if len(entry.embeddings) > self.max_per_id:
                del entry.embeddings[0 : len(entry.embeddings) - self.max_per_id]
        return sid

    def match_or_enroll(
        self,
        feat: np.ndarray | None,
        threshold: float | None = None,
        allowed_ids: set[str] | None = None,
        excluded_ids: set[str] | None = None,
    ) -> str | None:
        """Match existing visitor or enroll. Returns None when feat is empty.

        Callers must not treat None as a new unique visitor — that was the
        source of 10k+ inflated unique counts when reid extraction is throttled.
        """
        matched = self.match(
            feat, threshold=threshold, allowed_ids=allowed_ids, excluded_ids=excluded_ids
        )
        if matched is not None:
            self.enroll(feat, matched)
            return matched
        return self.enroll(feat)

    def merge_entries(self, source_id: str, target_id: str) -> None:
        """Merge appearance embeddings of source_id into target_id."""
        if not source_id or not target_id or source_id == target_id:
            return
        src_entry = self.entries.pop(source_id, None)
        if src_entry is None or not src_entry.embeddings:
            return
        tgt_entry = self.entries.get(target_id)
        if tgt_entry is None:
            self.entries[target_id] = _GalleryEntry(
                subject_id=target_id,
                embeddings=src_entry.embeddings[: self.max_per_id],
            )
            return
        for emb in src_entry.embeddings:
            sims = [float(np.dot(e, emb)) for e in tgt_entry.embeddings]
            max_sim = max(sims) if sims else 0.0
            if max_sim < 0.95:
                tgt_entry.embeddings.append(emb)
        if len(tgt_entry.embeddings) > self.max_per_id:
            tgt_entry.embeddings = tgt_entry.embeddings[-self.max_per_id :]


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
        staff_memory: Any | None = None,
        workplace: object | None = None,
        store_customer_avatars: bool | None = None,
        branch_id: str = "",
        camera_role: str = "",
        telegram_out: Any | None = None,
        wait_sla_seconds: float = WAIT_SLA_SECONDS,
        wait_sla_cooldown_seconds: float = WAIT_SLA_COOLDOWN_SECONDS,
    ) -> None:
        self.confirm_seconds = confirm_seconds
        self.clear_seconds = clear_seconds
        self.grace_seconds = grace_seconds
        self.conn = conn
        self.staff_memory = staff_memory
        self.gallery = AnonymousVisitorGallery(threshold=match_threshold)
        self._gates: dict[str, OccupancyGate] = {}
        # Per (subject_id, zone_id) confirm before writing a visit row.
        self._visit_gates: dict[tuple[str, str], OccupancyGate] = {}
        self._open: dict[tuple[str, str], _OpenVisit] = {}
        self._track_bind: dict[int, _TrackBinding] = {}
        self._recent_lost_tracks: dict[int, _RecentLostTrack] = {}
        self._saved_avatars: set[str] = set()
        self._last_snapshots: list[VisitSnapshot] = []
        self.workplace = parse_workplace_id(workplace) if workplace is not None else "massage"
        # Privacy: massage never stores visitor JPEGs unless explicitly opted in.
        if store_customer_avatars is None:
            self.store_customer_avatars = self.workplace != "massage"
        else:
            self.store_customer_avatars = bool(store_customer_avatars)
        self.branch_id = str(branch_id or "").strip()
        self.camera_role = str(camera_role or "").strip()
        self.telegram_out = telegram_out
        self.wait_sla_seconds = float(wait_sla_seconds)
        self.wait_sla_cooldown_seconds = float(wait_sla_cooldown_seconds)
        self._sla_first_seen: dict[tuple[str, str], float] = {}
        self._sla_alerted_at: dict[tuple[str, str], float] = {}
        self.set_zones(zones)

    def merge_subjects(self, source_id: str, target_id: str) -> None:
        """Combine in-memory state, open visits, and gallery embeddings for two profiles."""
        if not source_id or not target_id or source_id == target_id:
            return
        self.gallery.merge_entries(source_id, target_id)

        for (sid, zid), visit in list(self._open.items()):
            if sid == source_id:
                del self._open[(sid, zid)]
                tgt_key = (target_id, zid)
                if tgt_key in self._open:
                    existing = self._open[tgt_key]
                    existing.started_at = min(existing.started_at, visit.started_at)
                    existing.last_seen = max(existing.last_seen, visit.last_seen)
                else:
                    visit.subject_id = target_id
                    self._open[tgt_key] = visit

        for (sid, zid), gate in list(self._visit_gates.items()):
            if sid == source_id:
                del self._visit_gates[(sid, zid)]
                tgt_key = (target_id, zid)
                if tgt_key not in self._visit_gates:
                    self._visit_gates[tgt_key] = gate

        for binding in self._track_bind.values():
            if binding.subject_id == source_id:
                binding.subject_id = target_id

        for lost in self._recent_lost_tracks.values():
            if lost.subject_id == source_id:
                lost.subject_id = target_id

        if source_id in self._saved_avatars:
            self._saved_avatars.add(target_id)
            self._saved_avatars.discard(source_id)

        if self.workplace == "massage" or not self.store_customer_avatars:
            return

        try:
            import shutil
            from paths import data_dir

            av_dir = data_dir() / "avatars" / "visitors"
            src_file = av_dir / f"{source_id}.jpg"
            tgt_file = av_dir / f"{target_id}.jpg"
            if src_file.is_file() and not tgt_file.is_file():
                shutil.copy2(src_file, tgt_file)
        except Exception as exc:
            print(f"[CustomerVisitMonitor.merge_subjects] avatar copy error: {exc}", flush=True)

    def reset(self) -> None:
        """Wipe all in-memory visitors, open sessions, and tracking bindings."""
        self.gallery = AnonymousVisitorGallery(threshold=self.gallery.threshold)
        self._open.clear()
        self._track_bind.clear()
        self._recent_lost_tracks.clear()
        self._saved_avatars.clear()
        self._last_snapshots.clear()
        self._sla_first_seen.clear()
        self._sla_alerted_at.clear()
        for gate in self._gates.values():
            gate.occupied = False
        self._visit_gates.clear()

    def configure_site(
        self,
        *,
        branch_id: str | None = None,
        camera_role: str | None = None,
        store_customer_avatars: bool | None = None,
        telegram_out: Any | None = None,
        workplace: object | None = None,
    ) -> None:
        """Update site identity / privacy / alert wiring without rebuilding the monitor."""
        if workplace is not None:
            self.workplace = parse_workplace_id(workplace)
            if store_customer_avatars is None and self.workplace == "massage":
                self.store_customer_avatars = False
        if branch_id is not None:
            self.branch_id = str(branch_id or "").strip()
        if camera_role is not None:
            self.camera_role = str(camera_role or "").strip()
        if store_customer_avatars is not None:
            self.store_customer_avatars = bool(store_customer_avatars)
        if telegram_out is not None:
            self.telegram_out = telegram_out
    def set_zones(self, zones: object | None, workplace: object | None = None) -> None:
        from workplaces import normalize_workplace_zones

        if workplace is not None:
            self.workplace = parse_workplace_id(workplace)
        seed = self.workplace == "massage"
        self.zones = normalize_workplace_zones(self.workplace, zones, seed_if_empty=seed)
        keep = {str(z["id"]) for z in self.zones}
        self._gates = {
            zid: self._gates.get(zid) or OccupancyGate(self.confirm_seconds, self.clear_seconds)
            for zid in keep
        }
        self._open = {key: visit for key, visit in self._open.items() if key[1] in keep}
        self._visit_gates = {
            key: gate for key, gate in self._visit_gates.items() if key[1] in keep
        }

    def configs(self) -> list[dict[str, Any]]:
        return list(self.zones)

    def _try_stitch_lost_track(
        self,
        cx: float,
        cy: float,
        zone_ids: list[str],
        feat: np.ndarray | None,
        now: float,
        max_dt: float = 3.5,
        max_dist: float = 0.25,
        excluded_ids: set[str] | None = None,
    ) -> str | None:
        """Reconnect recently lost tracks (e.g. angle turn dropped track) before minting new visitor."""
        best_candidate: str | None = None
        best_dist = max_dist

        for tid, lost in list(self._recent_lost_tracks.items()):
            if excluded_ids and lost.subject_id in excluded_ids:
                continue
            dt = now - lost.last_seen
            if dt > max_dt:
                continue
            dist = float(((cx - lost.cx) ** 2 + (cy - lost.cy) ** 2) ** 0.5)
            same_zone = any(z in lost.zone_ids for z in zone_ids) if (zone_ids and lost.zone_ids) else True
            if same_zone and dist < best_dist:
                if feat is not None and lost.feat is not None:
                    vec = _normalize_feat(feat)
                    lost_vec = _normalize_feat(lost.feat)
                    if vec is not None and lost_vec is not None:
                        sim = float(np.dot(vec, lost_vec))
                        if sim < 0.40:
                            continue
                best_dist = dist
                best_candidate = lost.subject_id

        return best_candidate

    def _resolve_subject(
        self,
        track_id: int,
        feat: np.ndarray | None,
        now: float,
        cx: float = 0.5,
        cy: float = 0.5,
        zone_ids: list[str] | None = None,
        face_customer_id: str | None = None,
        claimed_ids: set[str] | None = None,
    ) -> str | None:
        """Sticky track binding + face biometric anchoring + multi-view gallery with frame-exclusivity."""
        claimed = claimed_ids if claimed_ids is not None else set()

        # 0. Direct Face-anchored customer identification (from any camera: Parking, Entrance, Reception)
        if face_customer_id:
            if track_id > 0:
                self._track_bind[track_id] = _TrackBinding(
                    subject_id=face_customer_id,
                    last_seen=now,
                    cx=cx,
                    cy=cy,
                    zone_ids=list(zone_ids or []),
                    last_feat=feat,
                )
            if feat is not None:
                self.gallery.enroll(feat, face_customer_id)
            return face_customer_id

        binding = self._track_bind.get(track_id) if track_id > 0 else None
        if binding is not None and (now - binding.last_seen) <= TRACK_STALE_SECONDS:
            if binding.subject_id not in claimed:
                binding.last_seen = now
                binding.cx = cx
                binding.cy = cy
                if zone_ids:
                    binding.zone_ids = list(zone_ids)
                if feat is not None:
                    binding.last_feat = feat
                    self.gallery.enroll(feat, binding.subject_id)
                return binding.subject_id

        # 1. Spatial-temporal track stitching (person turned angle, brief occlusion, Kalman track ID hopped)
        stitched_id = self._try_stitch_lost_track(
            cx, cy, zone_ids or [], feat, now, excluded_ids=claimed
        )
        if stitched_id is not None:
            if track_id > 0:
                self._track_bind[track_id] = _TrackBinding(
                    subject_id=stitched_id,
                    last_seen=now,
                    cx=cx,
                    cy=cy,
                    zone_ids=list(zone_ids or []),
                    last_feat=feat,
                )
            if feat is not None:
                self.gallery.enroll(feat, stitched_id)
            return stitched_id

        # 2. Match active/open visitors with calibrated threshold, respecting frame exclusivity
        active_ids = {visit.subject_id for visit in self._open.values()} - claimed
        matched = None
        if feat is not None:
            thresh = self.gallery.threshold
            if active_ids:
                matched = self.gallery.match(
                    feat, threshold=thresh, allowed_ids=active_ids, excluded_ids=claimed
                )
            if matched is None:
                matched = self.gallery.match_or_enroll(
                    feat, threshold=thresh, excluded_ids=claimed
                )
        else:
            return None

        if matched is None:
            return None

        # Absolute safeguard: if matched ID is already claimed in this frame, mint fresh unique ID
        if matched in claimed:
            matched = self.gallery.enroll(feat)
            if matched is None:
                return None

        if track_id > 0:
            self._track_bind[track_id] = _TrackBinding(
                subject_id=matched,
                last_seen=now,
                cx=cx,
                cy=cy,
                zone_ids=list(zone_ids or []),
                last_feat=feat,
            )
        return matched


    def _prune_track_bindings(self, now: float, live_tracks: set[int]) -> None:
        stale = [
            tid
            for tid, bind in self._track_bind.items()
            if tid not in live_tracks and (now - bind.last_seen) > TRACK_STALE_SECONDS
        ]
        for tid in stale:
            bind = self._track_bind.pop(tid, None)
            if bind is not None:
                self._recent_lost_tracks[tid] = _RecentLostTrack(
                    subject_id=bind.subject_id,
                    zone_ids=list(getattr(bind, "zone_ids", [])),
                    cx=getattr(bind, "cx", 0.5),
                    cy=getattr(bind, "cy", 0.5),
                    last_seen=bind.last_seen,
                    feat=getattr(bind, "last_feat", None),
                )
        # Purge stale lost tracks older than 6 seconds
        self._recent_lost_tracks = {
            tid: lost
            for tid, lost in self._recent_lost_tracks.items()
            if (now - lost.last_seen) <= 6.0
        }

    def update(
        self,
        detections: list[Any],
        frame_w: int,
        frame_h: int,
        now: float | None = None,
        embeddings: dict[int, np.ndarray] | None = None,
        frame: Any | None = None,
        staff_memory: Any | None = None,
    ) -> list[VisitSnapshot]:
        now = time.time() if now is None else float(now)
        memory = staff_memory if staff_memory is not None else self.staff_memory
        if memory is not None:
            memory.annotate_known(detections or [], embeddings)
        occupants_in_zone: dict[str, list[str]] = {str(z["id"]): [] for z in self.zones}
        visitors_in_zone: dict[str, list[str]] = {str(z["id"]): [] for z in self.zones}
        present_subject_zones: set[tuple[str, str]] = set()
        present_sla_keys: set[tuple[str, str]] = set()
        live_tracks: set[int] = set()
        claimed_ids: set[str] = set()

        for det in detections or []:
            # 1. Skip inanimate clutter or dead tracks
            if getattr(det, "clutter", False):
                try:
                    det.identity = None
                except Exception:
                    pass
                continue

            # 2. Skip unconfirmed detections (e.g. 1st frame flash of background or low confidence)
            if hasattr(det, "confirmed") and not det.confirmed and getattr(det, "hits", 1) < 2:
                try:
                    det.identity = None
                except Exception:
                    pass
                continue

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
            if track_id > 0:
                live_tracks.add(track_id)
            feat = None
            if embeddings and track_id in embeddings:
                feat = embeddings[track_id]
            else:
                feat = getattr(det, "reid_feat", None)

            already_staff = bool(getattr(det, "is_staff", False)) or is_staff_id(
                getattr(det, "identity", None)
            )
            if already_staff:
                subject_id = str(getattr(det, "identity", None) or "")
                claimed_ids.add(subject_id)
                try:
                    det.is_staff = True
                except Exception:
                    pass
                for zone in self.zones:
                    zid = str(zone["id"])
                    if point_in_roi(cx, cy, zone["roi"]):
                        occupants_in_zone[zid].append(subject_id)
                continue

            det_zones = [
                str(z["id"]) for z in self.zones if point_in_roi(cx, cy, z["roi"])
            ]
            face_customer_id = getattr(det, "customer_id", None)
            subject_id = self._resolve_subject(
                track_id, feat, now, cx=cx, cy=cy, zone_ids=det_zones,
                face_customer_id=face_customer_id, claimed_ids=claimed_ids,
            )

            if subject_id is None:
                # No embedding yet and no sticky track — wait. Do not invent a visitor.
                try:
                    det.identity = None
                    det.is_staff = False
                except Exception:
                    pass
                for zone in self.zones:
                    zid = str(zone["id"])
                    if point_in_roi(cx, cy, zone["roi"]):
                        occupants_in_zone[zid].append("pending")
                continue

            claimed_ids.add(subject_id)
            try:
                det.identity = subject_id
                det.is_staff = False
            except Exception:
                pass

            if frame is not None:
                self._maybe_save_avatar(subject_id, det, frame, now)

            in_reception = False
            visit_zones: list[str] = []
            sla_keys: list[tuple[str, str]] = []
            for zone in self.zones:
                if not point_in_roi(cx, cy, zone["roi"]):
                    continue
                zid = str(zone["id"])
                kind = parse_zone_kind(zone.get("type"), "massage")
                occupants_in_zone[zid].append(subject_id)
                if kind == "reception":
                    in_reception = True
                elif kind in VISIT_ZONE_KINDS:
                    visit_zones.append(zid)
                if kind in WAIT_SLA_ZONE_KINDS:
                    sla_keys.append((subject_id, zid))

            enrolled_staff = False
            if memory is not None:
                enrolled_staff = bool(
                    memory.observe(subject_id, feat, det, frame, in_reception, now)
                )
            if enrolled_staff or getattr(det, "is_staff", False) or is_staff_id(
                getattr(det, "identity", None)
            ):
                continue
            for zid in visit_zones:
                visitors_in_zone[zid].append(subject_id)
                present_subject_zones.add((subject_id, zid))
            for key in sla_keys:
                present_sla_keys.add(key)

        self._prune_track_bindings(now, live_tracks)

        # Confirm per subject+zone before writing a visit row (massage visit law).
        confirmed_keys: set[tuple[str, str]] = set()
        active_gate_keys = set(present_subject_zones) | set(self._visit_gates.keys())
        for key in list(active_gate_keys):
            gate = self._visit_gates.get(key)
            if gate is None:
                gate = OccupancyGate(self.confirm_seconds, self.clear_seconds)
                self._visit_gates[key] = gate
            present = key in present_subject_zones
            if gate.update(present, now):
                confirmed_keys.add(key)
                self._touch_visit(key[0], key[1], now)
            elif not present and not gate.occupied:
                self._visit_gates.pop(key, None)

        snapshots: list[VisitSnapshot] = []
        present_keys = set(confirmed_keys) | {
            key for key in self._open if key in present_subject_zones
        }
        for zone in self.zones:
            zid = str(zone["id"])
            occupant_ids = list(dict.fromkeys(occupants_in_zone.get(zid) or []))
            ids = list(dict.fromkeys(visitors_in_zone.get(zid) or []))
            occupied = self._gates[zid].update(bool(occupant_ids), now)
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
        self._check_wait_sla(present_sla_keys, now)
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

    def _maybe_save_avatar(self, subject_id: str, det: Any, frame: Any, now: float) -> None:
        # Privacy hard-gate: massage / explicit opt-out never writes visitor JPEGs.
        if self.workplace == "massage" or not self.store_customer_avatars:
            return
        if frame is None or getattr(frame, "size", 0) == 0:
            return
        if subject_id in self._saved_avatars:
            return
        if hasattr(det, "x1"):
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        else:
            bbox = getattr(det, "bbox", None) or getattr(det, "xyxy", None)
            if bbox is None or len(bbox) < 4:
                return
            x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        h_f, w_f = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_f, x2), min(h_f, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 20 or bh < 30:
            return
        crop = frame[y1 : min(h_f, y1 + int(bh * 0.55)), x1:x2]
        if crop.size == 0:
            return
        try:
            import cv2
            thumb = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
            from paths import data_dir
            av_dir = data_dir() / "avatars" / "visitors"
            av_dir.mkdir(parents=True, exist_ok=True)
            av_file = av_dir / f"{subject_id}.jpg"
            cv2.imwrite(str(av_file), thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            self._saved_avatars.add(subject_id)
            if self.conn is not None:
                from db import save_subject_avatar
                save_subject_avatar(self.conn, subject_id, f"/api/workplace/avatar/{subject_id}")
        except Exception:
            pass

    def _check_wait_sla(self, present_keys: set[tuple[str, str]], now: float) -> None:
        """Alert once when a guest dwells in waiting/reception for >= wait_sla_seconds."""
        for key in list(self._sla_first_seen.keys()):
            if key not in present_keys:
                self._sla_first_seen.pop(key, None)

        zone_meta = {
            str(z["id"]): {
                "kind": parse_zone_kind(z.get("type"), self.workplace),
                "name": str(z.get("name") or z.get("id") or ""),
            }
            for z in self.zones
        }

        for key in present_keys:
            first = self._sla_first_seen.get(key)
            if first is None:
                self._sla_first_seen[key] = now
                continue
            dwell = now - first
            if dwell < self.wait_sla_seconds:
                continue
            last_alert = self._sla_alerted_at.get(key)
            if last_alert is not None and (now - last_alert) < self.wait_sla_cooldown_seconds:
                continue
            self._sla_alerted_at[key] = now
            subject_id, zone_id = key
            meta = zone_meta.get(zone_id) or {}
            branch = self.branch_id or "branch"
            role = self.camera_role or "front_desk"
            zone_name = meta.get("name") or zone_id
            msg = (
                f"[{branch}] WAIT BOTTLENECK: Guest waiting >= 3 mins at Front Desk "
                f"({zone_name}, role={role}). Action: Greet / assist."
            )
            bot = self.telegram_out
            if bot is not None and getattr(bot, "enabled", False):
                try:
                    bot.send_message(msg)
                except Exception as exc:
                    print(f"[wait_sla] telegram failed: {exc}", flush=True)
            else:
                print(f"[wait_sla] {msg}", flush=True)
            if self.conn is not None:
                try:
                    from db import insert_event

                    insert_event(
                        self.conn,
                        "wait_bottleneck",
                        datetime.fromtimestamp(now),
                        abs_path=f"{subject_id}@{zone_id}",
                        branch_id=self.branch_id or None,
                        camera_role=self.camera_role or None,
                    )
                except Exception as exc:
                    print(f"[wait_sla] db event failed: {exc}", flush=True)

    def open_sessions(self) -> list[dict[str, Any]]:
        names = {str(z["id"]): str(z["name"]) for z in self.zones}
        now = time.time()
        sessions: list[dict[str, Any]] = []
        for visit in self._open.values():
            row: dict[str, Any] = {
                "subject_id": visit.subject_id,
                "zone_id": visit.zone_id,
                "zone_name": names.get(visit.zone_id, visit.zone_id),
                "started_at": _iso(visit.started_at),
                "dwell_seconds": max(0.0, round(now - visit.started_at, 1)),
            }
            if self.store_customer_avatars and self.workplace != "massage":
                row["avatar_url"] = f"/api/workplace/avatar/{visit.subject_id}"
            sessions.append(row)
        return sessions
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
