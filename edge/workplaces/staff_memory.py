"""Remember staff from AI-verified reception-counter presence.

Zones stay where, not who. A person who dwells in the reception ROI is gated
like the garage AI auditor, then a VLM decides staff-behind-counter vs
customer-at-desk. On a staff verdict the appearance embedding is stored under
an opaque ``staff_<hex>`` id (no name, no photo). Later frames rematch by
cosine. Missing ``FIREWORKS_API_KEY`` uses simulated-audit so tests can mock
verdicts without a network call.
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ai_auditor import TokenSaverGate, _load_env_or_config, extract_dual_crops
from occupancy import point_in_roi
from workplaces import STAFF_ID_PREFIX, parse_workplace_id, parse_zone_kind

logger = logging.getLogger("staff_memory")

STAFF_MATCH_THRESHOLD = 0.72
STAFF_COUNTER_DWELL_SECONDS = 3.0
STAFF_COUNTER_COOLDOWN_SECONDS = 45.0
STAFF_COUNTER_GRACE_SECONDS = 2.0


def _normalize_feat(feat: np.ndarray | None) -> np.ndarray | None:
    vec = np.asarray(feat if feat is not None else [], dtype=np.float32).flatten()
    if vec.size == 0 or float(np.linalg.norm(vec)) <= 1e-6:
        return None
    return vec / max(float(np.linalg.norm(vec)), 1e-6)


def _feat_from_det(det: Any, embeddings: dict[int, np.ndarray] | None) -> np.ndarray | None:
    track_id = int(getattr(det, "track_id", 0) or 0)
    if embeddings and track_id in embeddings:
        return embeddings[track_id]
    return getattr(det, "reid_feat", None)


def _bbox(det: Any) -> tuple[int, int, int, int] | None:
    if hasattr(det, "x1"):
        return (int(det.x1), int(det.y1), int(det.x2), int(det.y2))
    bbox = getattr(det, "bbox", None) or getattr(det, "xyxy", None)
    if bbox is None or len(bbox) < 4:
        return None
    return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))


@dataclass
class StaffVerdict:
    subject_id: str
    role: str
    confidence: float
    explanation: str
    is_staff: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "role": self.role,
            "confidence": round(self.confidence, 2),
            "explanation": self.explanation,
            "is_staff": self.is_staff,
        }


@dataclass
class _StaffEntry:
    staff_id: str
    embeddings: list[np.ndarray] = field(default_factory=list)


class StaffVLMClient:
    """Fireworks VLM client for counter staff vs customer. Copied from ai_auditor."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 25.0,
    ) -> None:
        loaded_key, loaded_model = _load_env_or_config()
        self.api_key = api_key if api_key is not None else loaded_key
        self.model = model or loaded_model
        self.timeout = timeout
        self.endpoint = "https://api.fireworks.ai/inference/v1/chat/completions"
        if self.api_key:
            print(f"[Staff-Memory] Ready with Fireworks VLM (model: {self.model})", flush=True)
        else:
            print(
                "[Staff-Memory] NOTICE: No FIREWORKS_API_KEY found. "
                "Running in Simulated Audit mode for testing.",
                flush=True,
            )

    def classify_counter(
        self,
        b64_person: str,
        b64_context: str,
        subject_id: str,
    ) -> StaffVerdict:
        if not self.api_key:
            return self._mock_verdict(subject_id)

        prompt = f"""You are verifying who is at a massage-shop reception desk.

Image 1: close-up of the person.
Image 2: wider desk / counter context.

STAFF: working behind the counter, seated at a receptionist station, using a
POS / computer from the staff side, handling check-in from behind the desk.
CUSTOMER: standing in front of the desk, waiting to check in, on the public
side of the counter.

Subject id (opaque, not a name): {subject_id}

Respond STRICTLY in JSON with keys:
{{
  "role": "staff" | "customer",
  "confidence": float (0.0 to 1.0),
  "explanation": "1-2 sentences citing visual evidence"
}}
"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_person}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_context}"},
                        },
                    ],
                }
            ],
            "max_tokens": 512,
            "temperature": 0.1,
        }
        if "deepseek" in str(self.model).lower():
            payload["thinking"] = {"type": "disabled"}
        try:
            import requests

            r = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
            r.raise_for_status()
            res_json = r.json()
            msg = res_json["choices"][0]["message"]
            raw = (msg.get("content") or "").strip()
            if not raw:
                raw = (msg.get("reasoning_content") or "").strip()
            if "```json" in raw:
                raw = raw.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in raw:
                raw = raw.split("```", 1)[1].split("```", 1)[0]
            start = raw.find("{")
            end = raw.rfind("}")
            raw_json = raw[start : end + 1] if start != -1 and end != -1 else raw.strip()
            data = json.loads(raw_json)
            role = str(data.get("role") or "customer").strip().lower()
            if role not in ("staff", "customer"):
                role = "customer"
            conf = float(data.get("confidence", 0.0))
            expl = str(data.get("explanation") or "")
            return StaffVerdict(
                subject_id=subject_id,
                role=role,
                confidence=conf,
                explanation=expl,
                is_staff=role == "staff",
            )
        except Exception as exc:
            logger.warning("Staff counter VLM failed: %s", exc)
            return StaffVerdict(
                subject_id=subject_id,
                role="customer",
                confidence=0.0,
                explanation=f"Audit error: {exc}",
                is_staff=False,
            )

    def _mock_verdict(self, subject_id: str) -> StaffVerdict:
        """Simulated customer verdict when no API key is provided."""
        return StaffVerdict(
            subject_id=subject_id,
            role="customer",
            confidence=0.94,
            explanation="[Simulated AI Audit] Person is standing at the public side of the desk.",
            is_staff=False,
        )


class StaffMemory:
    """Appearance gallery of AI-verified staff. Opaque ids only."""

    def __init__(
        self,
        zones: object | None = None,
        *,
        threshold: float = STAFF_MATCH_THRESHOLD,
        dwell_seconds: float = STAFF_COUNTER_DWELL_SECONDS,
        cooldown_seconds: float = STAFF_COUNTER_COOLDOWN_SECONDS,
        grace_seconds: float = STAFF_COUNTER_GRACE_SECONDS,
        max_per_id: int = 12,
        conn: Any | None = None,
        vlm_client: StaffVLMClient | None = None,
        gate: TokenSaverGate | None = None,
        inline: bool | None = None,
        workplace: object | None = None,
    ) -> None:
        self.threshold = threshold
        self.max_per_id = max_per_id
        self.conn = conn
        self.vlm = vlm_client or StaffVLMClient()
        self.gate = gate or TokenSaverGate(
            duration_threshold=dwell_seconds,
            cooldown_seconds=cooldown_seconds,
            grace_seconds=grace_seconds,
        )
        self._inline = bool(self.vlm.api_key == "") if inline is None else bool(inline)
        self.entries: dict[str, _StaffEntry] = {}
        self.zones: list[dict[str, Any]] = []
        self.workplace = parse_workplace_id(workplace) if workplace is not None else "massage"
        self._executor: ThreadPoolExecutor | None = None
        if not self._inline:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="staff_memory")
        self.set_zones(zones)
        self.load()

    def set_conn(self, conn: Any | None) -> None:
        self.conn = conn
        self.load()

    def set_zones(self, zones: object | None, workplace: object | None = None) -> None:
        from workplaces import normalize_workplace_zones

        if workplace is not None:
            self.workplace = parse_workplace_id(workplace)
        if zones is None:
            self.zones = []
            return
        self.zones = normalize_workplace_zones(self.workplace, zones, seed_if_empty=False)

    def known_ids(self) -> set[str]:
        return set(self.entries)

    def load(self) -> None:
        if self.conn is None:
            return
        from db import list_staff_memory

        for row in list_staff_memory(self.conn):
            staff_id = str(row.get("id") or "")
            vec = row.get("embedding")
            if not staff_id or vec is None:
                continue
            arr = _normalize_feat(np.asarray(vec, dtype=np.float32))
            if arr is None:
                continue
            entry = self.entries.get(staff_id) or _StaffEntry(staff_id=staff_id)
            if not entry.embeddings:
                entry.embeddings.append(arr)
            self.entries[staff_id] = entry

    def match(self, feat: np.ndarray | None) -> str | None:
        vec = _normalize_feat(feat)
        if vec is None:
            return None
        best_id: str | None = None
        best = self.threshold
        for staff_id, entry in self.entries.items():
            if not entry.embeddings:
                continue
            proto = np.mean(np.stack(entry.embeddings, axis=0), axis=0)
            proto = proto / max(float(np.linalg.norm(proto)), 1e-6)
            score = float(np.dot(proto, vec))
            if score > best:
                best = score
                best_id = staff_id
        return best_id

    def remember(self, feat: np.ndarray | None, now: float | None = None) -> str | None:
        vec = _normalize_feat(feat)
        if vec is None:
            return None
        existing = self.match(vec)
        if existing:
            self._add_embedding(existing, vec, now)
            return existing
        staff_id = f"{STAFF_ID_PREFIX}{secrets.token_hex(4)}"
        self.entries[staff_id] = _StaffEntry(staff_id=staff_id, embeddings=[vec])
        self._persist(staff_id, vec, now)
        return staff_id

    def in_reception(self, cx: float, cy: float) -> bool:
        for zone in self.zones:
            if parse_zone_kind(zone.get("type"), "massage") != "reception":
                continue
            if point_in_roi(cx, cy, zone["roi"]):
                return True
        return False

    def observe(
        self,
        subject_id: str,
        feat: np.ndarray | None,
        det: Any,
        frame: np.ndarray | None,
        in_reception: bool,
        now: float | None = None,
    ) -> bool:
        """Gate dwell in reception, then VLM. Returns True if enrolled as staff this call."""
        now = time.time() if now is None else float(now)
        if not subject_id:
            return False
        if not in_reception:
            self.gate.evaluate(subject_id, False, now)
            return False
        if frame is None:
            return False
        if not self.gate.evaluate(subject_id, True, now):
            return False
        bbox = _bbox(det)
        b64_person, b64_context, _, _ = extract_dual_crops(frame, bbox=bbox)
        if self._inline or self._executor is None:
            return self._apply_verdict(subject_id, feat, det, b64_person, b64_context, now)
        self._executor.submit(
            self._apply_verdict, subject_id, feat, det, b64_person, b64_context, now
        )
        return False

    def annotate_known(
        self,
        detections: list[Any],
        embeddings: dict[int, np.ndarray] | None = None,
    ) -> None:
        for det in detections or []:
            if getattr(det, "is_staff", False) or str(getattr(det, "identity", "") or "").startswith(
                STAFF_ID_PREFIX
            ):
                continue
            feat = _feat_from_det(det, embeddings)
            staff_id = self.match(feat)
            if not staff_id:
                continue
            try:
                det.identity = staff_id
                det.is_staff = True
            except Exception:
                pass

    def _apply_verdict(
        self,
        subject_id: str,
        feat: np.ndarray | None,
        det: Any,
        b64_person: str,
        b64_context: str,
        now: float,
    ) -> bool:
        verdict = self.vlm.classify_counter(b64_person, b64_context, subject_id)
        if not verdict.is_staff:
            return False
        staff_id = self.remember(feat, now)
        if not staff_id:
            return False
        try:
            det.identity = staff_id
            det.is_staff = True
        except Exception:
            pass
        return True

    def _add_embedding(self, staff_id: str, vec: np.ndarray, now: float | None) -> None:
        entry = self.entries.get(staff_id)
        if entry is None:
            entry = _StaffEntry(staff_id=staff_id)
            self.entries[staff_id] = entry
        entry.embeddings.append(vec)
        if len(entry.embeddings) > self.max_per_id:
            del entry.embeddings[0 : len(entry.embeddings) - self.max_per_id]
        proto = np.mean(np.stack(entry.embeddings, axis=0), axis=0)
        proto = proto / max(float(np.linalg.norm(proto)), 1e-6)
        self._persist(staff_id, proto, now)

    def _persist(self, staff_id: str, vec: np.ndarray, now: float | None) -> None:
        if self.conn is None:
            return
        from db import upsert_staff_memory

        upsert_staff_memory(self.conn, staff_id, vec, now if now is not None else time.time())

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
