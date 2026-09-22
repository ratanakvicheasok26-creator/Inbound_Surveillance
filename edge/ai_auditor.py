"""Tier 2 Cloud Vision AI Auditor for Inbound Surveillance."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import requests

logger = logging.getLogger("ai_auditor")


@dataclass
class AIAuditVerdict:
    bay_id: str
    technician_id: str
    action: str
    category: str
    confidence: float
    explanation: str
    is_work_activity: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    crop_path: str | None = None

    @property
    def action_label(self) -> str:
        return self.action

    @property
    def classification(self) -> str:
        return self.category

    @property
    def reasoning(self) -> str:
        return self.explanation

    @property
    def tool_or_object(self) -> str:
        return self.action

    def as_dict(self) -> dict[str, Any]:
        return {
            "bay_id": self.bay_id,
            "technician_id": self.technician_id,
            "action": self.action,
            "action_label": self.action,
            "category": self.category,
            "classification": self.category,
            "confidence": round(self.confidence, 2),
            "explanation": self.explanation,
            "reasoning": self.explanation,
            "tool_or_object": self.action,
            "is_work_activity": self.is_work_activity,
            "timestamp": self.timestamp,
            "crop_path": self.crop_path,
        }


class TokenSaverGate:
    """Gates API requests to avoid redundant token expenditure.

    - duration_threshold: minimum time the suspicious pattern must persist (seconds)
    - cooldown_seconds: minimum gap between consecutive audits for the same bay (seconds)
    - grace_seconds: tolerate momentary skeletal keypoint dropouts/jitter without resetting timer
    """

    def __init__(
        self,
        duration_threshold: float = 0.0,
        cooldown_seconds: float = 45.0,
        grace_seconds: float = 2.0,
    ):
        self.duration_threshold = duration_threshold
        self.cooldown_seconds = cooldown_seconds
        self.grace_seconds = grace_seconds
        self._pattern_starts: dict[str, float] = {}
        self._pattern_last_seen: dict[str, float] = {}
        self._last_audit_times: dict[str, float] = {}

    def evaluate(self, bay_id: str, pattern_matched: bool, now: float | None = None) -> bool:
        if now is None:
            now = time.time()

        if not pattern_matched:
            # Do not immediately reset if tracking dropped for 1-2 frames
            last_seen = self._pattern_last_seen.get(bay_id)
            if last_seen is not None and (now - last_seen) > self.grace_seconds:
                self._pattern_starts.pop(bay_id, None)
                self._pattern_last_seen.pop(bay_id, None)
            return False

        self._pattern_last_seen[bay_id] = now
        if bay_id not in self._pattern_starts:
            self._pattern_starts[bay_id] = now

        # 1. Check pattern duration threshold
        elapsed = now - self._pattern_starts[bay_id]
        if elapsed < self.duration_threshold:
            return False

        # 2. Check cooldown interval
        last_audit = self._last_audit_times.get(bay_id)
        if last_audit is not None and (now - last_audit) < self.cooldown_seconds:
            return False

        self._last_audit_times[bay_id] = now
        self._pattern_starts.pop(bay_id, None)
        self._pattern_last_seen.pop(bay_id, None)
        return True

    def reset_bay(self, bay_id: str) -> None:
        self._pattern_starts.pop(bay_id, None)
        self._pattern_last_seen.pop(bay_id, None)
        self._last_audit_times.pop(bay_id, None)



def extract_dual_crops(frame: np.ndarray, keypoints: list | None = None, bbox: tuple[int, int, int, int] | None = None, max_dim: int = 512, jpeg_quality: int = 80) -> tuple[str, str, np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    if bbox:
        x1, y1, x2, y2 = bbox
        pad_x, pad_y = int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)
        context_img = frame[max(0, y1 - pad_y):min(h, y2 + pad_y), max(0, x1 - pad_x):min(w, x2 + pad_x)].copy()
    else:
        context_img = frame.copy()

    hand_img = None
    if keypoints and len(keypoints) >= 11:
        wrists = [(int(keypoints[i][0]), int(keypoints[i][1])) for i in (9, 10) if len(keypoints[i]) >= 3 and keypoints[i][2] >= 0.25]
        if wrists:
            avg_x, avg_y = int(sum(p[0] for p in wrists) / len(wrists)), int(sum(p[1] for p in wrists) / len(wrists))
            bs = int(min(w, h) * 0.22)
            hand_img = frame[max(0, avg_y - bs):min(h, avg_y + bs), max(0, avg_x - bs):min(w, avg_x + bs)].copy()

    if hand_img is None or hand_img.size == 0:
        hand_img = context_img.copy()

    def _b64(im: np.ndarray) -> str:
        ih, iw = im.shape[:2]
        if max(ih, iw) > max_dim:
            sc = max_dim / max(ih, iw)
            im = cv2.resize(im, (int(iw * sc), int(ih * sc)), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        return base64.b64encode(buf).decode("utf-8")

    return _b64(hand_img), _b64(context_img), hand_img, context_img


def _load_env_or_config() -> tuple[str, str]:
    from paths import data_dir, is_frozen, resource_dir

    api_key = os.environ.get("FIREWORKS_API_KEY", "").strip().strip('"').strip("'")
    model = os.environ.get("FIREWORKS_MODEL", "").strip().strip('"').strip("'")
    data = data_dir()
    resource = resource_dir()
    env_candidates = [data / ".env", resource / ".env"]
    if not is_frozen():
        env_candidates.append(resource.parent / ".env")
    for p in env_candidates:
        if not p.is_file():
            continue
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("FIREWORKS_API_KEY=") and not api_key:
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                elif line.startswith("FIREWORKS_MODEL=") and not model:
                    model = line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    cfg_path = data / "config.yaml"
    if cfg_path.is_file():
        try:
            import yaml

            payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            if not api_key and "fireworks_api_key" in payload:
                api_key = str(payload["fireworks_api_key"]).strip().strip('"').strip("'")
            if not model and "fireworks_model" in payload:
                model = str(payload["fireworks_model"]).strip().strip('"').strip("'")
        except Exception:
            pass
    if not model:
        model = "accounts/fireworks/models/deepseek-v4-flash-vision-exp"
    return api_key, model


class FireworksVLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 25.0,
    ):
        loaded_key, loaded_model = _load_env_or_config()
        self.api_key = api_key if api_key is not None else loaded_key
        self.model = model or loaded_model
        self.timeout = timeout
        self.endpoint = "https://api.fireworks.ai/inference/v1/chat/completions"

        if self.api_key:
            print(f"[AI-Auditor] Ready with Fireworks VLM (model: {self.model})", flush=True)
        else:
            print(
                "[AI-Auditor] NOTICE: No FIREWORKS_API_KEY found. Running in Simulated Audit mode for testing.",
                flush=True,
            )

    def audit_activity(
        self,
        b64_hand: str,
        b64_context: str,
        bay_id: str,
        technician_id: str,
        context_history: str = "",
    ) -> AIAuditVerdict:
        if not self.api_key:
            return self._mock_verdict(bay_id, technician_id)

        history_block = f"- Recent Event History:\n{context_history}" if context_history else "- Recent Event History: None"

        prompt = f"""You are an expert workshop supervisor AI for automotive service bays.

WORKSHOP KNOWLEDGE BASE:
1. Technicians frequently operate black Autel Maxisys / Launch OBD-II diagnostic tablets, multimeters, or clipboards with paper job orders.
   - Using a diagnostic scanner, tablet, multimeter, or technical manual is ACTIVE WORK (DIAGNOSTIC_TOOL or MANUAL_WORK).
2. Personal smartphone usage for non-work entertainment, texting, or gaming is DISTRACTION (PHONE_USAGE).
3. Resting or taking a deliberate break while sitting or leaning is RESTING (ON_BREAK).

CONTEXT & RECENT ACTIVITY:
- Bay: {bay_id}
- Technician: {technician_id}
{history_block}

TASK:
Analyze the attached dual images (Image 1: close-up of hands/object, Image 2: wide body & bay context).
Classify the activity into one of:
- DIAGNOSTIC_TOOL (Active work with diagnostic scanner, OBD tablet, multimeter)
- MANUAL_WORK (Active mechanical work, holding tools, parts)
- DOCUMENT_READING (Reviewing printed work order or manual)
- PHONE_USAGE (Personal smartphone use, distraction)
- RESTING (Idle, sitting without work tools)

Respond STRICTLY in JSON with keys:
{{
  "action": "DIAGNOSTIC_TOOL" | "MANUAL_WORK" | "DOCUMENT_READING" | "PHONE_USAGE" | "RESTING",
  "category": "ACTIVE_WORK" | "DISTRACTION" | "ON_BREAK",
  "confidence": float (0.0 to 1.0),
  "explanation": "1-2 sentence detailed reason citing visual evidence",
  "is_work_activity": boolean
}}
"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_hand}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_context}"},
                        },
                    ],
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
        }
        if "deepseek" in self.model.lower():
            payload["thinking"] = {"type": "disabled"}
        try:
            r = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
            r.raise_for_status()
            res_json = r.json()
            msg = res_json["choices"][0]["message"]
            raw = (msg.get("content") or "").strip()
            if not raw:
                raw = (msg.get("reasoning_content") or "").strip()

            # Clean JSON markdown fences
            if "```json" in raw:
                raw = raw.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in raw:
                raw = raw.split("```", 1)[1].split("```", 1)[0]
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                raw_json = raw[start : end + 1]
            else:
                raw_json = raw.strip()

            data = json.loads(raw_json)
            action = str(data.get("action", "UNKNOWN")).upper()
            cat = str(data.get("category", "UNKNOWN")).upper()
            conf = float(data.get("confidence", 0.85))
            expl = str(data.get("explanation", ""))
            is_work = bool(
                data.get(
                    "is_work_activity",
                    cat in ("ACTIVE_WORK", "WORKING")
                    or action in ("DIAGNOSTIC_TOOL", "MANUAL_WORK", "DOCUMENT_READING"),
                )
            )
            return AIAuditVerdict(
                bay_id=bay_id,
                technician_id=technician_id,
                action=action,
                category=cat,
                confidence=conf,
                explanation=expl,
                is_work_activity=is_work,
            )
        except Exception as e:
            err_detail = str(e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    err_json = e.response.json()
                    if isinstance(err_json, dict) and "error" in err_json:
                        api_err = err_json["error"]
                        if isinstance(api_err, dict) and "message" in api_err:
                            err_detail = f"{e} ({api_err['message']})"
                        elif isinstance(api_err, str):
                            err_detail = f"{e} ({api_err})"
                except Exception:
                    pass
            print(f"[AI-Auditor Error] Fireworks API call failed: {err_detail}", flush=True)
            return AIAuditVerdict(
                bay_id=bay_id,
                technician_id=technician_id,
                action="UNKNOWN",
                category="UNKNOWN",
                confidence=0.0,
                explanation=f"Audit error: {err_detail}",
                is_work_activity=True,
            )

    def _mock_verdict(self, bay_id: str, technician_id: str) -> AIAuditVerdict:
        """Simulated verdict for offline testing when no API key is provided."""
        return AIAuditVerdict(
            bay_id=bay_id,
            technician_id=technician_id,
            action="DIAGNOSTIC_TOOL",
            category="ACTIVE_WORK",
            confidence=0.94,
            explanation="[Simulated AI Audit] Technician is operating an OBD-II diagnostic scan tool connected to the vehicle.",
            is_work_activity=True,
        )


class AIAuditorQueue:
    def __init__(
        self,
        vlm_client: FireworksVLMClient | None = None,
        gate: TokenSaverGate | None = None,
        max_workers: int = 2,
        on_verdict_callback: Callable[[AIAuditVerdict], None] | None = None,
        save_crops_dir: Path | None = None,
    ):
        self.vlm_client = vlm_client or FireworksVLMClient()
        self.gate = gate or TokenSaverGate()
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ai_auditor")
        self.on_verdict_callback = on_verdict_callback
        self.save_crops_dir = save_crops_dir
        self.latest_verdicts: dict[str, AIAuditVerdict] = {}

    def maybe_audit_bay(
        self,
        bay_id: str,
        technician_id: str,
        frame: np.ndarray,
        keypoints: list | None,
        bbox: tuple[int, int, int, int] | None = None,
        pattern_matched: bool = False,
        context_history: str = "",
        now: float | None = None,
    ) -> bool:
        if not self.gate.evaluate(bay_id, pattern_matched, now):
            return False

        b64_h, b64_c, hand_img, _ = extract_dual_crops(frame, keypoints, bbox)
        crop_path = None
        if self.save_crops_dir:
            self.save_crops_dir.mkdir(parents=True, exist_ok=True)
            crop_name = f"audit_{bay_id}_{int(time.time() * 1000)}.jpg"
            crop_path = str(self.save_crops_dir / crop_name)
            cv2.imwrite(crop_path, hand_img)

        print(
            f"[AI-Auditor] Triggering audit for {bay_id} (tech: {technician_id})...",
            flush=True,
        )
        self.executor.submit(
            self._task, b64_h, b64_c, bay_id, technician_id, context_history, crop_path
        )
        return True

    def force_audit(
        self,
        bay_id: str,
        technician_id: str,
        frame: np.ndarray,
        keypoints: list | None = None,
        bbox: tuple[int, int, int, int] | None = None,
        context_history: str = "",
    ) -> bool:
        """Bypasses TokenSaverGate to immediately submit an audit for testing/verification."""
        b64_h, b64_c, hand_img, _ = extract_dual_crops(frame, keypoints, bbox)
        crop_path = None
        if self.save_crops_dir:
            self.save_crops_dir.mkdir(parents=True, exist_ok=True)
            crop_name = f"audit_{bay_id}_{int(time.time() * 1000)}.jpg"
            crop_path = str(self.save_crops_dir / crop_name)
            cv2.imwrite(crop_path, hand_img)

        print(
            f"[AI-Auditor] Force-triggering audit for {bay_id} (tech: {technician_id})...",
            flush=True,
        )
        self.executor.submit(
            self._task, b64_h, b64_c, bay_id, technician_id, context_history, crop_path
        )
        return True

    def _task(self, b64_h, b64_c, bay_id, tech_id, hist, crop_path):
        try:
            v = self.vlm_client.audit_activity(b64_h, b64_c, bay_id, tech_id, hist)
            v.crop_path = crop_path
            self.latest_verdicts[bay_id] = v
            print(
                f"[AI-Auditor] Verdict for {bay_id}: {v.action} ({v.category}, conf={v.confidence:.2f}, work={v.is_work_activity}) - {v.explanation}",
                flush=True,
            )
            if self.on_verdict_callback:
                self.on_verdict_callback(v)
        except Exception as e:
            print(f"[AI-Auditor Task Error] {e}", flush=True)

    def get_latest_verdict(self, bay_id: str, max_age_seconds: float = 600.0) -> AIAuditVerdict | None:
        v = self.latest_verdicts.get(bay_id)
        if v is None:
            return None
        try:
            v_time = datetime.fromisoformat(v.timestamp).timestamp()
            if time.time() - v_time > max_age_seconds:
                return None
        except Exception:
            pass
        return v

    def clear_verdict(self, bay_id: str) -> None:
        self.latest_verdicts.pop(bay_id, None)
        self.gate.reset_bay(bay_id)

    def shutdown(self):
        self.executor.shutdown(wait=False)

