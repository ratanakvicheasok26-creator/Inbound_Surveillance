from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import requests

API = "https://api.telegram.org/bot{token}/{method}"

STAFF_EVENTS = frozenset({"wait_bottleneck", "vip_arrival", "guest_arrival", "early_departure"})
OWNER_EVENTS = frozenset(
    {"daily_scorecard", "silent_churn", "early_departure", "weekly_customer_brief"}
)


def format_alert_header(branch_id: str, title: str) -> str:
    """Hospitality-safe prefix: [branch_id] TITLE."""
    branch = str(branch_id or "").strip() or "branch"
    event = str(title or "").strip() or "ALERT"
    return f"[{branch}] {event}"


def resolve_telegram_credentials(
    token: Any = None,
    chat_id: Any = None,
) -> tuple[str, str]:
    """Prefer TELEGRAM_* env vars; fall back to config values. Never invent a token."""
    env_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    env_chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    cfg_token = str(token or "").strip()
    cfg_chat = normalize_chat_id(chat_id)
    return (env_token or cfg_token, normalize_chat_id(env_chat) or cfg_chat)


def resolve_routed_chat_ids(primary_chat_id: str = "") -> tuple[str, str]:
    """Return (staff_chat_id, owner_chat_id); each falls back to primary TELEGRAM_CHAT_ID."""
    primary = normalize_chat_id(
        os.environ.get("TELEGRAM_CHAT_ID", "").strip() or primary_chat_id
    )
    staff = normalize_chat_id(os.environ.get("TELEGRAM_STAFF_CHAT_ID", "").strip()) or primary
    owner = normalize_chat_id(os.environ.get("TELEGRAM_OWNER_CHAT_ID", "").strip()) or primary
    return staff, owner


def normalize_chat_id(value: Any) -> str:
    """Return a single Telegram chat id. If several were stored, keep the latest."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        parts = [normalize_chat_id(item) for item in value]
        parts = [part for part in parts if part]
        return parts[-1] if parts else ""
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return normalize_chat_id(parsed)
    for sep in ("\n", ";", ","):
        if sep in text:
            parts = [part.strip() for part in text.split(sep) if part.strip()]
            return parts[-1] if parts else ""
    return text


class TelegramOut:
    def __init__(self, token: str = "", chat_id: str = "") -> None:
        resolved_token, resolved_chat = resolve_telegram_credentials(token, chat_id)
        self.token = resolved_token
        self.chat_id = resolved_chat
        self.staff_chat_id, self.owner_chat_id = resolve_routed_chat_ids(resolved_chat)

    def refresh_from_env(self, token: str = "", chat_id: str = "") -> None:
        """Re-resolve credentials (env wins over passed config)."""
        self.token, self.chat_id = resolve_telegram_credentials(token, chat_id)
        self.staff_chat_id, self.owner_chat_id = resolve_routed_chat_ids(self.chat_id)

    @property
    def enabled(self) -> bool:
        return bool(self.token and (self.chat_id or self.staff_chat_id or self.owner_chat_id))

    def _url(self, method: str) -> str:
        return API.format(token=self.token, method=method)

    def set_chat(self, chat_id: str) -> None:
        self.chat_id = normalize_chat_id(chat_id)
        if not self.staff_chat_id:
            self.staff_chat_id = self.chat_id
        if not self.owner_chat_id:
            self.owner_chat_id = self.chat_id

    def resolve_chat_for_event(self, event_type: str) -> str:
        kind = str(event_type or "").strip().lower()
        if kind in OWNER_EVENTS:
            return self.owner_chat_id or self.chat_id
        if kind in STAFF_EVENTS:
            return self.staff_chat_id or self.chat_id
        return self.chat_id or self.staff_chat_id or self.owner_chat_id

    def send_message_to(self, chat_id: str, text: str) -> bool:
        target = normalize_chat_id(chat_id)
        if not self.token or not target:
            print("[telegram] skipped sendMessage (no token/chat_id)")
            return False
        try:
            response = requests.post(
                self._url("sendMessage"),
                data={"chat_id": target, "text": text},
                timeout=30,
            )
        except Exception as exc:
            print(f"[telegram] sendMessage error: {exc}")
            return False
        if not response.ok:
            print(f"[telegram] sendMessage failed: {response.text}")
            return False
        return True

    def send_message(self, text: str) -> bool:
        if not self.token or not self.chat_id:
            print("[telegram] skipped sendMessage (no token/chat_id)")
            return False
        return self.send_message_to(self.chat_id, text)

    def send_photo_bytes_to(self, chat_id: str, photo: bytes, caption: str = "") -> bool:
        target = normalize_chat_id(chat_id)
        if not self.token or not target or not photo:
            print("[telegram] skipped sendPhoto bytes (no token/chat_id/photo)")
            return False
        try:
            response = requests.post(
                self._url("sendPhoto"),
                data={"chat_id": target, "caption": (caption or "")[:1024]},
                files={"photo": ("alert.jpg", BytesIO(photo), "image/jpeg")},
                timeout=60,
            )
        except Exception as exc:
            print(f"[telegram] sendPhoto bytes error: {exc}")
            return False
        if not response.ok:
            print(f"[telegram] sendPhoto failed: {response.text}")
            return False
        return True

    def resolve_chats_for_event(self, event_type: str) -> list[str]:
        """Return one or more chat ids for an event (dual-route when in both sets)."""
        kind = str(event_type or "").strip().lower()
        in_staff = kind in STAFF_EVENTS
        in_owner = kind in OWNER_EVENTS
        targets: list[str] = []
        if in_staff and in_owner:
            staff = normalize_chat_id(self.staff_chat_id) or normalize_chat_id(self.chat_id)
            owner = normalize_chat_id(self.owner_chat_id) or normalize_chat_id(self.chat_id)
            for cid in (staff, owner):
                if cid and cid not in targets:
                    targets.append(cid)
            return targets
        primary = self.resolve_chat_for_event(kind)
        return [primary] if primary else []

    def send_alert(
        self,
        event_type: str,
        text: str,
        photo: bytes | None = None,
    ) -> bool:
        """Route an alert by event_type. Dual-routes early_departure to staff+owner. Never raises."""
        try:
            targets = self.resolve_chats_for_event(event_type)
            if not self.token or not targets:
                print(f"[telegram] skipped send_alert({event_type}) (no token/chat_id)")
                return False
            any_ok = False
            for target in targets:
                if photo:
                    ok = self.send_photo_bytes_to(target, photo, caption=text)
                else:
                    ok = self.send_message_to(target, text)
                any_ok = any_ok or ok
            return any_ok
        except Exception as exc:
            print(f"[telegram] send_alert({event_type}) error: {exc}")
            return False

    def send_photo(self, path: Path, caption: str) -> bool:
        if not self.token or not self.chat_id:
            print(f"[telegram] skipped sendPhoto (kept local): {path}")
            return False
        try:
            with path.open("rb") as handle:
                response = requests.post(
                    self._url("sendPhoto"),
                    data={"chat_id": self.chat_id, "caption": caption[:1024]},
                    files={"photo": handle},
                    timeout=60,
                )
        except Exception as exc:
            print(f"[telegram] sendPhoto error: {exc}")
            return False
        if not response.ok:
            print(f"[telegram] sendPhoto failed: {response.text}")
            return False
        return True

    def send_album(self, paths: list[Path], caption: str) -> bool:
        if not paths:
            return True
        if not self.token or not self.chat_id:
            print(f"[telegram] skipped album of {len(paths)} stills")
            return False
        batch = paths[:10]
        media = []
        files: dict[str, tuple[str, object, str]] = {}
        handles = []
        try:
            for index, path in enumerate(batch):
                key = f"photo{index}"
                handle = path.open("rb")
                handles.append(handle)
                files[key] = (path.name, handle, "image/jpeg")
                item: dict[str, str] = {"type": "photo", "media": f"attach://{key}"}
                if index == 0 and caption:
                    item["caption"] = caption[:1024]
                media.append(item)
            response = requests.post(
                self._url("sendMediaGroup"),
                data={"chat_id": self.chat_id, "media": json.dumps(media)},
                files=files,
                timeout=120,
            )
        except Exception as exc:
            print(f"[telegram] sendMediaGroup error: {exc}")
            return False
        finally:
            for handle in handles:
                handle.close()
        if not response.ok:
            print(f"[telegram] sendMediaGroup failed: {response.text}")
            return False
        return True

    def send_voice(self, path: Path, caption: str = "") -> bool:
        if not self.token or not self.chat_id:
            print(f"[telegram] skipped sendVoice (kept local): {path}")
            return False
        if not path.exists():
            print(f"[telegram] sendVoice file not found: {path}")
            return False
        try:
            with path.open("rb") as handle:
                response = requests.post(
                    self._url("sendVoice"),
                    data={"chat_id": self.chat_id, "caption": caption[:1024]},
                    files={"voice": handle},
                    timeout=60,
                )
        except Exception as exc:
            print(f"[telegram] sendVoice error: {exc}")
            return False
        if not response.ok:
            return self.send_audio(path, caption)
        return True

    def send_audio(self, path: Path, caption: str = "") -> bool:
        if not self.token or not self.chat_id:
            print(f"[telegram] skipped sendAudio (kept local): {path}")
            return False
        if not path.exists():
            print(f"[telegram] sendAudio file not found: {path}")
            return False
        try:
            with path.open("rb") as handle:
                response = requests.post(
                    self._url("sendAudio"),
                    data={"chat_id": self.chat_id, "caption": caption[:1024]},
                    files={"audio": handle},
                    timeout=60,
                )
        except Exception as exc:
            print(f"[telegram] sendAudio error: {exc}")
            return False
        if not response.ok:
            print(f"[telegram] sendAudio failed: {response.text}")
            return False
        return True

    def send_complaint_alert(
        self,
        complaint: dict,
        audio_path: Path | None = None,
        photo_path: Path | None = None,
    ) -> bool:
        """Send complete 3-part complaint evidence to the linked Telegram chat."""
        if not self.enabled:
            print("[telegram] skipped send_complaint_alert (Telegram not linked)")
            return False

        cid = complaint.get("complaint_id", "CMP-UNKNOWN")
        cat = str(complaint.get("category", "General")).replace("_", " ").title()
        sev = str(complaint.get("severity", "Medium")).upper()
        km = complaint.get("khmer_transcript", "")
        en = complaint.get("english_transcript", "")
        summary = complaint.get("summary", "")
        ts = complaint.get("timestamp", "")
        customer = complaint.get("customer_id") or "Guest"
        branch = str(complaint.get("branch_id") or "").strip()

        sev_emoji = "🔴" if sev in ("HIGH", "CRITICAL") else "🟡" if sev == "MEDIUM" else "🔵"
        header = (
            format_alert_header(branch, "CUSTOMER COMPLAINT")
            if branch
            else "CUSTOMER COMPLAINT DETECTED"
        )

        msg = (
            f"🚨 *{header}*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📋 *ID:* `{cid}`\n"
            f"👤 *Person:* {customer}\n"
            f"🏷️ *Category:* {cat}\n"
            f"{sev_emoji} *Severity:* {sev}\n"
            f"⏰ *Time:* `{ts}`\n\n"
            f"🇰🇭 *Khmer Transcript:*\n_{km}_\n\n"
            f"🇬🇧 *English Translation:*\n_{en}_\n\n"
            f"📝 *AI Summary:*\n{summary}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🎤 Original audio & 📸 Camera frame attached below:"
        )

        ok_text = self.send_message(msg)

        ok_audio = True
        if audio_path and audio_path.exists():
            ok_audio = self.send_voice(audio_path, caption=f"🎤 Original Guest Voice [{cid}]")

        ok_photo = True
        if photo_path and photo_path.exists():
            ok_photo = self.send_photo(
                photo_path, caption=f"📸 Camera Frame at Complaint Moment [{cid}]"
            )

        return ok_text or ok_audio or ok_photo
