from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

API = "https://api.telegram.org/bot{token}/{method}"


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
    def __init__(self, token: str, chat_id: str) -> None:
        self.token = (token or "").strip()
        self.chat_id = normalize_chat_id(chat_id)

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def _url(self, method: str) -> str:
        return API.format(token=self.token, method=method)

    def set_chat(self, chat_id: str) -> None:
        self.chat_id = normalize_chat_id(chat_id)

    def send_message_to(self, chat_id: str, text: str) -> bool:
        target = normalize_chat_id(chat_id)
        if not self.token or not target:
            print("[telegram] skipped sendMessage (no token/chat_id)")
            return False
        response = requests.post(
            self._url("sendMessage"),
            data={"chat_id": target, "text": text},
            timeout=30,
        )
        if not response.ok:
            print(f"[telegram] sendMessage failed: {response.text}")
            return False
        return True

    def send_message(self, text: str) -> bool:
        if not self.enabled:
            print("[telegram] skipped sendMessage (no token/chat_id)")
            return False
        response = requests.post(
            self._url("sendMessage"),
            data={"chat_id": self.chat_id, "text": text},
            timeout=30,
        )
        if not response.ok:
            print(f"[telegram] sendMessage failed: {response.text}")
            return False
        return True

    def send_photo(self, path: Path, caption: str) -> bool:
        if not self.enabled:
            print(f"[telegram] skipped sendPhoto (kept local): {path}")
            return False
        with path.open("rb") as handle:
            response = requests.post(
                self._url("sendPhoto"),
                data={"chat_id": self.chat_id, "caption": caption[:1024]},
                files={"photo": handle},
                timeout=60,
            )
        if not response.ok:
            print(f"[telegram] sendPhoto failed: {response.text}")
            return False
        return True

    def send_album(self, paths: list[Path], caption: str) -> bool:
        if not paths:
            return True
        if not self.enabled:
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
        finally:
            for handle in handles:
                handle.close()
        if not response.ok:
            print(f"[telegram] sendMediaGroup failed: {response.text}")
            return False
        if len(paths) > 10:
            extra = len(paths) - 10
            self.send_message(
                f"{extra} more proof stills remain on disk (Telegram album max 10)."
            )
        return True
