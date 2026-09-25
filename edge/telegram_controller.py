"""Inbound Telegram staff/owner controller for Champei edge.

Polls getUpdates for /name, /info, /scorecard, /churn, /weekly, /help.
Does not touch vision/workplace modules.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import requests

from analytics.churn import compute_churn_risks, format_churn_watchlist
from analytics.scorecard import build_daily_scorecard
from analytics.weekly_customer_brief import build_weekly_customer_brief
from telegram_out import normalize_chat_id
from visitor_registry import get_visitor_display_name, set_visitor_alias

API = "https://api.telegram.org/bot{token}/{method}"

DENIED = "⛔ Permission denied. Management only."


def _strip_bot_command(cmd: str) -> str:
    """Normalize '/name@BotName' → '/name'."""
    text = str(cmd or "").strip()
    if not text:
        return ""
    head = text.split(maxsplit=1)[0]
    if "@" in head:
        head = head.split("@", 1)[0]
    return head.lower()


def parse_name_args(payload: str) -> tuple[str, str, str, str] | None:
    """Parse ``<visitor_id> <alias> [| <tier> | <notes>]``.

    Returns (visitor_id, alias, vip_tier, notes) or None if invalid.
    Default tier is Standard (matches set_visitor_alias).
    """
    text = str(payload or "").strip()
    if not text:
        return None
    pipe_parts = [p.strip() for p in text.split("|")]
    head = pipe_parts[0]
    bits = head.split(maxsplit=1)
    if len(bits) < 2:
        return None
    visitor_id = bits[0].strip()
    alias = bits[1].strip()
    if not visitor_id or not alias:
        return None
    tier = "Standard"
    notes = ""
    if len(pipe_parts) >= 2 and pipe_parts[1]:
        tier = pipe_parts[1].strip() or "Standard"
    if len(pipe_parts) >= 3:
        notes = pipe_parts[2].strip()
    return visitor_id, alias, tier, notes


class TelegramController:
    """Role-based inbound Telegram command poller."""

    def __init__(
        self,
        *,
        token: str | None = None,
        staff_chat_id: str | None = None,
        owner_chat_id: str | None = None,
        db_path: Path | str | None = None,
        branch_id: str = "champei-pp-01",
        session: requests.Session | None = None,
    ) -> None:
        self.token = (
            str(token if token is not None else os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
        )
        self.staff_chat_id = normalize_chat_id(
            staff_chat_id
            if staff_chat_id is not None
            else os.environ.get("TELEGRAM_STAFF_CHAT_ID", "")
        )
        self.owner_chat_id = normalize_chat_id(
            owner_chat_id
            if owner_chat_id is not None
            else os.environ.get("TELEGRAM_OWNER_CHAT_ID", "")
        )
        self.db_path = Path(db_path) if db_path is not None else None
        self.branch_id = str(branch_id or "champei-pp-01").strip() or "champei-pp-01"
        self._session = session or requests.Session()
        self._offset = 0
        self._last_error = ""
        self._backoff = 1.0

    def role_for(self, chat_id: str) -> str | None:
        cid = normalize_chat_id(chat_id)
        if not cid:
            return None
        if self.owner_chat_id and cid == self.owner_chat_id:
            return "owner"
        if self.staff_chat_id and cid == self.staff_chat_id:
            return "staff"
        return None

    def send_reply(self, chat_id: str, text: str) -> bool:
        if not self.token or not chat_id or not text:
            return False
        try:
            response = self._session.post(
                API.format(token=self.token, method="sendMessage"),
                json={"chat_id": chat_id, "text": text},
                timeout=20,
            )
            data = response.json() if response.content else {}
            return bool(response.ok and data.get("ok"))
        except Exception as exc:
            self._last_error = str(exc)
            return False

    def handle_message(self, chat_id: str, text: str) -> str | None:
        """Process one message. Returns reply text, or None if dropped."""
        role = self.role_for(chat_id)
        if role is None:
            return None

        raw = str(text or "").strip()
        if not raw:
            return None
        parts = raw.split(maxsplit=1)
        cmd = _strip_bot_command(parts[0])
        payload = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/help":
            return self._help_text(role)
        if cmd == "/name":
            return self._cmd_name(payload)
        if cmd == "/info":
            return self._cmd_info(payload)
        if cmd == "/scorecard":
            if role != "owner":
                return DENIED
            return build_daily_scorecard(db_path=self.db_path, branch_id=self.branch_id)
        if cmd == "/churn":
            if role != "owner":
                return DENIED
            risks = compute_churn_risks(db_path=self.db_path)
            return format_churn_watchlist(risks, branch_id=self.branch_id)
        if cmd == "/weekly":
            if role != "owner":
                return DENIED
            return build_weekly_customer_brief(
                db_path=self.db_path,
                branch_id=self.branch_id,
            )
        return (
            f"Unknown command. Try /help.\n\n{self._help_text(role)}"
            if cmd.startswith("/")
            else None
        )

    def _help_text(self, role: str) -> str:
        lines = [
            "Champei staff commands:",
            "/name <visitor_id> <alias> [| <tier> | <notes>]",
            "/info <visitor_id>",
            "/help",
        ]
        if role == "owner":
            lines.extend(["/scorecard", "/churn", "/weekly"])
        return "\n".join(lines)

    def _cmd_name(self, payload: str) -> str:
        parsed = parse_name_args(payload)
        if parsed is None:
            return "Usage: /name <visitor_id> <alias> [| <tier> | <notes>]"
        visitor_id, alias, tier, notes = parsed
        try:
            info = set_visitor_alias(
                visitor_id,
                alias,
                vip_tier=tier,
                notes=notes,
                db_path=self.db_path,
            )
        except ValueError as exc:
            return f"Error: {exc}"
        display = info.get("display_name") or alias
        vip = info.get("tier") or tier
        return f"✅ Saved profile for {visitor_id} → {display} ({vip})"

    def _cmd_info(self, payload: str) -> str:
        visitor_id = str(payload or "").strip().split(maxsplit=1)[0] if payload else ""
        if not visitor_id:
            return "Usage: /info <visitor_id>"
        info = get_visitor_display_name(visitor_id, db_path=self.db_path)
        name = info.get("display_name") or visitor_id
        tier = info.get("tier") or "Standard"
        notes = str(info.get("notes") or "").strip() or "—"
        named = "named" if info.get("is_named") else "unnamed"
        return (
            f"👤 {visitor_id}\n"
            f"Name: {name} ({named})\n"
            f"Tier: {tier}\n"
            f"Notes: {notes}"
        )

    def poll_once(self, timeout: int = 10) -> int:
        if not self.token:
            return 0
        try:
            response = self._session.get(
                API.format(token=self.token, method="getUpdates"),
                params={
                    "offset": self._offset,
                    "timeout": max(0, int(timeout)),
                    "limit": 20,
                },
                timeout=max(3, int(timeout) + 2),
            )
            data = response.json() if response.content else {}
        except Exception as exc:
            self._last_error = str(exc)
            raise

        if not response.ok or not data.get("ok"):
            err = data.get("description") or f"HTTP {response.status_code}"
            self._last_error = err
            return 0

        updates = data.get("result") or []
        handled = 0
        for update in updates:
            update_id = int(update.get("update_id") or 0)
            self._offset = max(self._offset, update_id + 1)
            if self._dispatch_update(update):
                handled += 1
        self._last_error = ""
        self._backoff = 1.0
        return handled

    def _dispatch_update(self, update: dict[str, Any]) -> bool:
        message = update.get("message") or update.get("edited_message") or {}
        text = str(message.get("text") or "").strip()
        if not text:
            return False
        chat = message.get("chat") or {}
        chat_id = normalize_chat_id(chat.get("id"))
        if not chat_id:
            return False
        reply = self.handle_message(chat_id, text)
        if reply is None:
            return False
        self.send_reply(chat_id, reply)
        return True

    def run_loop(self, stop_event: threading.Event | None = None) -> None:
        stop = stop_event if stop_event is not None else threading.Event()
        while not stop.is_set():
            if not self.token:
                if stop.wait(0.5):
                    break
                continue
            if stop.is_set():
                break
            try:
                self.poll_once(timeout=2)
                self._backoff = 1.0
            except Exception as exc:
                self._last_error = str(exc)
                delay = min(60.0, float(self._backoff))
                print(
                    f"[telegram_controller] poll error: {exc}; backoff {delay:.0f}s",
                    flush=True,
                )
                stop.wait(delay)
                self._backoff = min(60.0, self._backoff * 2.0)
                continue
            if stop.wait(0.05):
                break

    @property
    def last_error(self) -> str:
        return self._last_error
