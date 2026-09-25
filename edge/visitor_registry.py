"""Local visitor metadata + hospitality Telegram handshake.

Vision engineers call ``notify_guest_arrival`` after a completed arrival detection.
This module never touches pose/tracker/visit-monitor code.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from paths import data_dir
from telegram_out import TelegramOut, format_alert_header

ARRIVAL_COOLDOWN_SECONDS = 3600.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS visitor_meta (
    visitor_id TEXT PRIMARY KEY,
    alias TEXT,
    vip_tier TEXT DEFAULT 'Standard',
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

# visitor_id -> monotonic timestamp of last successful notify
_last_notify: dict[str, float] = {}


def reset_notify_cooldowns() -> None:
    """Test helper: clear in-memory arrival cooldown state."""
    _last_notify.clear()


def default_db_path() -> Path:
    return data_dir() / "events.db"


def connect_registry(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    ensure_visitor_meta(conn)
    return conn


def ensure_visitor_meta(conn: sqlite3.Connection) -> None:
    conn.execute(_SCHEMA)
    conn.commit()


def get_visitor_display_name(
    visitor_id: str,
    *,
    conn: sqlite3.Connection | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Return display fields for a visitor. Unnamed guests fall back to visitor_id."""
    vid = str(visitor_id or "").strip()
    own = conn is None
    db = conn if conn is not None else connect_registry(db_path)
    try:
        if not vid:
            return {
                "display_name": "",
                "tier": "Standard",
                "notes": "",
                "is_named": False,
            }
        row = db.execute(
            "SELECT alias, vip_tier, notes FROM visitor_meta WHERE visitor_id = ?",
            (vid,),
        ).fetchone()
        if row is None:
            return {
                "display_name": vid,
                "tier": "Standard",
                "notes": "",
                "is_named": False,
            }
        alias = str(row["alias"] or "").strip()
        tier = str(row["vip_tier"] or "Standard").strip() or "Standard"
        notes = str(row["notes"] or "").strip()
        if alias:
            return {
                "display_name": alias,
                "tier": tier,
                "notes": notes,
                "is_named": True,
            }
        return {
            "display_name": vid,
            "tier": tier,
            "notes": notes,
            "is_named": False,
        }
    finally:
        if own:
            db.close()


def set_visitor_alias(
    visitor_id: str,
    alias: str,
    vip_tier: str = "Standard",
    notes: str = "",
    *,
    conn: sqlite3.Connection | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Upsert visitor_meta for the given opaque visitor_id."""
    vid = str(visitor_id or "").strip()
    if not vid:
        raise ValueError("visitor_id is required")
    name = str(alias or "").strip()[:64]
    if not name:
        raise ValueError("alias/name is required")
    tier = str(vip_tier or "Standard").strip()[:32] or "Standard"
    note = str(notes or "").strip()[:256]

    own = conn is None
    db = conn if conn is not None else connect_registry(db_path)
    try:
        db.execute(
            """
            INSERT INTO visitor_meta (visitor_id, alias, vip_tier, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(visitor_id) DO UPDATE SET
                alias = excluded.alias,
                vip_tier = excluded.vip_tier,
                notes = excluded.notes,
                updated_at = CURRENT_TIMESTAMP
            """,
            (vid, name, tier, note),
        )
        db.commit()
        return get_visitor_display_name(vid, conn=db)
    finally:
        if own:
            db.close()


def format_guest_arrival_message(
    visitor_id: str,
    visit_count: int,
    branch_id: str = "champei-pp-01",
    zone: str = "shoe_lounge",
    *,
    meta: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Return (event_type, message_text) for an arrival."""
    info = meta if meta is not None else get_visitor_display_name(visitor_id)
    count = max(1, int(visit_count or 1))
    branch = str(branch_id or "champei-pp-01").strip() or "champei-pp-01"
    zone_label = str(zone or "shoe_lounge").strip() or "shoe_lounge"

    if info.get("is_named"):
        notes = str(info.get("notes") or "").strip() or "—"
        text = (
            f"✨ {format_alert_header(branch, 'VIP ARRIVAL')}: "
            f"{info['display_name']} (Visit #{count}) | "
            f"Tier: {info.get('tier') or 'Standard'} | Notes: {notes}"
        )
        return "vip_arrival", text

    if count > 1:
        text = (
            f"👋 {format_alert_header(branch, 'RETURNING GUEST')}: "
            f"{visitor_id} (Visit #{count}) at {zone_label}."
        )
        return "guest_arrival", text

    text = f"🆕 {format_alert_header(branch, 'NEW GUEST DETECTED')} at {zone_label}."
    return "guest_arrival", text


def notify_guest_arrival(
    visitor_id: str,
    visit_count: int,
    branch_id: str = "champei-pp-01",
    zone: str = "shoe_lounge",
    *,
    telegram: TelegramOut | None = None,
    conn: sqlite3.Connection | None = None,
    db_path: Path | None = None,
    cooldown_seconds: float = ARRIVAL_COOLDOWN_SECONDS,
    now: float | None = None,
) -> bool:
    """Look up alias, format a privacy-safe message, dispatch to staff Telegram.

    Rate-limits: same visitor_id will not alert again within ``cooldown_seconds``
    (default 60 minutes). Returns False when skipped or send failed.
    """
    vid = str(visitor_id or "").strip()
    if not vid:
        return False

    ts = time.monotonic() if now is None else float(now)
    last = _last_notify.get(vid)
    if last is not None and (ts - last) < float(cooldown_seconds):
        return False

    meta = get_visitor_display_name(vid, conn=conn, db_path=db_path)
    event_type, text = format_guest_arrival_message(
        vid,
        visit_count,
        branch_id=branch_id,
        zone=zone,
        meta=meta,
    )

    bot = telegram if telegram is not None else TelegramOut()
    ok = bool(bot.send_alert(event_type, text))
    if ok:
        _last_notify[vid] = ts
    return ok
