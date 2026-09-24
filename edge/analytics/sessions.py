"""Session lifecycle helpers: completion, early departure, lobby walk-aways.

Vision teammates call these after a visit ends or a bounce is observed.
Does not touch pose/tracker/visit-monitor modules.
"""

from __future__ import annotations

import secrets
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from paths import data_dir
from telegram_out import TelegramOut
from visitor_registry import get_visitor_display_name

EARLY_DEPARTURE_SECONDS = 1200  # 20 minutes


def _default_db_path() -> Path:
    return data_dir() / "events.db"


def _connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else _default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _parse_dt(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    # Normalize trailing Z
    if text.endswith("Z"):
        text = text[:-1]
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(text[:26], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def ensure_session_columns(conn: sqlite3.Connection) -> None:
    """Safely add ended_at / duration_seconds / status to customer_visits if missing."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS customer_visits (
            id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL,
            zone_id TEXT NOT NULL DEFAULT 'waiting',
            started_at TEXT NOT NULL,
            ended_at TEXT,
            source TEXT NOT NULL DEFAULT 'edge'
        )
        """
    )
    # PRAGMA rows: (cid, name, type, notnull, dflt_value, pk) — works with or without Row factory
    cols = {r[1] for r in conn.execute("PRAGMA table_info(customer_visits)").fetchall()}
    if "ended_at" not in cols:
        try:
            conn.execute("ALTER TABLE customer_visits ADD COLUMN ended_at TEXT")
        except Exception:
            pass
    if "duration_seconds" not in cols:
        try:
            conn.execute("ALTER TABLE customer_visits ADD COLUMN duration_seconds REAL")
        except Exception:
            pass
    if "status" not in cols:
        try:
            conn.execute(
                "ALTER TABLE customer_visits ADD COLUMN status TEXT DEFAULT 'completed'"
            )
        except Exception:
            pass
    conn.commit()


def ensure_events_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            event_type TEXT NOT NULL,
            abs_path TEXT,
            branch_id TEXT,
            camera_role TEXT
        )
        """
    )
    cols = {r[1] for r in conn.execute("PRAGMA table_info(events)").fetchall()}
    if "branch_id" not in cols:
        try:
            conn.execute("ALTER TABLE events ADD COLUMN branch_id TEXT")
        except Exception:
            pass
    conn.commit()


def record_session_completion(
    subject_id: str,
    ended_at: str | None = None,
    db_path: Path | str | None = None,
    branch_id: str = "champei-pp-01",
    telegram: TelegramOut | None = None,
) -> dict[str, Any]:
    """Close the latest open visit for subject_id and optionally alert on early exit."""
    sid = str(subject_id or "").strip()
    if not sid:
        return {
            "subject_id": "",
            "duration_seconds": 0,
            "is_early_departure": False,
        }

    path = Path(db_path) if db_path is not None else None
    end_raw = str(ended_at).strip() if ended_at else datetime.now().isoformat(timespec="seconds")
    end_dt = _parse_dt(end_raw) or datetime.now()
    end_iso = end_dt.isoformat(timespec="seconds")

    conn = _connect(path)
    try:
        ensure_session_columns(conn)
        row = conn.execute(
            """
            SELECT id, subject_id, started_at, ended_at
            FROM customer_visits
            WHERE subject_id = ? AND ended_at IS NULL
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (sid,),
        ).fetchone()
        if row is None:
            row = conn.execute(
                """
                SELECT id, subject_id, started_at, ended_at
                FROM customer_visits
                WHERE subject_id = ?
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (sid,),
            ).fetchone()
        if row is None:
            return {
                "subject_id": sid,
                "duration_seconds": 0,
                "is_early_departure": False,
            }

        start_dt = _parse_dt(str(row["started_at"] or ""))
        if start_dt is None:
            start_dt = end_dt
        duration_seconds = max(0, int((end_dt - start_dt).total_seconds()))

        conn.execute(
            """
            UPDATE customer_visits
            SET ended_at = ?, duration_seconds = ?, status = 'completed'
            WHERE id = ?
            """,
            (end_iso, float(duration_seconds), row["id"]),
        )
        conn.commit()
    finally:
        conn.close()

    is_early = duration_seconds < EARLY_DEPARTURE_SECONDS
    if is_early:
        meta = get_visitor_display_name(sid, db_path=path)
        display = str(meta.get("display_name") or sid)
        duration_minutes = max(0, int(round(duration_seconds / 60.0)))
        branch = str(branch_id or "champei-pp-01").strip() or "champei-pp-01"
        text = (
            f"⚠️ [{branch}] EARLY DEPARTURE ALERT\n"
            f"Guest: {display}\n"
            f"Duration: {duration_minutes} mins\n"
            f"Notice: Service duration dropped below standard threshold. "
            f"Possible service or customer issue."
        )
        bot = telegram if telegram is not None else TelegramOut()
        try:
            bot.send_alert("early_departure", text)
        except Exception as exc:
            print(f"[sessions] early_departure send failed: {exc}", flush=True)

    return {
        "subject_id": sid,
        "duration_seconds": duration_seconds,
        "is_early_departure": is_early,
    }


def record_walk_away(
    branch_id: str = "champei-pp-01",
    ts: str | None = None,
    db_path: Path | str | None = None,
) -> bool:
    """Log a lobby bounce / walk-away event for scorecard analytics."""
    path = Path(db_path) if db_path is not None else None
    stamp = str(ts).strip() if ts else datetime.now().isoformat(timespec="seconds")
    branch = str(branch_id or "champei-pp-01").strip() or "champei-pp-01"
    conn = _connect(path)
    try:
        ensure_events_table(conn)
        conn.execute(
            "INSERT INTO events (ts, event_type, abs_path, branch_id) VALUES (?, ?, ?, ?)",
            (stamp, "walk_away", f"walk_away-{secrets.token_hex(4)}", branch),
        )
        conn.commit()
        return True
    except Exception as exc:
        print(f"[sessions] record_walk_away failed: {exc}", flush=True)
        return False
    finally:
        conn.close()
