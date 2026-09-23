"""Daily operations scorecard for Champei owner Telegram."""

from __future__ import annotations

import argparse
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from paths import data_dir
from telegram_out import TelegramOut


def _default_db_path() -> Path:
    return data_dir() / "events.db"


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else _default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def query_daily_counts(
    conn: sqlite3.Connection,
    day: str,
) -> dict[str, int]:
    """Return total_visits, unique_guests, bottlenecks for a local YYYY-MM-DD day."""
    total = 0
    uniques = 0
    bottlenecks = 0
    prefix = f"{day}%"

    if _table_exists(conn, "customer_visits"):
        total = int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM customer_visits WHERE started_at LIKE ?",
                (prefix,),
            ).fetchone()["n"]
            or 0
        )
        uniques = int(
            conn.execute(
                "SELECT COUNT(DISTINCT subject_id) AS n FROM customer_visits WHERE started_at LIKE ?",
                (prefix,),
            ).fetchone()["n"]
            or 0
        )

    if _table_exists(conn, "events"):
        bottlenecks = int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM events WHERE event_type = 'wait_bottleneck' AND ts LIKE ?",
                (prefix,),
            ).fetchone()["n"]
            or 0
        )

    return {
        "total_visits": total,
        "unique_guests": uniques,
        "bottlenecks": bottlenecks,
    }


def build_daily_scorecard(
    db_path: Path | str | None = None,
    branch_id: str = "champei-pp-01",
    day: str | date | None = None,
) -> str:
    """Format the owner-facing daily operations scorecard text."""
    day_str = (
        day.isoformat()
        if isinstance(day, date)
        else (str(day).strip() if day else date.today().isoformat())
    )
    branch = str(branch_id or "champei-pp-01").strip() or "champei-pp-01"
    path = Path(db_path) if db_path is not None else None
    conn = _connect(path)
    try:
        counts = query_daily_counts(conn, day_str)
    finally:
        conn.close()

    return (
        f"📊 [{branch}] DAILY OPERATIONS SCORECARD\n"
        f"Date: {day_str}\n"
        f"• Total Visits: {counts['total_visits']}\n"
        f"• Unique Guests: {counts['unique_guests']}\n"
        f"• Front-Desk Bottlenecks (>3m): {counts['bottlenecks']}"
    )


def send_daily_scorecard(
    db_path: Path | str | None = None,
    branch_id: str = "champei-pp-01",
    day: str | date | None = None,
    *,
    telegram: TelegramOut | None = None,
) -> bool:
    """Build scorecard and dispatch to owner chat via daily_scorecard route."""
    text = build_daily_scorecard(db_path=db_path, branch_id=branch_id, day=day)
    bot = telegram if telegram is not None else TelegramOut()
    return bool(bot.send_alert("daily_scorecard", text))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Champei daily operations scorecard")
    parser.add_argument("--dry-run", action="store_true", help="Print text; do not send Telegram")
    parser.add_argument("--branch", default="champei-pp-01", help="Branch id for the header")
    parser.add_argument("--db", default="", help="Optional path to events.db")
    parser.add_argument("--day", default="", help="Optional YYYY-MM-DD (default: today)")
    args = parser.parse_args(argv)

    db_path: Path | None = Path(args.db).expanduser() if args.db else None
    day = args.day.strip() or None
    text = build_daily_scorecard(db_path=db_path, branch_id=args.branch, day=day)
    if args.dry_run:
        print(text)
        return 0
    ok = send_daily_scorecard(db_path=db_path, branch_id=args.branch, day=day)
    print(text)
    print("[scorecard] sent" if ok else "[scorecard] send skipped/failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
