"""Silent churn watchlist for Champei owner Telegram."""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from db import connect as db_connect
from paths import data_dir
from telegram_out import TelegramOut
from visitor_registry import get_visitor_display_name


def _default_db_path() -> Path:
    return data_dir() / "events.db"


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else _default_db_path()
    return db_connect(path)


def _parse_day(value: str) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    # Accept ISO datetime or date prefix
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def compute_churn_risks(
    db_path: Path | str | None = None,
    threshold_multiplier: float = 2.0,
    min_days_overdue: int = 21,
    as_of: date | str | None = None,
) -> list[dict[str, Any]]:
    """Return at-risk visitors sorted by days_overdue descending."""
    if isinstance(as_of, date):
        as_of_day = as_of
    elif as_of:
        parsed = _parse_day(str(as_of))
        as_of_day = parsed or date.today()
    else:
        as_of_day = date.today()

    path = Path(db_path) if db_path is not None else None
    conn = _connect(path)
    try:
        if not _table_exists(conn, "customer_visits"):
            return []
        rows = conn.execute(
            "SELECT subject_id, started_at FROM customer_visits ORDER BY subject_id, started_at"
        ).fetchall()
    finally:
        conn.close()

    by_subject: dict[str, list[date]] = defaultdict(list)
    for row in rows:
        sid = str(row["subject_id"] or "").strip()
        day = _parse_day(str(row["started_at"] or ""))
        if not sid or day is None:
            continue
        if not by_subject[sid] or by_subject[sid][-1] != day:
            by_subject[sid].append(day)

    risks: list[dict[str, Any]] = []
    mult = float(threshold_multiplier)
    min_overdue = float(min_days_overdue)

    for sid, days in by_subject.items():
        if len(days) < 2:
            continue
        gaps = [(days[i] - days[i - 1]).days for i in range(1, len(days))]
        gaps = [g for g in gaps if g > 0]
        if not gaps:
            continue
        avg_cadence = sum(gaps) / len(gaps)
        last_visit = days[-1]
        days_overdue = (as_of_day - last_visit).days
        threshold = max(avg_cadence * mult, min_overdue)
        if days_overdue <= threshold:
            continue

        meta = get_visitor_display_name(sid, db_path=path)
        risks.append(
            {
                "visitor_id": sid,
                "display_name": meta.get("display_name") or sid,
                "tier": meta.get("tier") or "Standard",
                "is_named": bool(meta.get("is_named")),
                "notes": meta.get("notes") or "",
                "avg_cadence_days": round(avg_cadence, 1),
                "days_overdue": int(days_overdue),
                "last_visit": last_visit.isoformat(),
                "visit_days": len(days),
            }
        )

    risks.sort(key=lambda r: int(r["days_overdue"]), reverse=True)
    return risks


def format_churn_watchlist(
    risks: list[dict[str, Any]],
    branch_id: str = "champei-pp-01",
) -> str:
    """Format owner-facing silent churn Telegram text."""
    branch = str(branch_id or "champei-pp-01").strip() or "champei-pp-01"
    header = f"📉 [{branch}] SILENT CHURN WATCHLIST"
    if not risks:
        return f"{header}\nNo at-risk guests detected this cycle."

    lines = [header]
    for risk in risks[:15]:
        name = str(risk.get("display_name") or risk.get("visitor_id") or "guest")
        overdue = int(risk.get("days_overdue") or 0)
        cadence = risk.get("avg_cadence_days")
        try:
            cadence_txt = f"{float(cadence):g}"
        except (TypeError, ValueError):
            cadence_txt = str(cadence)
        lines.append(f"• {name} — overdue {overdue}d (avg cadence {cadence_txt}d)")
    return "\n".join(lines)


def send_weekly_churn(
    db_path: Path | str | None = None,
    branch_id: str = "champei-pp-01",
    *,
    threshold_multiplier: float = 2.0,
    min_days_overdue: int = 21,
    as_of: date | str | None = None,
    telegram: TelegramOut | None = None,
) -> bool:
    """Compute, format, and dispatch silent churn watchlist to owner chat."""
    risks = compute_churn_risks(
        db_path=db_path,
        threshold_multiplier=threshold_multiplier,
        min_days_overdue=min_days_overdue,
        as_of=as_of,
    )
    text = format_churn_watchlist(risks, branch_id=branch_id)
    bot = telegram if telegram is not None else TelegramOut()
    return bool(bot.send_alert("silent_churn", text))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Champei silent churn watchlist")
    parser.add_argument("--dry-run", action="store_true", help="Print text; do not send Telegram")
    parser.add_argument("--branch", default="champei-pp-01", help="Branch id for the header")
    parser.add_argument("--db", default="", help="Optional path to events.db")
    parser.add_argument("--as-of", default="", help="Optional YYYY-MM-DD (default: today)")
    parser.add_argument("--threshold-multiplier", type=float, default=2.0)
    parser.add_argument("--min-days-overdue", type=int, default=21)
    args = parser.parse_args(argv)

    db_path: Path | None = Path(args.db).expanduser() if args.db else None
    as_of = args.as_of.strip() or None
    risks = compute_churn_risks(
        db_path=db_path,
        threshold_multiplier=args.threshold_multiplier,
        min_days_overdue=args.min_days_overdue,
        as_of=as_of,
    )
    text = format_churn_watchlist(risks, branch_id=args.branch)
    if args.dry_run:
        print(text)
        return 0
    ok = send_weekly_churn(
        db_path=db_path,
        branch_id=args.branch,
        threshold_multiplier=args.threshold_multiplier,
        min_days_overdue=args.min_days_overdue,
        as_of=as_of,
    )
    print(text)
    print("[churn] sent" if ok else "[churn] send skipped/failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
