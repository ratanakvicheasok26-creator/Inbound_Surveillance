"""Weekly customer activity brief for Champei owner Telegram.

Answers the three owner questions once per week:
- when does each customer usually come to the spa,
- who comes most often,
- which weekday has the most customers.

Zone rows are collapsed into one canonical arrival per customer per local
calendar day, so a guest crossing entrance, waiting, and treatment zones is
counted once. Usual visit times use a trailing lookback window (one week is
not enough history to establish a habit).
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from paths import data_dir
from telegram_out import TelegramOut

EVENT_TYPE = "weekly_customer_brief"
DEFAULT_BRANCH = "champei-pp-01"
DEFAULT_LOOKBACK_DAYS = 90
STAFF_SUBJECT_PREFIX = "staff_"
TELEGRAM_CHUNK_LIMIT = 3500

WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


def _default_db_path() -> Path:
    return data_dir() / "events.db"


def _connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else _default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    return {str(r["name"]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _parse_day(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1]
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    day = _parse_day(text)
    return datetime.combine(day, datetime.min.time()) if day is not None else None


def _resolve_day(value: date | str | datetime | None) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = _parse_day(value) if value else None
    return parsed or date.today()


def week_window(as_of: date) -> tuple[date, date]:
    """Return the Monday..Sunday window that contains ``as_of``."""
    week_start = as_of - timedelta(days=as_of.weekday())
    return week_start, week_start + timedelta(days=6)


def _hour_label(hour: int) -> str:
    return f"{int(hour):02d}:00-{int(hour):02d}:59"


def _load_canonical_visits(
    conn: sqlite3.Connection,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    """One row per customer per local day, using the earliest zone entry."""
    cols = _column_names(conn, "customer_visits")
    if "subject_id" not in cols or "started_at" not in cols:
        return []
    rows = conn.execute(
        """
        SELECT TRIM(subject_id) AS subject_id,
               SUBSTR(started_at, 1, 10) AS visit_date,
               MIN(started_at) AS arrived_at
        FROM customer_visits
        WHERE started_at IS NOT NULL
          AND TRIM(subject_id) <> ''
          AND started_at >= ?
          AND started_at < ?
        GROUP BY TRIM(subject_id), SUBSTR(started_at, 1, 10)
        ORDER BY visit_date ASC, arrived_at ASC, subject_id ASC
        """,
        (
            f"{start.isoformat()}T00:00:00",
            f"{(end + timedelta(days=1)).isoformat()}T00:00:00",
        ),
    ).fetchall()

    visits: list[dict[str, Any]] = []
    for row in rows:
        subject_id = str(row["subject_id"] or "").strip()
        if not subject_id or subject_id.startswith(STAFF_SUBJECT_PREFIX):
            continue
        visit_date = _parse_day(row["visit_date"])
        arrived_at = _parse_dt(row["arrived_at"])
        if visit_date is None or arrived_at is None:
            continue
        visits.append(
            {
                "customer_id": subject_id,
                "visit_date": visit_date,
                "arrived_at": arrived_at,
            }
        )
    return visits


def _load_display_names(
    conn: sqlite3.Connection,
    customer_ids: list[str],
) -> dict[str, str]:
    """Resolve aliases, preferring visitor_meta over anonymous_subjects."""
    names: dict[str, str] = {}
    ids = sorted({cid for cid in customer_ids if cid})
    if not ids:
        return names
    for table, id_col in (("visitor_meta", "visitor_id"), ("anonymous_subjects", "id")):
        cols = _column_names(conn, table)
        if id_col not in cols or "alias" not in cols:
            continue
        for offset in range(0, len(ids), 400):
            chunk = ids[offset : offset + 400]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT {id_col} AS cid, alias FROM {table} "
                f"WHERE {id_col} IN ({placeholders})",
                tuple(chunk),
            ).fetchall()
            for row in rows:
                cid = str(row["cid"] or "").strip()
                alias = str(row["alias"] or "").strip()
                if cid and alias and cid not in names:
                    names[cid] = alias
    return names


def _usual_hour_buckets(visits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter = Counter(int(visit["arrived_at"].hour) for visit in visits)
    if not counter:
        return []
    best = max(counter.values())
    return [
        {"hour": hour, "count": count, "label": _hour_label(hour)}
        for hour, count in sorted(counter.items())
        if count == best
    ]


def compute_customer_activity_brief(
    db_path: Path | str | None = None,
    branch_id: str = DEFAULT_BRANCH,
    as_of: date | str | datetime | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> dict[str, Any]:
    """Summarize who comes to the spa, how often, when, and on which day."""
    branch = str(branch_id or DEFAULT_BRANCH).strip() or DEFAULT_BRANCH
    report_day = _resolve_day(as_of)
    week_start, week_end = week_window(report_day)
    week_end = min(week_end, report_day)
    lookback = max(1, int(lookback_days))
    history_start = week_start - timedelta(days=lookback)

    conn = _connect(db_path)
    try:
        history = _load_canonical_visits(conn, history_start, report_day)
        names = _load_display_names(conn, [v["customer_id"] for v in history])
    finally:
        conn.close()

    history_by_customer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for visit in history:
        history_by_customer[str(visit["customer_id"])].append(visit)

    week_by_customer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for visit in history:
        if week_start <= visit["visit_date"] <= week_end:
            week_by_customer[str(visit["customer_id"])].append(visit)

    customers: list[dict[str, Any]] = []
    for customer_id, week_visits in week_by_customer.items():
        customer_history = history_by_customer.get(customer_id, week_visits)
        buckets = _usual_hour_buckets(customer_history)
        visit_days = len({visit["visit_date"] for visit in customer_history})
        display_name = names.get(customer_id, customer_id)
        customers.append(
            {
                "customer_id": customer_id,
                "display_name": display_name,
                "is_named": customer_id in names,
                "week_visits": len(week_visits),
                "visit_days": visit_days,
                "usual_hours": buckets,
                "usual_time": " / ".join(b["label"] for b in buckets) or "unknown",
                "usual_share": f"{buckets[0]['count']} of {visit_days}" if buckets else "",
            }
        )

    customers.sort(
        key=lambda c: (
            -int(c["week_visits"]),
            -int(c["visit_days"]),
            str(c["display_name"]).lower(),
            str(c["customer_id"]),
        )
    )

    day_counter = Counter(
        visit["visit_date"] for visits in week_by_customer.values() for visit in visits
    )
    busiest: list[dict[str, Any]] = []
    if day_counter:
        best = max(day_counter.values())
        busiest = [
            {
                "date": day.isoformat(),
                "weekday": WEEKDAY_NAMES[day.weekday()],
                "visits": count,
            }
            for day, count in sorted(day_counter.items())
            if count == best
        ]

    top_days = max((int(c["visit_days"]) for c in customers), default=0)
    top_customers = [c for c in customers if int(c["visit_days"]) == top_days] if customers else []

    return {
        "branch_id": branch,
        "generated_for": report_day.isoformat(),
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "history_start": history_start.isoformat(),
        "history_end": report_day.isoformat(),
        "lookback_days": lookback,
        "total_visits": sum(int(c["week_visits"]) for c in customers),
        "unique_customers": len(customers),
        "busiest_days": busiest,
        "top_customers": top_customers,
        "customers": customers,
    }


def format_customer_activity_brief(brief: dict[str, Any]) -> str:
    """Format the owner-facing weekly customer activity text."""
    branch = str(brief.get("branch_id") or DEFAULT_BRANCH).strip() or DEFAULT_BRANCH
    lookback = int(brief.get("lookback_days") or DEFAULT_LOOKBACK_DAYS)
    lines = [
        f"[{branch}] WEEKLY CUSTOMER ACTIVITY BRIEF",
        f"Week: {brief.get('week_start')} to {brief.get('week_end')}",
        f"- Customer visits this week: {int(brief.get('total_visits') or 0)}",
        f"- Unique customers: {int(brief.get('unique_customers') or 0)}",
    ]

    top_customers = brief.get("top_customers") or []
    if top_customers:
        text = " / ".join(
            f"{c['display_name']} ({int(c['visit_days'])} visit days)"
            for c in top_customers
        )
        lines.append(f"- Most frequent customer (last {lookback}d): {text}")

    busiest_days = brief.get("busiest_days") or []
    if busiest_days:
        text = " / ".join(
            f"{d['weekday']} {d['date']} ({int(d['visits'])} visits)"
            for d in busiest_days
        )
        lines.append(f"- Busiest day: {text}")

    customers = brief.get("customers") or []
    if not customers:
        lines.append("")
        lines.append("No customer visits recorded this week.")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"Usual visit times (based on the last {lookback} days):")
    for customer in customers:
        line = f"- {customer['display_name']} - {customer['usual_time']}"
        if customer.get("usual_share"):
            line += f" ({customer['usual_share']} visit days)"
        lines.append(f"{line} - {int(customer.get('week_visits') or 0)} visit(s) this week")
    return "\n".join(lines)


def _split_message(text: str, limit: int = TELEGRAM_CHUNK_LIMIT) -> list[str]:
    """Split long text at line boundaries so every Telegram send stays valid."""
    body = str(text or "").strip()
    if not body:
        return []
    size_limit = max(200, int(limit))
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in body.splitlines():
        if len(line) > size_limit:
            if current:
                chunks.append("\n".join(current))
                current = []
                size = 0
            for start in range(0, len(line), size_limit):
                chunks.append(line[start : start + size_limit])
            continue
        extra = len(line) + 1
        if current and size + extra > size_limit:
            chunks.append("\n".join(current))
            current = []
            size = 0
        current.append(line)
        size += extra
    if current:
        chunks.append("\n".join(current))
    if len(chunks) > 1:
        total = len(chunks)
        chunks = [f"{chunk}\n(continued {i}/{total})" for i, chunk in enumerate(chunks, 1)]
    return chunks


def build_weekly_customer_brief(
    db_path: Path | str | None = None,
    branch_id: str = DEFAULT_BRANCH,
    as_of: date | str | datetime | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> str:
    """Compute and format the weekly brief without sending it."""
    return format_customer_activity_brief(
        compute_customer_activity_brief(
            db_path=db_path,
            branch_id=branch_id,
            as_of=as_of,
            lookback_days=lookback_days,
        )
    )


def send_weekly_customer_brief(
    db_path: Path | str | None = None,
    branch_id: str = DEFAULT_BRANCH,
    as_of: date | str | datetime | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    *,
    telegram: TelegramOut | None = None,
) -> bool:
    """Build the brief and dispatch it to the owner chat as weekly_customer_brief."""
    text = build_weekly_customer_brief(
        db_path=db_path,
        branch_id=branch_id,
        as_of=as_of,
        lookback_days=lookback_days,
    )
    chunks = _split_message(text)
    if not chunks:
        return False
    bot = telegram if telegram is not None else TelegramOut()
    results = [bool(bot.send_alert(EVENT_TYPE, chunk)) for chunk in chunks]
    return all(results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Champei weekly customer activity brief")
    parser.add_argument("--dry-run", action="store_true", help="Print text; do not send Telegram")
    parser.add_argument("--branch", default=DEFAULT_BRANCH, help="Branch id for the header")
    parser.add_argument("--db", default="", help="Optional path to events.db")
    parser.add_argument("--as-of", default="", help="Optional YYYY-MM-DD (default: today)")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="History window used for usual visit times",
    )
    args = parser.parse_args(argv)

    db_path: Path | None = Path(args.db).expanduser() if args.db else None
    as_of = args.as_of.strip() or None
    text = build_weekly_customer_brief(
        db_path=db_path,
        branch_id=args.branch,
        as_of=as_of,
        lookback_days=args.lookback_days,
    )
    if args.dry_run:
        print(text)
        return 0
    ok = send_weekly_customer_brief(
        db_path=db_path,
        branch_id=args.branch,
        as_of=as_of,
        lookback_days=args.lookback_days,
    )
    print(text)
    print("[weekly-brief] sent" if ok else "[weekly-brief] send skipped/failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
