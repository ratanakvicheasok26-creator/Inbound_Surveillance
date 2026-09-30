"""Massage-domain weekly customer activity brief for owner Telegram.

Summarizes a fixed 7-day window (epoch-aligned, default 2026-09-25), highlights
the most frequent anonymous visitor IDs, busiest day / hour, and optionally
asks a local Ollama model for a short owner-facing narrative.

Runs off the video loop — callers must use a background thread / CLI only.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import requests

from paths import data_dir
from telegram_out import TelegramOut

DEFAULT_EPOCH = date(2026, 9, 25)
DEFAULT_PERIOD_DAYS = 7
DEFAULT_LOOKBACK_DAYS = 90
STAFF_SUBJECT_PREFIX = "staff_"
TELEGRAM_CHUNK_LIMIT = 3500
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

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


def parse_epoch(value: Any, default: date = DEFAULT_EPOCH) -> date:
    parsed = _parse_day(value) if value not in (None, "") else None
    return parsed or default


def period_window(
    as_of: date | str | datetime | None = None,
    *,
    epoch: date | str | None = None,
    period_days: int = DEFAULT_PERIOD_DAYS,
) -> tuple[date, date, int] | None:
    """Return (start, end, index) for the 7-day window containing ``as_of``.

    Windows are contiguous from ``epoch`` (default 2026-09-25):
    index 0 = epoch .. epoch+6, index 1 = epoch+7 .. epoch+13, etc.
    Returns None if ``as_of`` is before the epoch.
    """
    report_day = _resolve_day(as_of)
    start_epoch = parse_epoch(epoch)
    length = max(1, int(period_days))
    if report_day < start_epoch:
        return None
    idx = (report_day - start_epoch).days // length
    start = start_epoch + timedelta(days=idx * length)
    end = start + timedelta(days=length - 1)
    return start, end, idx


def period_key(start: date, end: date) -> str:
    return f"{start.isoformat()}_{end.isoformat()}"


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
    """Resolve aliases from anonymous_subjects (massage privacy gallery)."""
    names: dict[str, str] = {}
    ids = sorted({cid for cid in customer_ids if cid})
    if not ids:
        return names
    cols = _column_names(conn, "anonymous_subjects")
    if "id" not in cols or "alias" not in cols:
        return names
    for offset in range(0, len(ids), 400):
        chunk = ids[offset : offset + 400]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT id AS cid, alias FROM anonymous_subjects WHERE id IN ({placeholders})",
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
    venue: str = "Massage",
    as_of: date | str | datetime | None = None,
    *,
    epoch: date | str | None = None,
    period_days: int = DEFAULT_PERIOD_DAYS,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> dict[str, Any]:
    """Summarize who came, how often, when, and which day was busiest."""
    report_day = _resolve_day(as_of)
    window = period_window(report_day, epoch=epoch, period_days=period_days)
    if window is None:
        start_epoch = parse_epoch(epoch)
        return {
            "venue": venue,
            "generated_for": report_day.isoformat(),
            "week_start": None,
            "week_end": None,
            "period_index": None,
            "period_key": None,
            "before_epoch": True,
            "epoch": start_epoch.isoformat(),
            "total_visits": 0,
            "unique_customers": 0,
            "busiest_days": [],
            "top_customers": [],
            "customers": [],
            "ai_summary": "",
        }

    week_start, week_end, period_index = window
    # Cap the reporting end at as_of so in-progress periods stay honest.
    effective_end = min(week_end, report_day)
    lookback = max(1, int(lookback_days))
    history_start = week_start - timedelta(days=lookback)

    conn = _connect(db_path)
    try:
        history = _load_canonical_visits(conn, history_start, effective_end)
        names = _load_display_names(conn, [v["customer_id"] for v in history])
    finally:
        conn.close()

    history_by_customer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for visit in history:
        history_by_customer[str(visit["customer_id"])].append(visit)

    week_by_customer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for visit in history:
        if week_start <= visit["visit_date"] <= effective_end:
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

    top_week = max((int(c["week_visits"]) for c in customers), default=0)
    top_customers = [c for c in customers if int(c["week_visits"]) == top_week] if customers else []

    return {
        "venue": str(venue or "Massage").strip() or "Massage",
        "generated_for": report_day.isoformat(),
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "effective_end": effective_end.isoformat(),
        "period_index": period_index,
        "period_key": period_key(week_start, week_end),
        "before_epoch": False,
        "epoch": parse_epoch(epoch).isoformat(),
        "period_days": max(1, int(period_days)),
        "history_start": history_start.isoformat(),
        "history_end": effective_end.isoformat(),
        "lookback_days": lookback,
        "total_visits": sum(int(c["week_visits"]) for c in customers),
        "unique_customers": len(customers),
        "busiest_days": busiest,
        "top_customers": top_customers,
        "customers": customers,
        "ai_summary": "",
    }


def format_customer_activity_brief(brief: dict[str, Any]) -> str:
    """Format the owner-facing weekly customer activity text."""
    venue = str(brief.get("venue") or "Massage").strip() or "Massage"
    lookback = int(brief.get("lookback_days") or DEFAULT_LOOKBACK_DAYS)

    if brief.get("before_epoch"):
        return (
            f"[{venue}] WEEKLY CUSTOMER ACTIVITY BRIEF\n"
            f"Reporting starts on {brief.get('epoch')}. No period is ready yet."
        )

    lines = [
        f"[{venue}] WEEKLY CUSTOMER ACTIVITY BRIEF",
        f"Period: {brief.get('week_start')} to {brief.get('week_end')}",
        f"- Customer visits this period: {int(brief.get('total_visits') or 0)}",
        f"- Unique customers: {int(brief.get('unique_customers') or 0)}",
    ]

    top_customers = brief.get("top_customers") or []
    if top_customers:
        text = " / ".join(
            f"{c['display_name']} [{c['customer_id']}] ({int(c['week_visits'])} visits)"
            for c in top_customers
        )
        lines.append(f"- Most frequent visitor this period: {text}")

    busiest_days = brief.get("busiest_days") or []
    if busiest_days:
        text = " / ".join(
            f"{d['weekday']} {d['date']} ({int(d['visits'])} visits)"
            for d in busiest_days
        )
        lines.append(f"- Busiest day: {text}")

    ai_summary = str(brief.get("ai_summary") or "").strip()
    if ai_summary:
        lines.append("")
        lines.append("AI summary:")
        lines.append(ai_summary)

    customers = brief.get("customers") or []
    if not customers:
        lines.append("")
        lines.append("No customer visits recorded in this period.")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"Usual visit times (based on the last {lookback} days):")
    for customer in customers[:25]:
        line = (
            f"- {customer['display_name']} [{customer['customer_id']}] "
            f"- {customer['usual_time']}"
        )
        if customer.get("usual_share"):
            line += f" ({customer['usual_share']} visit days)"
        lines.append(f"{line} - {int(customer.get('week_visits') or 0)} visit(s) this period")
    if len(customers) > 25:
        lines.append(f"- …and {len(customers) - 25} more unique visitors")
    return "\n".join(lines)


def _facts_for_ai(brief: dict[str, Any]) -> str:
    top = brief.get("top_customers") or []
    busiest = brief.get("busiest_days") or []
    customers = brief.get("customers") or []
    top_lines = [
        f"{c.get('display_name')} id={c.get('customer_id')} visits={c.get('week_visits')} usual={c.get('usual_time')}"
        for c in top[:5]
    ]
    busy_lines = [
        f"{d.get('weekday')} {d.get('date')} visits={d.get('visits')}" for d in busiest[:5]
    ]
    sample = [
        f"{c.get('display_name')} id={c.get('customer_id')} visits={c.get('week_visits')} usual={c.get('usual_time')}"
        for c in customers[:12]
    ]
    return (
        f"Venue: {brief.get('venue')}\n"
        f"Period: {brief.get('week_start')} to {brief.get('week_end')}\n"
        f"Total visits: {brief.get('total_visits')}\n"
        f"Unique customers: {brief.get('unique_customers')}\n"
        f"Most frequent: {'; '.join(top_lines) or 'none'}\n"
        f"Busiest days: {'; '.join(busy_lines) or 'none'}\n"
        f"Sample visitors: {'; '.join(sample) or 'none'}\n"
    )


def generate_ai_summary(
    brief: dict[str, Any],
    *,
    enabled: bool = True,
    model: str | None = None,
    host: str | None = None,
    timeout: float = 60.0,
    request_fn: Callable[..., Any] | None = None,
) -> str:
    """Ask local Ollama for a short owner narrative; empty string on failure."""
    if not enabled or brief.get("before_epoch"):
        return ""
    if int(brief.get("unique_customers") or 0) == 0 and int(brief.get("total_visits") or 0) == 0:
        return "No visitor activity was recorded in this 7-day period."

    llm_model = (model or DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL
    llm_host = (host or DEFAULT_OLLAMA_HOST).rstrip("/")
    prompt = (
        "You are an operations analyst for a massage / wellness venue. "
        "Using ONLY the facts below, write 3-5 short sentences for the owner. "
        "Mention total visits, unique customers, the most frequent visitor unique ID(s), "
        "and the busiest day or usual visit times when available. "
        "Do not invent names, counts, or IDs. Plain text only.\n\n"
        f"{_facts_for_ai(brief)}"
    )
    payload = {
        "model": llm_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    post = request_fn or requests.post
    try:
        response = post(
            f"{llm_host}/api/generate",
            json=payload,
            timeout=timeout,
        )
        if not getattr(response, "ok", False):
            return ""
        data = response.json() if hasattr(response, "json") else {}
        text = str(data.get("response") or "").strip()
        return text[:2000]
    except Exception as exc:
        print(f"[weekly-brief] AI summary skipped: {exc}", flush=True)
        return ""


def _split_message(text: str, limit: int = TELEGRAM_CHUNK_LIMIT) -> list[str]:
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
    venue: str = "Massage",
    as_of: date | str | datetime | None = None,
    *,
    epoch: date | str | None = None,
    period_days: int = DEFAULT_PERIOD_DAYS,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ai_summary: bool = True,
    ollama_model: str | None = None,
    ollama_host: str | None = None,
) -> str:
    """Compute, optionally AI-summarize, and format the weekly brief."""
    brief = compute_customer_activity_brief(
        db_path=db_path,
        venue=venue,
        as_of=as_of,
        epoch=epoch,
        period_days=period_days,
        lookback_days=lookback_days,
    )
    brief["ai_summary"] = generate_ai_summary(
        brief,
        enabled=ai_summary,
        model=ollama_model,
        host=ollama_host,
    )
    return format_customer_activity_brief(brief)


def send_weekly_customer_brief(
    db_path: Path | str | None = None,
    venue: str = "Massage",
    as_of: date | str | datetime | None = None,
    *,
    epoch: date | str | None = None,
    period_days: int = DEFAULT_PERIOD_DAYS,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ai_summary: bool = True,
    ollama_model: str | None = None,
    ollama_host: str | None = None,
    telegram: TelegramOut | None = None,
    token: str = "",
    chat_id: str = "",
) -> bool:
    """Build the brief and dispatch it to the linked owner Telegram chat."""
    text = build_weekly_customer_brief(
        db_path=db_path,
        venue=venue,
        as_of=as_of,
        epoch=epoch,
        period_days=period_days,
        lookback_days=lookback_days,
        ai_summary=ai_summary,
        ollama_model=ollama_model,
        ollama_host=ollama_host,
    )
    chunks = _split_message(text)
    if not chunks:
        return False
    bot = telegram if telegram is not None else TelegramOut(token, chat_id)
    results = [bool(bot.send_message(chunk)) for chunk in chunks]
    return all(results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Massage weekly customer activity brief")
    parser.add_argument("--dry-run", action="store_true", help="Print text; do not send Telegram")
    parser.add_argument("--venue", default="Massage", help="Venue name for the header")
    parser.add_argument("--db", default="", help="Optional path to events.db")
    parser.add_argument("--as-of", default="", help="Optional YYYY-MM-DD (default: today)")
    parser.add_argument("--epoch", default=DEFAULT_EPOCH.isoformat(), help="Period epoch YYYY-MM-DD")
    parser.add_argument("--period-days", type=int, default=DEFAULT_PERIOD_DAYS)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--no-ai", action="store_true", help="Skip Ollama narrative")
    parser.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    parser.add_argument("--chat-id", default=os.environ.get("TELEGRAM_CHAT_ID", ""))
    args = parser.parse_args(argv)

    db_path: Path | None = Path(args.db).expanduser() if args.db else None
    as_of = args.as_of.strip() or None
    text = build_weekly_customer_brief(
        db_path=db_path,
        venue=args.venue,
        as_of=as_of,
        epoch=args.epoch,
        period_days=args.period_days,
        lookback_days=args.lookback_days,
        ai_summary=not args.no_ai,
    )
    print(text)
    if args.dry_run:
        return 0
    ok = send_weekly_customer_brief(
        db_path=db_path,
        venue=args.venue,
        as_of=as_of,
        epoch=args.epoch,
        period_days=args.period_days,
        lookback_days=args.lookback_days,
        ai_summary=not args.no_ai,
        token=args.token,
        chat_id=args.chat_id,
    )
    print("[weekly-brief] sent" if ok else "[weekly-brief] send skipped/failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
