"""Weekly customer executive business report for Champei owner Telegram.

Calculates the 5-section executive intelligence brief:
1. Weekly Performance Snapshot
2. Customer Composition & Retention
3. Top Frequent & High-Value Guests (by Customer ID)
4. Day-by-Day Traffic Breakdown (Mon-Sun visual bars)
5. Retention & Silent Churn Alert
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

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _default_db_path() -> Path:
    return data_dir() / "events.db"


def _connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    from analytics.sessions import ensure_session_columns
    path = Path(db_path) if db_path is not None else _default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    if _table_exists(conn, "customer_visits"):
        ensure_session_columns(conn)
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
    week_start = as_of - timedelta(days=as_of.weekday())
    return week_start, week_start + timedelta(days=6)


def _make_bar(value: int, max_val: int, length: int = 10) -> str:
    if max_val <= 0:
        return "░" * length
    filled = int(round((value / max_val) * length))
    filled = max(0, min(length, filled))
    return "█" * filled + "░" * (length - filled)


def compute_weekly_executive_brief(
    db_path: Path | str | None = None,
    branch_id: str = DEFAULT_BRANCH,
    as_of: date | str | datetime | None = None,
) -> dict[str, Any]:
    branch = str(branch_id or DEFAULT_BRANCH).strip() or DEFAULT_BRANCH
    report_day = _resolve_day(as_of)
    week_start, week_end = week_window(report_day)
    prior_week_start = week_start - timedelta(days=7)
    prior_week_end = week_start - timedelta(days=1)

    conn = _connect(db_path)
    try:
        cols = _column_names(conn, "customer_visits")
        has_visits = _table_exists(conn, "customer_visits") and "started_at" in cols
        rows_this_week = []
        rows_prior_week = []
        rows_lifetime = []

        if has_visits:
            rows_this_week = conn.execute(
                """
                SELECT subject_id, started_at, ended_at, duration_seconds, status
                FROM customer_visits
                WHERE started_at >= ? AND started_at <= ?
                """,
                (f"{week_start.isoformat()}T00:00:00", f"{week_end.isoformat()}T23:59:59"),
            ).fetchall()

            rows_prior_week = conn.execute(
                """
                SELECT subject_id, started_at, ended_at, duration_seconds, status
                FROM customer_visits
                WHERE started_at >= ? AND started_at <= ?
                """,
                (f"{prior_week_start.isoformat()}T00:00:00", f"{prior_week_end.isoformat()}T23:59:59"),
            ).fetchall()

            rows_lifetime = conn.execute(
                "SELECT subject_id, started_at FROM customer_visits WHERE started_at IS NOT NULL"
            ).fetchall()

        # Load names / aliases from visitor_meta
        names_map: dict[str, str] = {}
        if _table_exists(conn, "visitor_meta"):
            meta_cols = {col["name"] for col in conn.execute("PRAGMA table_info(visitor_meta)").fetchall()}
            vid_col = "visitor_id" if "visitor_id" in meta_cols else "subject_id"
            if vid_col in meta_cols and "alias" in meta_cols:
                meta_rows = conn.execute(f"SELECT {vid_col} AS vid, alias FROM visitor_meta").fetchall()
                for r in meta_rows:
                    vid = str(r["vid"] or "").strip()
                    alias = str(r["alias"] or "").strip()
                    if vid and alias:
                        names_map[vid] = alias
    finally:
        conn.close()

    # Calculate metrics
    total_visits = len(rows_this_week)
    prior_visits = len(rows_prior_week)
    growth_pct = round(((total_visits - prior_visits) / max(1, prior_visits)) * 100.0, 1)

    completed_count = 0
    total_duration_sec = 0.0
    daily_counts = defaultdict(int)
    subject_week_visits = defaultdict(int)
    subject_week_duration = defaultdict(float)

    for r in rows_this_week:
        sid = str(r["subject_id"] or "").strip()
        st = r["status"]
        ended = r["ended_at"]
        dur = float(r["duration_seconds"] or 0.0) if "duration_seconds" in r.keys() else 0.0

        if st == "completed" or ended is not None:
            completed_count += 1

        if dur <= 0 and ended and r["started_at"]:
            try:
                t0 = datetime.fromisoformat(str(r["started_at"]).replace("Z", ""))
                t1 = datetime.fromisoformat(str(ended).replace("Z", ""))
                dur = max(0.0, (t1 - t0).total_seconds())
            except Exception:
                dur = 0.0

        if dur > 0:
            total_duration_sec += dur
            subject_week_duration[sid] += dur

        subject_week_visits[sid] += 1

        day = _parse_day(r["started_at"])
        if day:
            daily_counts[day.isoformat()] += 1

    completion_rate = round((completed_count / max(1, total_visits)) * 100.0, 1)
    unique_guests = len(subject_week_visits)
    total_hours = round(total_duration_sec / 3600.0, 1)
    avg_duration_min = round((total_duration_sec / max(1, completed_count)) / 60.0, 1) if completed_count else 0.0

    # Lifetime visits per subject
    lifetime_counts = defaultdict(int)
    lifetime_days = defaultdict(set)
    for r in rows_lifetime:
        sid = str(r["subject_id"] or "").strip()
        lifetime_counts[sid] += 1
        day = _parse_day(r["started_at"])
        if day:
            lifetime_days[sid].add(day)

    # Regular vs New vs Returning
    regular_count = 0
    returning_count = 0
    new_count = 0

    for sid in subject_week_visits:
        tot_days = len(lifetime_days.get(sid, set()))
        if tot_days >= 3:
            regular_count += 1
        elif tot_days >= 2:
            returning_count += 1
        else:
            new_count += 1

    retention_rate = round(((regular_count + returning_count) / max(1, unique_guests)) * 100.0, 1)

    # Top guests leaderboard
    sorted_guests = sorted(
        subject_week_visits.items(),
        key=lambda x: (x[1], subject_week_duration[x[0]]),
        reverse=True,
    )
    top_guests = []
    for sid, v_count in sorted_guests[:5]:
        hours_spent = round(subject_week_duration[sid] / 3600.0, 1)
        lifetimes = lifetime_counts.get(sid, v_count)
        disp_name = names_map.get(sid, sid)
        top_guests.append({
            "id": sid,
            "display_name": disp_name,
            "week_visits": v_count,
            "hours": hours_spent,
            "lifetime_visits": lifetimes,
        })

    # Day-by-day table
    day_breakdown = []
    max_day_visits = max(daily_counts.values()) if daily_counts else 1
    weekend_visits = 0

    for i in range(7):
        cur_day = week_start + timedelta(days=i)
        iso = cur_day.isoformat()
        count = daily_counts.get(iso, 0)
        is_weekend = cur_day.weekday() >= 4  # Fri, Sat, Sun
        if is_weekend:
            weekend_visits += count

        day_breakdown.append({
            "weekday": WEEKDAYS[cur_day.weekday()],
            "date": cur_day.strftime("%m/%d"),
            "count": count,
            "bar": _make_bar(count, max_day_visits, length=10),
            "is_peak": count == max_day_visits and count > 0,
        })

    weekend_share = round((weekend_visits / max(1, total_visits)) * 100.0, 1)

    # Churn / overdue regulars (>14 days absent)
    churn_risks = []
    for sid, d_set in lifetime_days.items():
        if len(d_set) >= 3 and sid not in subject_week_visits:
            last_v = max(d_set)
            days_absent = (report_day - last_v).days
            if days_absent >= 14:
                disp_name = names_map.get(sid, sid)
                churn_risks.append({
                    "id": sid,
                    "display_name": disp_name,
                    "days_absent": days_absent,
                })
    churn_risks.sort(key=lambda x: x["days_absent"], reverse=True)

    return {
        "branch_id": branch,
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "total_visits": total_visits,
        "growth_pct": growth_pct,
        "completed_treatments": completed_count,
        "completion_rate": completion_rate,
        "unique_guests": unique_guests,
        "total_hours": total_hours,
        "avg_duration_min": avg_duration_min,
        "regular_count": regular_count,
        "returning_count": returning_count,
        "new_count": new_count,
        "retention_rate": retention_rate,
        "top_guests": top_guests,
        "day_breakdown": day_breakdown,
        "weekend_share": weekend_share,
        "churn_risks": churn_risks[:3],
    }


def format_weekly_executive_brief(data: dict[str, Any]) -> str:
    branch = data.get("branch_id", DEFAULT_BRANCH)
    w_start = data.get("week_start", "")
    w_end = data.get("week_end", "")

    growth_sign = "+" if data["growth_pct"] >= 0 else ""
    growth_txt = f"({growth_sign}{data['growth_pct']}% vs last week)"

    # Top guests formatting
    medals = ["🥇 1.", "🥈 2.", "🥉 3.", "🏅 4.", "🏅 5."]
    top_lines = []
    for i, g in enumerate(data.get("top_guests", [])):
        prefix = medals[i] if i < len(medals) else f"• {i+1}."
        top_lines.append(
            f"{prefix} {g['display_name']}\n"
            f"   └ {g['week_visits']} visits this week | {g['hours']} hrs total | {g['lifetime_visits']} lifetime visits"
        )
    top_text = "\n".join(top_lines) if top_lines else "• No customer visits recorded this week."

    # Day breakdown formatting
    day_lines = []
    for d in data.get("day_breakdown", []):
        tag = "  🏆 Peak Day" if d["is_peak"] else ("  🔥 Busy" if d["count"] >= 25 else "")
        day_lines.append(f"• {d['weekday']} ({d['date']}): {d['count']:02d} visits  [{d['bar']}]{tag}")
    day_text = "\n".join(day_lines)

    # Churn formatting
    churn_lines = []
    for c in data.get("churn_risks", []):
        churn_lines.append(f"• {c['display_name']} — Absent {c['days_absent']} days (Prior regular)")
    if not churn_lines:
        churn_text = "• No high-risk customer churn detected."
    else:
        churn_text = "\n".join(churn_lines) + "\n💡 Suggested Action: Send reminder, voucher, or check-in."

    return (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🌿 [{branch}] WEEKLY EXECUTIVE BUSINESS REPORT\n"
        f"📅 Week: {w_start} to {w_end} (Mon – Sun)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📈 1. WEEKLY PERFORMANCE SNAPSHOT\n"
        f"• Total Customer Visits: {data['total_visits']} {growth_txt}\n"
        f"• Completed Treatments: {data['completed_treatments']} ({data['completion_rate']}% completion rate)\n"
        f"• Unique Individuals: {data['unique_guests']} guests\n"
        f"• Total Treatment Time: {data['total_hours']} hours delivered\n"
        f"• Avg Treatment Duration: {data['avg_duration_min']:.0f} mins / guest\n\n"
        f"👥 2. CUSTOMER COMPOSITION & RETENTION\n"
        f"• 👑 Regular Guests: {data['regular_count']} guests ({data['regular_count']/max(1, data['unique_guests'])*100:.1f}%)\n"
        f"• 🔄 Returning Guests: {data['returning_count']} guests ({data['returning_count']/max(1, data['unique_guests'])*100:.1f}%)\n"
        f"• 🆕 First-Time Clients: {data['new_count']} guests ({data['new_count']/max(1, data['unique_guests'])*100:.1f}%)\n"
        f"• 💎 Client Retention Rate: {data['retention_rate']}%\n\n"
        f"🏆 3. TOP FREQUENT & HIGH-VALUE GUESTS\n"
        f"{top_text}\n\n"
        f"📅 4. DAY-BY-DAY TRAFFIC BREAKDOWN\n"
        f"{day_text}\n"
        f"└ Weekend Share (Fri–Sun): {data['weekend_share']}% of total weekly volume\n\n"
        f"⚠️ 5. RETENTION & SILENT CHURN ALERT\n"
        f"{churn_text}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


def build_weekly_customer_brief(
    db_path: Path | str | None = None,
    branch_id: str = DEFAULT_BRANCH,
    as_of: date | str | datetime | None = None,
) -> str:
    return format_weekly_executive_brief(
        compute_weekly_executive_brief(
            db_path=db_path,
            branch_id=branch_id,
            as_of=as_of,
        )
    )


def send_weekly_customer_brief(
    db_path: Path | str | None = None,
    branch_id: str = DEFAULT_BRANCH,
    as_of: date | str | datetime | None = None,
    *,
    telegram: TelegramOut | None = None,
) -> bool:
    text = build_weekly_customer_brief(
        db_path=db_path,
        branch_id=branch_id,
        as_of=as_of,
    )
    bot = telegram if telegram is not None else TelegramOut()
    return bool(bot.send_alert(EVENT_TYPE, text))


# Backward compatibility aliases
compute_customer_activity_brief = compute_weekly_executive_brief
format_customer_activity_brief = format_weekly_executive_brief


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Champei weekly customer executive report")
    parser.add_argument("--dry-run", action="store_true", help="Print text; do not send Telegram")
    parser.add_argument("--branch", default=DEFAULT_BRANCH, help="Branch ID")
    parser.add_argument("--db", default="", help="Optional path to events.db")
    parser.add_argument("--as-of", default="", help="Optional YYYY-MM-DD")
    args = parser.parse_args(argv)

    db_path = Path(args.db).expanduser() if args.db else None
    as_of = args.as_of.strip() or None
    text = build_weekly_customer_brief(db_path=db_path, branch_id=args.branch, as_of=as_of)
    print(text)

    if not args.dry_run:
        ok = send_weekly_customer_brief(db_path=db_path, branch_id=args.branch, as_of=as_of)
        print("[weekly-brief] sent:", ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
