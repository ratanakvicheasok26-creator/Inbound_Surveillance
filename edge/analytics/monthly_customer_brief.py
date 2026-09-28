"""Monthly strategic customer intelligence report for Champei owner Telegram.

Aggregates the 4-5 weekly cycles of a calendar month:
1. Monthly Overview & MoM Growth
2. Week-by-Week Progression Matrix (with visual bars)
3. Top 5 Client Champions of the Month (by Customer ID & hours)
4. Monthly Audience Retention & Expansion
5. Monthly Churn & At-Risk Inactive Accounts
"""

from __future__ import annotations

import argparse
import calendar
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from paths import data_dir
from telegram_out import TelegramOut

EVENT_TYPE = "monthly_customer_brief"
DEFAULT_BRANCH = "champei-pp-01"


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


def _resolve_day(value: date | str | datetime | None) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = _parse_day(value) if value else None
    return parsed or date.today()


def month_window(as_of: date) -> tuple[date, date]:
    month_start = as_of.replace(day=1)
    last_day = calendar.monthrange(as_of.year, as_of.month)[1]
    month_end = as_of.replace(day=last_day)
    return month_start, month_end


def prior_month_window(as_of: date) -> tuple[date, date]:
    month_start = as_of.replace(day=1)
    prior_end = month_start - timedelta(days=1)
    prior_start = prior_end.replace(day=1)
    return prior_start, prior_end


def _make_bar(value: int, max_val: int, length: int = 10) -> str:
    if max_val <= 0:
        return "░" * length
    filled = int(round((value / max_val) * length))
    filled = max(0, min(length, filled))
    return "█" * filled + "░" * (length - filled)


def compute_monthly_strategic_brief(
    db_path: Path | str | None = None,
    branch_id: str = DEFAULT_BRANCH,
    as_of: date | str | datetime | None = None,
) -> dict[str, Any]:
    branch = str(branch_id or DEFAULT_BRANCH).strip() or DEFAULT_BRANCH
    report_day = _resolve_day(as_of)
    m_start, m_end = month_window(report_day)
    p_start, p_end = prior_month_window(report_day)

    conn = _connect(db_path)
    try:
        cols = _column_names(conn, "customer_visits")
        has_visits = _table_exists(conn, "customer_visits") and "started_at" in cols
        rows_this_month = []
        rows_prior_month = []
        rows_history = []

        if has_visits:
            rows_this_month = conn.execute(
                """
                SELECT subject_id, started_at, ended_at, duration_seconds, status
                FROM customer_visits
                WHERE started_at >= ? AND started_at <= ?
                """,
                (f"{m_start.isoformat()}T00:00:00", f"{m_end.isoformat()}T23:59:59"),
            ).fetchall()

            rows_prior_month = conn.execute(
                """
                SELECT subject_id, started_at, ended_at, duration_seconds, status
                FROM customer_visits
                WHERE started_at >= ? AND started_at <= ?
                """,
                (f"{p_start.isoformat()}T00:00:00", f"{p_end.isoformat()}T23:59:59"),
            ).fetchall()

            rows_history = conn.execute(
                "SELECT subject_id, started_at FROM customer_visits WHERE started_at < ?",
                (f"{m_start.isoformat()}T00:00:00",),
            ).fetchall()

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

    total_visits = len(rows_this_month)
    prior_visits = len(rows_prior_month)
    growth_pct = round(((total_visits - prior_visits) / max(1, prior_visits)) * 100.0, 1)

    completed_count = 0
    total_duration_sec = 0.0
    subject_month_visits = defaultdict(int)
    subject_month_duration = defaultdict(float)

    # Week-by-week buckets (7-day intervals: 1-7, 8-14, 15-21, 22-28, 29-end)
    week_buckets = [
        {"name": "Week 1", "start": m_start, "end": m_start + timedelta(days=6), "count": 0},
        {"name": "Week 2", "start": m_start + timedelta(days=7), "end": m_start + timedelta(days=13), "count": 0},
        {"name": "Week 3", "start": m_start + timedelta(days=14), "end": m_start + timedelta(days=20), "count": 0},
        {"name": "Week 4", "start": m_start + timedelta(days=21), "end": m_start + timedelta(days=27), "count": 0},
        {"name": "Week 5", "start": m_start + timedelta(days=28), "end": m_end, "count": 0},
    ]

    for r in rows_this_month:
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
            subject_month_duration[sid] += dur

        subject_month_visits[sid] += 1

        v_day = _parse_day(r["started_at"])
        if v_day:
            for wb in week_buckets:
                if wb["start"] <= v_day <= wb["end"]:
                    wb["count"] += 1
                    break

    completion_rate = round((completed_count / max(1, total_visits)) * 100.0, 1)
    unique_guests = len(subject_month_visits)
    total_hours = round(total_duration_sec / 3600.0, 1)
    avg_freq = round(total_visits / max(1, unique_guests), 2)

    # Week progression bars
    max_week_visits = max((wb["count"] for wb in week_buckets), default=1)
    best_week = max(week_buckets, key=lambda x: x["count"]) if week_buckets else None

    for wb in week_buckets:
        wb["bar"] = _make_bar(wb["count"], max_week_visits, length=10)
        wb["span_str"] = f"{wb['start'].strftime('%m/%d')} - {wb['end'].strftime('%m/%d')}"

    # Top 5 client champions
    sorted_champions = sorted(
        subject_month_visits.items(),
        key=lambda x: (x[1], subject_month_duration[x[0]]),
        reverse=True,
    )
    champions = []
    for sid, v_count in sorted_champions[:5]:
        hours_spent = round(subject_month_duration[sid] / 3600.0, 1)
        disp_name = names_map.get(sid, sid)
        champions.append({
            "id": sid,
            "display_name": disp_name,
            "visits": v_count,
            "hours": hours_spent,
        })

    # History / Prior clients for New vs Returning
    prior_subjects = {str(r["subject_id"] or "").strip() for r in rows_history if str(r["subject_id"] or "").strip()}
    new_guests = 0
    returning_guests = 0

    for sid in subject_month_visits:
        if sid in prior_subjects:
            returning_guests += 1
        else:
            new_guests += 1

    retention_rate = round((returning_guests / max(1, unique_guests)) * 100.0, 1)

    # Churn: active last month but 0 visits this month
    prior_active_subjects = defaultdict(int)
    for r in rows_prior_month:
        sid = str(r["subject_id"] or "").strip()
        if sid:
            prior_active_subjects[sid] += 1

    churn_accounts = []
    for sid, p_count in prior_active_subjects.items():
        if p_count >= 3 and sid not in subject_month_visits:
            disp_name = names_map.get(sid, sid)
            churn_accounts.append({
                "id": sid,
                "display_name": disp_name,
                "prior_month_visits": p_count,
            })
    churn_accounts.sort(key=lambda x: x["prior_month_visits"], reverse=True)

    return {
        "branch_id": branch,
        "month_name": m_start.strftime("%B %Y"),
        "month_start": m_start.isoformat(),
        "month_end": m_end.isoformat(),
        "prior_month_name": p_start.strftime("%B"),
        "total_visits": total_visits,
        "growth_pct": growth_pct,
        "completed_treatments": completed_count,
        "completion_rate": completion_rate,
        "unique_guests": unique_guests,
        "total_hours": total_hours,
        "avg_freq": avg_freq,
        "week_buckets": week_buckets,
        "best_week": best_week,
        "champions": champions,
        "new_guests": new_guests,
        "returning_guests": returning_guests,
        "retention_rate": retention_rate,
        "churn_accounts": churn_accounts[:3],
    }


def format_monthly_strategic_brief(data: dict[str, Any]) -> str:
    branch = data.get("branch_id", DEFAULT_BRANCH)
    m_name = data.get("month_name", "")
    m_start = data.get("month_start", "")
    m_end = data.get("month_end", "")
    p_name = data.get("prior_month_name", "prior month")

    growth_sign = "+" if data["growth_pct"] >= 0 else ""
    growth_txt = f"({growth_sign}{data['growth_pct']}% vs {p_name})"

    # Week progression lines
    week_lines = []
    for wb in data.get("week_buckets", []):
        days_in_bucket = (wb["end"] - wb["start"]).days + 1
        note = f" ({days_in_bucket} days)" if days_in_bucket < 7 else ""
        week_lines.append(f"• {wb['name']} ({wb['span_str']}): {wb['count']:03d} visits [{wb['bar']}]{note}")
    if data.get("best_week"):
        week_lines.append(f"└ 🏆 Best Performing Week: {data['best_week']['name']} ({data['best_week']['count']} visits)")
    week_text = "\n".join(week_lines)

    # Champions lines
    medals = ["🥇 1.", "🥈 2.", "🥉 3.", "🏅 4.", "🏅 5."]
    champ_lines = []
    for i, c in enumerate(data.get("champions", [])):
        prefix = medals[i] if i < len(medals) else f"• {i+1}."
        champ_lines.append(
            f"{prefix} {c['display_name']}\n"
            f"   └ {c['visits']} visits this month | {c['hours']} hrs total"
        )
    champ_text = "\n".join(champ_lines) if champ_lines else "• No customer visits recorded this month."

    # Churn lines
    churn_lines = []
    for ch in data.get("churn_accounts", []):
        churn_lines.append(f"• {ch['display_name']} — 0 visits in {m_name.split()[0]} (Had {ch['prior_month_visits']} visits in {p_name})")
    if not churn_lines:
        churn_text = "• No churned regular clients detected this month."
    else:
        churn_text = "\n".join(churn_lines) + "\n💡 Recommended Strategy: Re-engage with VIP special promotion."

    return (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🏛️ [{branch}] MONTHLY STRATEGIC INTELLIGENCE REPORT\n"
        f"📅 Month: {m_name} ({m_start} to {m_end})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📊 1. MONTHLY OVERVIEW & MoM GROWTH\n"
        f"• Total Monthly Visits: {data['total_visits']} {growth_txt}\n"
        f"• Completed Treatments: {data['completed_treatments']} ({data['completion_rate']}% completion rate)\n"
        f"• Total Unique Guests: {data['unique_guests']} individuals\n"
        f"• Total Spa Treatment Hours: {data['total_hours']} hours delivered\n"
        f"• Avg Monthly Visit Frequency: {data['avg_freq']} visits / guest\n\n"
        f"🗓️ 2. WEEK-BY-WEEK PROGRESSION MATRIX\n"
        f"{week_text}\n\n"
        f"👑 3. TOP 5 CLIENT CHAMPIONS OF THE MONTH\n"
        f"{champ_text}\n\n"
        f"👥 4. MONTHLY AUDIENCE RETENTION & EXPANSION\n"
        f"• New Customer Acquisition: {data['new_guests']} new guests ({data['new_guests']/max(1, data['unique_guests'])*100:.1f}%)\n"
        f"• Loyal Returning Base: {data['returning_guests']} repeat guests ({data['returning_guests']/max(1, data['unique_guests'])*100:.1f}%)\n"
        f"• Monthly Retention Rate: {data['retention_rate']}%\n\n"
        f"⚠️ 5. MONTHLY CHURN & AT-RISK ACCOUNTS\n"
        f"{churn_text}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


def build_monthly_strategic_brief(
    db_path: Path | str | None = None,
    branch_id: str = DEFAULT_BRANCH,
    as_of: date | str | datetime | None = None,
) -> str:
    return format_monthly_strategic_brief(
        compute_monthly_strategic_brief(
            db_path=db_path,
            branch_id=branch_id,
            as_of=as_of,
        )
    )


def send_monthly_strategic_brief(
    db_path: Path | str | None = None,
    branch_id: str = DEFAULT_BRANCH,
    as_of: date | str | datetime | None = None,
    *,
    telegram: TelegramOut | None = None,
) -> bool:
    text = build_monthly_strategic_brief(
        db_path=db_path,
        branch_id=branch_id,
        as_of=as_of,
    )
    bot = telegram if telegram is not None else TelegramOut()
    return bool(bot.send_alert(EVENT_TYPE, text))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Champei monthly strategic intelligence report")
    parser.add_argument("--dry-run", action="store_true", help="Print text; do not send Telegram")
    parser.add_argument("--branch", default=DEFAULT_BRANCH, help="Branch ID")
    parser.add_argument("--db", default="", help="Optional path to events.db")
    parser.add_argument("--as-of", default="", help="Optional YYYY-MM-DD")
    args = parser.parse_args(argv)

    db_path = Path(args.db).expanduser() if args.db else None
    as_of = args.as_of.strip() or None
    text = build_monthly_strategic_brief(db_path=db_path, branch_id=args.branch, as_of=as_of)
    print(text)

    if not args.dry_run:
        ok = send_monthly_strategic_brief(db_path=db_path, branch_id=args.branch, as_of=as_of)
        print("[monthly-brief] sent:", ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
