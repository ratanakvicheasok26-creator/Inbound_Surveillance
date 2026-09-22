from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

from db import day_events, day_minutes, get_daily_garage_summary


def _parse_open_time(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def efficiency_badge(score: float) -> str:
    if score >= 70:
        return "🟢 High"
    if score >= 40:
        return "🟡 Normal"
    return "🔴 Low"


def _fmt_clock(value: str | None) -> str:
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(value).strftime("%H:%M")
    except ValueError:
        return value


def _fmt_hours_from_minutes(minutes: float) -> str:
    hours = max(0.0, float(minutes)) / 60.0
    return f"{hours:.1f}h"


def build_garage_report(
    conn,
    day: date,
    venue: str,
    open_time: str,
    close_time: str = "18:00",
    bay_ids: list[str] | None = None,
) -> tuple[str, list[Path]]:
    summary = get_daily_garage_summary(
        conn,
        day,
        open_time=open_time,
        close_time=close_time,
        bay_ids=bay_ids,
    )
    shop = summary["shop"]
    techs = summary["technicians"]
    bays = summary["bays"]
    garage = venue or "Garage"

    lines = [
        f"🔧 {garage} — Daily Scorecard",
        day.isoformat(),
        "",
        f"Shop hours: {open_time}–{close_time}  ({shop['operating_hours']:.1f}h elapsed)",
        f"Total operating hours (staff): {shop['total_shift_hours']:.1f}h",
        f"Shop utilization: {shop['utilization_pct']:.0f}%",
        "",
    ]

    if not techs:
        lines.append("No technician attendance recorded today.")
    else:
        lines.append("Mechanics")
        for tech in techs:
            score = float(tech["performance_score"])
            shift_h = _fmt_hours_from_minutes(tech["total_shift_minutes"])
            wrench_h = _fmt_hours_from_minutes(tech["wrench_minutes"])
            idle_h = _fmt_hours_from_minutes(tech["idle_minutes"])
            lines.append(
                f"• {tech['staff_name']}: in {_fmt_clock(tech['clock_in_time'])}  "
                f"out {_fmt_clock(tech['clock_out_time'])}"
            )
            lines.append(
                f"  Wrench {wrench_h} ({score:.0f}%)  Idle {idle_h}  "
                f"{efficiency_badge(score)}"
            )

    lines.append("")
    if not bays:
        lines.append("Bay utilization: no sessions.")
    else:
        lines.append("Bays")
        for bay in bays:
            active_h = float(bay["active_seconds"]) / 3600.0
            lines.append(
                f"• {bay['bay_id']}: {bay['utilization_pct']:.0f}% utilized  "
                f"wrench {active_h:.1f}h"
            )

    events = day_events(conn, day)
    abandoned = [row for row in events if row["event_type"] == "abandoned"]
    paths: list[Path] = []
    for row in abandoned:
        if not row["abs_path"]:
            continue
        path = Path(row["abs_path"])
        if path.is_file():
            paths.append(path)

    if len(abandoned) > 10:
        lines.append(
            f"Album capped at 10 stills; {len(abandoned) - 10} more remain on disk."
        )
    if len(paths) > 10:
        paths = paths[:9] + [paths[-1]]

    return "\n".join(lines), paths


def build_report(
    conn,
    day: date,
    venue: str,
    open_time: str,
    close_time: str = "18:00",
    bay_ids: list[str] | None = None,
) -> tuple[str, list[Path]]:
    """End-of-day garage scorecard (legacy name kept for CLI / Telegram)."""
    summary = get_daily_garage_summary(
        conn, day, open_time=open_time, close_time=close_time, bay_ids=bay_ids
    )
    if summary["technicians"] or summary["bays"]:
        return build_garage_report(
            conn, day, venue, open_time, close_time=close_time, bay_ids=bay_ids
        )

    events = day_events(conn, day)
    minutes = day_minutes(conn, day)
    abandoned = [row for row in events if row["event_type"] == "abandoned"]
    opened_rows = [row for row in events if row["event_type"] == "opened"]

    lines = [
        f"Shift report — {venue}",
        day.isoformat(),
        "",
    ]

    if opened_rows:
        opened_at = datetime.fromisoformat(opened_rows[0]["ts"])
        scheduled = datetime.combine(day, _parse_open_time(open_time))
        delta = opened_at - scheduled
        late_min = int(delta.total_seconds() // 60)
        if late_min > 0:
            lines.append(
                f"Store opened {late_min} min late ({opened_at.strftime('%H:%M')} vs {open_time})."
            )
        else:
            lines.append(f"Store opened {opened_at.strftime('%H:%M')} (on time vs {open_time}).")
    else:
        lines.append("No occupancy recorded today.")

    lines.append(f"Unattended alerts: {len(abandoned)}.")

    if minutes:
        peak = max(minutes, key=lambda row: row["max_persons"])
        peak_clock = peak["minute"].split(" ")[-1]
        lines.append(f"Peak traffic at {peak_clock} ({peak['max_persons']} people in frame).")
    else:
        lines.append("Peak traffic: no samples.")

    paths: list[Path] = []
    for row in abandoned:
        if not row["abs_path"]:
            continue
        path = Path(row["abs_path"])
        if path.is_file():
            paths.append(path)

    if len(paths) > 10:
        paths = paths[:9] + [paths[-1]]

    return "\n".join(lines), paths
