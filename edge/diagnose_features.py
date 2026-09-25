#!/usr/bin/env python3
"""Read-only Champei feature audit diagnostic.

Inspects shipped hospitality modules and prints a structured matrix of
features, spatial coverage, edge cases, and recommendations. Does not
modify vision, tracker, or workplace files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Dynamic inspection (read-only imports)
# ---------------------------------------------------------------------------

try:
    import visitor_registry as _vr
except Exception:  # pragma: no cover - defensive
    _vr = None  # type: ignore

try:
    from analytics import sessions as _sessions
except Exception:  # pragma: no cover
    _sessions = None  # type: ignore

try:
    from analytics import scorecard as _scorecard  # noqa: F401
except Exception:  # pragma: no cover
    _scorecard = None  # type: ignore

try:
    from analytics import churn as _churn  # noqa: F401
except Exception:  # pragma: no cover
    _churn = None  # type: ignore

try:
    import telegram_out as _tg
except Exception:  # pragma: no cover
    _tg = None  # type: ignore

try:
    import db as _db  # noqa: F401
except Exception:  # pragma: no cover
    _db = None  # type: ignore


def _const(mod: Any, name: str, default: Any) -> Any:
    if mod is None:
        return default
    return getattr(mod, name, default)


ARRIVAL_COOLDOWN_SECONDS = int(_const(_vr, "ARRIVAL_COOLDOWN_SECONDS", 3600))
EARLY_DEPARTURE_SECONDS = int(_const(_sessions, "EARLY_DEPARTURE_SECONDS", 1200))
STAFF_EVENTS = frozenset(_const(_tg, "STAFF_EVENTS", frozenset()))
OWNER_EVENTS = frozenset(_const(_tg, "OWNER_EVENTS", frozenset()))

EARLY_DEPARTURE_MINS = max(1, EARLY_DEPARTURE_SECONDS // 60)

BOLD = "\033[1m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"


@dataclass(frozen=True)
class FeatureRow:
    feature: str
    module: str
    db: str
    event_role: str
    trigger: str
    status: str  # OK | WARNING | UNCOVERED (+ detail)


@dataclass(frozen=True)
class EdgeCase:
    name: str
    status: str
    mitigation: str
    gap: str


def _status_tag(status: str) -> str:
    key = status.split("—")[0].split("-")[0].strip().upper()
    if key == "OK":
        return f"{GREEN}OK{RESET}"
    if key == "WARNING":
        return f"{YELLOW}WARNING{RESET}"
    if key == "UNCOVERED":
        return f"{RED}UNCOVERED{RESET}"
    return status


def _status_bucket(status: str) -> str:
    key = status.split("—")[0].split("-")[0].strip().upper()
    if key in ("OK", "WARNING", "UNCOVERED"):
        return key
    return "WARNING"


def feature_matrix() -> list[FeatureRow]:
    return [
        FeatureRow(
            feature="VIP Handshake & Cooldown",
            module="visitor_registry.notify_guest_arrival",
            db="visitor_meta(alias, vip_tier)",
            event_role="vip_arrival / Staff",
            trigger=f"Named alias + cooldown ({ARRIVAL_COOLDOWN_SECONDS}s)",
            status="WARNING — Helper ready, pending live launcher wiring",
        ),
        FeatureRow(
            feature="Guest Arrival (Unnamed)",
            module="visitor_registry.notify_guest_arrival",
            db="visitor_meta",
            event_role="guest_arrival / Staff",
            trigger=f"Visit count check + cooldown ({ARRIVAL_COOLDOWN_SECONDS}s)",
            status="WARNING — Helper ready, pending live launcher wiring",
        ),
        FeatureRow(
            feature="Wait SLA Bottleneck",
            module="customer_visits.py (adjacent) + scorecard",
            db="events(event_type='wait_bottleneck')",
            event_role="wait_bottleneck / Staff",
            trigger="180s dwell in waiting/reception zone",
            status="WARNING — Live path uses send_message, bypasses send_alert staff route",
        ),
        FeatureRow(
            feature="Session Duration",
            module="sessions.record_session_completion / db.end_customer_visit",
            db="customer_visits(duration_seconds, ended_at, status)",
            event_role="None (Internal DB metric)",
            trigger="Duration calculated on session close",
            status="WARNING — Live end path skips status & early-departure helper",
        ),
        FeatureRow(
            feature="Early Departure Anomaly",
            module="sessions.record_session_completion",
            db="customer_visits",
            event_role="early_departure / Dual (Staff + Owner)",
            trigger=f"< {EARLY_DEPARTURE_SECONDS}s (<{EARLY_DEPARTURE_MINS}m)",
            status="WARNING — Active in helper, pending live exit trigger",
        ),
        FeatureRow(
            feature="Lobby Bounce / Walk-Away",
            module="sessions.record_walk_away",
            db="events(event_type='walk_away')",
            event_role="Internal DB only (Scorecard)",
            trigger="Explicit helper invocation",
            status="UNCOVERED — No vision bounce detector wired",
        ),
        FeatureRow(
            feature="Daily Ops Scorecard",
            module="scorecard.build_daily_scorecard / send_daily_scorecard",
            db="customer_visits + events",
            event_role="daily_scorecard / Owner",
            trigger="Scheduled daily at 21:00 or manual CLI",
            status="WARNING — Ready via CLI, pending background scheduler",
        ),
        FeatureRow(
            feature="Silent Churn Watchlist",
            module="churn.compute_churn_risks / send_weekly_churn",
            db="customer_visits + visitor_meta",
            event_role="silent_churn / Owner",
            trigger="Days overdue > max(cadence * 2.0, 21d)",
            status="WARNING — Ready via CLI, pending background scheduler",
        ),
    ]


def edge_cases() -> list[EdgeCase]:
    return [
        EdgeCase(
            name="Staff vs. Customer Confusion",
            status="WARNING",
            mitigation="staff_memory and is_staff flags in workplace monitor.",
            gap=(
                "Therapists attending guests at the bench or rack outside the "
                "reception desk ROI may enroll as customers if not filtered by "
                "uniform color or staff ID."
            ),
        ),
        EdgeCase(
            name="Temporary Exits (Smoke / Phone Calls)",
            status="UNCOVERED",
            mitigation="Live end_customer_visit uses ~8s track loss.",
            gap=(
                "No 5–10 minute re-entry grace period in record_session_completion. "
                "A short exit can split 1 treatment into 2 visits or fire a false "
                "early departure alert."
            ),
        ),
        EdgeCase(
            name="Multi-Guest Occlusion at Bench",
            status="WARNING",
            mitigation="DeepSORT tracking bounding boxes.",
            gap=(
                "Two guests sitting closely or swapping shoes at the same time "
                "can cause bounding-box collisions or Re-ID vector swaps."
            ),
        ),
        EdgeCase(
            name="Slipper Re-Identification",
            status="WARNING",
            mitigation="OSNet full-body feature extraction.",
            gap=(
                "Guests enter in dark shoes and return in white spa slippers. "
                "The tracker must prioritize torso/upper-body embeddings to avoid "
                "identity fragmentation."
            ),
        ),
        EdgeCase(
            name="Entourage / Non-Paying Seating",
            status="WARNING",
            mitigation="None.",
            gap=(
                "Bench dwell time fires the Wait SLA bottleneck alert even if the "
                "person is a driver or bodyguard who never intended to take a service."
            ),
        ),
    ]


def recommendations() -> list[str]:
    return [
        (
            f"Arrival Cooldown ({ARRIVAL_COOLDOWN_SECONDS}s): Evaluate lowering to "
            "900s–1800s (15–30m) to accommodate same-day repeat visits or quick "
            "add-on services."
        ),
        (
            f"Early Departure Threshold ({EARLY_DEPARTURE_SECONDS}s): Champei's "
            "minimum menu service is 60 minutes ($17–$27). Consider raising the "
            "anomaly threshold to 1800s–2400s (30–40m) to catch shortened 1-hour "
            "sessions."
        ),
        (
            "Re-Entry Grace Window: Add a 5–10 minute buffer before finalizing "
            "record_session_completion to handle mid-service cigarette or phone breaks."
        ),
        (
            "Live Pipeline Wiring: Connect notify_guest_arrival, "
            "record_session_completion, and record_walk_away directly to "
            "CustomerVisitMonitor state transitions. Route live SLA alerts through "
            'TelegramOut.send_alert("wait_bottleneck", ...).'
        ),
        (
            "Staff Greeting Latency: Track elapsed seconds from glass door opening "
            "until a therapist arrives in the lounge."
        ),
        (
            "Entourage Filter: Require verified shoe-swap posture before enrolling "
            "a visitor into the wait SLA counter."
        ),
        (
            "Zone Alignment: Add shoe_lounge to workplace configuration files to "
            "match notification metadata."
        ),
    ]


def _wrap(text: str, width: int) -> list[str]:
    if not text:
        return [""]
    if width < 8:
        width = 8
    lines: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= width:
            lines.append(remaining)
            break
        window = remaining[: width + 1]
        # Prefer break after . _ / space within the window
        brk = -1
        for i in range(width, max(width // 3, 0), -1):
            if remaining[i - 1] in "._/ ":
                brk = i
                break
        if brk < 0:
            brk = width
        lines.append(remaining[:brk].rstrip())
        remaining = remaining[brk:].lstrip()
    return lines or [""]


def _ascii_table(headers: list[str], rows: list[list[str]], widths: list[int]) -> str:
    # Expand each cell to wrapped lines, then pad row height
    rendered: list[list[list[str]]] = []
    for row in rows:
        wrapped_cells = [_wrap(c, widths[i]) for i, c in enumerate(row)]
        height = max(len(c) for c in wrapped_cells)
        padded = [c + [""] * (height - len(c)) for c in wrapped_cells]
        rendered.append(padded)

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt_line(parts: list[str]) -> str:
        return "| " + " | ".join(parts[i].ljust(widths[i]) for i in range(len(widths))) + " |"

    out = [sep, fmt_line([h[: widths[i]].ljust(widths[i]) for i, h in enumerate(headers)]), sep]
    for row_cells in rendered:
        height = len(row_cells[0])
        for li in range(height):
            out.append(fmt_line([row_cells[i][li] for i in range(len(widths))]))
        out.append(sep)
    return "\n".join(out)


def _section(title: str) -> str:
    bar = "=" * 72
    return f"\n{BOLD}{CYAN}{bar}\n{title}\n{bar}{RESET}\n"


def _preamble_ascii() -> str:
    lines = [
        f"{BOLD}Champei Feature Audit Diagnostic{RESET}",
        "Mode: READ-ONLY inspection (no vision/tracker/workplace edits)",
        "",
        "Live constants:",
        f"  ARRIVAL_COOLDOWN_SECONDS = {ARRIVAL_COOLDOWN_SECONDS}",
        f"  EARLY_DEPARTURE_SECONDS  = {EARLY_DEPARTURE_SECONDS}",
        f"  STAFF_EVENTS  = {sorted(STAFF_EVENTS) if STAFF_EVENTS else '(unavailable)'}",
        f"  OWNER_EVENTS  = {sorted(OWNER_EVENTS) if OWNER_EVENTS else '(unavailable)'}",
        "",
        "Schema notes:",
        "  customer_visits: subject_id, started_at, ended_at, duration_seconds, status, zone_id",
        "    (status via sessions.ensure_session_columns; db.connect migrates duration_seconds)",
        "  events: ts, event_type, branch_id (+ camera_role migrate)",
        "  visitor_meta: visitor_id, alias, vip_tier, notes (registry-owned, not in db.connect)",
        "  adjacent: staff_memory, anonymous_subjects",
    ]
    return "\n".join(lines)


def render_ascii() -> str:
    parts: list[str] = [_preamble_ascii()]

    # Section 1
    parts.append(_section("SECTION 1: ACTIVE FEATURE MATRIX"))
    features = feature_matrix()
    rows = [
        [f.feature, f.module, f.db, f.event_role, f.trigger, f.status]
        for f in features
    ]
    parts.append(
        _ascii_table(
            ["Feature", "Module.Function", "DB Tables/Cols", "Event + Role", "Trigger", "Status"],
            rows,
            [22, 28, 28, 26, 28, 36],
        )
    )

    # Section 2
    parts.append(_section("SECTION 2: CAMERA & SPATIAL COVERAGE (ENTRANCE CCTV VIEW)"))
    zones = [
        (
            "Zone A (Glass Door & Welcome Mat)",
            "Coverage: Entry/Exit tripwires, Lobby Bounces, Group Clustering.",
            "Diagnostic: Walk-away detection is UNCOVERED. Visit start relies on "
            "generalized workplace zones rather than shoe-swap gating.",
        ),
        (
            "Zone B (Wooden Changing Bench)",
            "Coverage: Seating dwell time, Wait SLA (>180s), Shoe-untie posture.",
            "Diagnostic: Wait SLA is OK if zone kinds include waiting/reception. "
            "Shoe-swap posture detection is UNCOVERED in hospitality helpers.",
        ),
        (
            "Zone C (Shoe Cubby & Hallway Threshold)",
            "Coverage: Footwear storage motion, Inward hallway crossing.",
            "Diagnostic: shoe_lounge label is used in notification text, but zone "
            "kind does not exist in standard massage VISIT_ZONE_KINDS. Status: WARNING "
            "(Naming mismatch).",
        ),
    ]
    for name, cov, diag in zones:
        parts.append(f"{BOLD}{name}{RESET}")
        parts.append(f"  {cov}")
        parts.append(f"  {diag}")
        parts.append("")

    # Section 3
    parts.append(_section("SECTION 3: EDGE CASE & VULNERABILITY SCAN"))
    cases = edge_cases()
    for case in cases:
        parts.append(f"{BOLD}{case.name}{RESET}  [{_status_tag(case.status)}]")
        parts.append(f"  Mitigation: {case.mitigation}")
        parts.append(f"  Gap:        {case.gap}")
        parts.append("")

    # Section 4
    parts.append(_section("SECTION 4: GAP & RECOMMENDATION ENGINE"))
    for i, rec in enumerate(recommendations(), 1):
        parts.append(f"  {i}. {rec}")
        parts.append("")

    # Section 5
    parts.append(_section("SECTION 5: SUMMARY TALLY"))
    buckets = {"OK": 0, "WARNING": 0, "UNCOVERED": 0}
    for f in features:
        buckets[_status_bucket(f.status)] += 1
    for case in cases:
        buckets[_status_bucket(case.status)] += 1
    # Zone diagnostics with explicit status tags
    buckets["UNCOVERED"] += 2  # Zone A walk-away, Zone B shoe posture
    buckets["WARNING"] += 1  # Zone C naming mismatch
    # Zone B also notes Wait SLA OK when configured
    buckets["OK"] += 1

    parts.append(
        f"  {GREEN}OK: {buckets['OK']}{RESET} | "
        f"{YELLOW}WARNING: {buckets['WARNING']}{RESET} | "
        f"{RED}UNCOVERED: {buckets['UNCOVERED']}{RESET}"
    )
    parts.append("")
    return "\n".join(parts)


def render_markdown() -> str:
    lines: list[str] = [
        "# Champei Feature Audit Diagnostic",
        "",
        "Mode: **READ-ONLY** inspection (no vision/tracker/workplace edits).",
        "",
        "## Live constants",
        "",
        f"- `ARRIVAL_COOLDOWN_SECONDS` = `{ARRIVAL_COOLDOWN_SECONDS}`",
        f"- `EARLY_DEPARTURE_SECONDS` = `{EARLY_DEPARTURE_SECONDS}`",
        f"- `STAFF_EVENTS` = `{sorted(STAFF_EVENTS) if STAFF_EVENTS else 'unavailable'}`",
        f"- `OWNER_EVENTS` = `{sorted(OWNER_EVENTS) if OWNER_EVENTS else 'unavailable'}`",
        "",
        "## Schema notes",
        "",
        "- `customer_visits`: `subject_id`, `started_at`, `ended_at`, `duration_seconds`, `status`, `zone_id`",
        "  - `status` via `sessions.ensure_session_columns`; `db.connect` migrates `duration_seconds`",
        "- `events`: `ts`, `event_type`, `branch_id` (+ `camera_role` migrate)",
        "- `visitor_meta`: `visitor_id`, `alias`, `vip_tier`, `notes` (registry-owned, not in `db.connect`)",
        "- Adjacent: `staff_memory`, `anonymous_subjects`",
        "",
        "## SECTION 1: Active Feature Matrix",
        "",
        "| Feature | Module.Function | DB Tables/Cols | Event + Role | Trigger | Status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for f in feature_matrix():
        lines.append(
            f"| {f.feature} | `{f.module}` | `{f.db}` | {f.event_role} | {f.trigger} | {f.status} |"
        )

    lines.extend(
        [
            "",
            "## SECTION 2: Camera & Spatial Coverage (Entrance CCTV View)",
            "",
            "### Zone A (Glass Door & Welcome Mat)",
            "",
            "- **Coverage:** Entry/Exit tripwires, Lobby Bounces, Group Clustering.",
            "- **Diagnostic:** Walk-away detection is UNCOVERED. Visit start relies on "
            "generalized workplace zones rather than shoe-swap gating.",
            "",
            "### Zone B (Wooden Changing Bench)",
            "",
            "- **Coverage:** Seating dwell time, Wait SLA (>180s), Shoe-untie posture.",
            "- **Diagnostic:** Wait SLA is OK if zone kinds include waiting/reception. "
            "Shoe-swap posture detection is UNCOVERED in hospitality helpers.",
            "",
            "### Zone C (Shoe Cubby & Hallway Threshold)",
            "",
            "- **Coverage:** Footwear storage motion, Inward hallway crossing.",
            "- **Diagnostic:** `shoe_lounge` label is used in notification text, but zone "
            "kind does not exist in standard massage `VISIT_ZONE_KINDS`. Status: WARNING "
            "(Naming mismatch).",
            "",
            "## SECTION 3: Edge Case & Vulnerability Scan",
            "",
            "| Risk | Status | Current Mitigation | Remaining Gap |",
            "| --- | --- | --- | --- |",
        ]
    )
    for case in edge_cases():
        lines.append(
            f"| {case.name} | {case.status} | {case.mitigation} | {case.gap} |"
        )

    lines.extend(["", "## SECTION 4: Gap & Recommendation Engine", ""])
    for i, rec in enumerate(recommendations(), 1):
        lines.append(f"{i}. {rec}")

    features = feature_matrix()
    cases = edge_cases()
    buckets = {"OK": 0, "WARNING": 0, "UNCOVERED": 0}
    for f in features:
        buckets[_status_bucket(f.status)] += 1
    for case in cases:
        buckets[_status_bucket(case.status)] += 1
    buckets["UNCOVERED"] += 2
    buckets["WARNING"] += 1
    buckets["OK"] += 1

    lines.extend(
        [
            "",
            "## SECTION 5: Summary Tally",
            "",
            f"**OK: {buckets['OK']} | WARNING: {buckets['WARNING']} | UNCOVERED: {buckets['UNCOVERED']}**",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only Champei feature audit diagnostic",
    )
    parser.add_argument(
        "--write-md",
        action="store_true",
        help="Also write edge/FEATURE_AUDIT.md",
    )
    args = parser.parse_args(argv)

    ascii_report = render_ascii()
    print(ascii_report)

    if args.write_md:
        out = Path(__file__).resolve().parent / "FEATURE_AUDIT.md"
        out.write_text(render_markdown(), encoding="utf-8")
        print(f"{GREEN}Wrote {out}{RESET}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
