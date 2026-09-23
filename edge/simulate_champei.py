#!/usr/bin/env python3
"""End-to-end Champei entrance lifecycle simulation harness.

Exercises visitor registry, session analytics, scorecard, and churn against a
shared temp SQLite DB. Does not touch vision/tracker modules.
"""

from __future__ import annotations

import argparse
import secrets
import sqlite3
import sys
import tempfile
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from analytics.churn import compute_churn_risks, format_churn_watchlist
from analytics.scorecard import build_daily_scorecard, query_daily_counts
from analytics.sessions import (
    ensure_events_table,
    ensure_session_columns,
    record_session_completion,
    record_walk_away,
)
from telegram_out import TelegramOut
from visitor_registry import (
    connect_registry,
    get_visitor_display_name,
    notify_guest_arrival,
    reset_notify_cooldowns,
    set_visitor_alias,
)

BRANCH = "champei-pp-01"
DEFAULT_DB = Path(tempfile.gettempdir()) / "champei_sim.db"

GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _banner(step: int, title: str) -> None:
    print(f"\n{BOLD}{CYAN}=== STEP {step}: {title} ==={RESET}")


def _ok(msg: str) -> None:
    print(f"{GREEN}[OK]{RESET} {msg}")


def _fail(step: int, title: str, exc: BaseException) -> int:
    print(f"{RED}[FAIL]{RESET} STEP {step}: {title}")
    print(f"{RED}{exc}{RESET}")
    traceback.print_exc()
    return 1


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def _bootstrap(db_path: Path) -> None:
    reg = connect_registry(db_path)
    reg.close()
    conn = sqlite3.connect(str(db_path), timeout=10)
    try:
        ensure_session_columns(conn)
        ensure_events_table(conn)
    finally:
        conn.close()


def _insert_open_visit(db_path: Path, subject_id: str, started_at: str, zone_id: str) -> str:
    visit_id = f"sim-{secrets.token_hex(4)}"
    conn = sqlite3.connect(str(db_path), timeout=10)
    try:
        ensure_session_columns(conn)
        conn.execute(
            """
            INSERT INTO customer_visits (
                id, subject_id, zone_id, started_at, ended_at, source, status
            ) VALUES (?, ?, ?, ?, NULL, 'sim', 'open')
            """,
            (visit_id, subject_id, zone_id, started_at),
        )
        conn.commit()
    finally:
        conn.close()
    return visit_id


def _insert_completed_visit(
    db_path: Path,
    subject_id: str,
    started_at: str,
    ended_at: str,
    duration_seconds: float,
) -> str:
    visit_id = f"sim-{secrets.token_hex(4)}"
    conn = sqlite3.connect(str(db_path), timeout=10)
    try:
        ensure_session_columns(conn)
        conn.execute(
            """
            INSERT INTO customer_visits (
                id, subject_id, zone_id, started_at, ended_at,
                duration_seconds, status, source
            ) VALUES (?, ?, 'shoe_lounge', ?, ?, ?, 'completed', 'sim')
            """,
            (visit_id, subject_id, started_at, ended_at, float(duration_seconds)),
        )
        conn.commit()
    finally:
        conn.close()
    return visit_id


def _insert_bottleneck(db_path: Path, ts: str) -> None:
    conn = sqlite3.connect(str(db_path), timeout=10)
    try:
        ensure_events_table(conn)
        conn.execute(
            "INSERT INTO events (ts, event_type, abs_path, branch_id) VALUES (?, ?, ?, ?)",
            (ts, "wait_bottleneck", f"sim-bottleneck-{secrets.token_hex(3)}", BRANCH),
        )
        conn.commit()
    finally:
        conn.close()


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _dry_bot() -> MagicMock:
    bot = MagicMock(spec=TelegramOut)
    bot.send_alert.return_value = True
    return bot


def _event_calls(bot: Any) -> list[tuple[Any, ...]]:
    return [c.args for c in bot.send_alert.call_args_list]


def run_simulation(*, dry_run: bool, db_path: Path) -> int:
    if dry_run:
        bot: Any = _dry_bot()
        mode = "dry-run (MagicMock TelegramOut)"
    else:
        bot = TelegramOut()
        mode = "live (real TelegramOut)"

    print(f"{BOLD}Champei entrance simulation{RESET}")
    print(f"  mode: {mode}")
    print(f"  db:   {db_path}")

    reset_notify_cooldowns()
    _bootstrap(db_path)
    now = datetime.now()

    # --- STEP 1 ---
    step, title = 1, "VIP REGISTRY UPSERT"
    try:
        _banner(step, title)
        set_visitor_alias(
            "guest_white_shirt",
            "Lok Chumteav Sophy",
            vip_tier="VIP",
            notes="Prefers Room 2, Lemongrass tea",
            db_path=db_path,
        )
        meta = get_visitor_display_name("guest_white_shirt", db_path=db_path)
        _assert(meta.get("is_named") is True, f"expected named VIP, got {meta!r}")
        _ok(f"Alias upserted: {meta.get('display_name')} ({meta.get('tier')})")
    except Exception as exc:
        return _fail(step, title, exc)

    # --- STEP 2 ---
    step, title = 2, "LOBBY BOUNCE / WALK-AWAY"
    try:
        _banner(step, title)
        ok = record_walk_away(branch_id=BRANCH, db_path=db_path)
        _assert(ok is True, "record_walk_away returned False")
        _ok("[EVENT] Walk-away / Lobby Bounce recorded.")
    except Exception as exc:
        return _fail(step, title, exc)

    # --- STEP 3 ---
    step, title = 3, "PAYING GUEST ARRIVAL"
    try:
        _banner(step, title)
        started = _iso(now - timedelta(minutes=15))
        _insert_open_visit(db_path, "guest_white_shirt", started, "shoe_lounge")
        notified = notify_guest_arrival(
            "guest_white_shirt",
            visit_count=4,
            branch_id=BRANCH,
            zone="shoe_lounge",
            telegram=bot,
            db_path=db_path,
        )
        _assert(notified is True, "notify_guest_arrival returned False")
        if dry_run:
            calls = _event_calls(bot)
            _assert(calls, "no send_alert calls after VIP arrival")
            etype, text = calls[-1][0], calls[-1][1]
            _assert(etype == "vip_arrival", f"expected vip_arrival, got {etype!r}")
            _assert(
                "Lok Chumteav Sophy" in str(text),
                f"VIP alias missing from alert text: {text!r}",
            )
        _ok("[EVENT] VIP Arrival alert triggered for Lok Chumteav Sophy.")
    except Exception as exc:
        return _fail(step, title, exc)

    # --- STEP 4 ---
    step, title = 4, "RATE-LIMIT COOLDOWN SUPPRESSION"
    try:
        _banner(step, title)
        suppressed = notify_guest_arrival(
            "guest_white_shirt",
            visit_count=4,
            branch_id=BRANCH,
            zone="shoe_lounge",
            telegram=bot,
            db_path=db_path,
        )
        _assert(suppressed is False, "expected cooldown to suppress second notify")
        _ok("[COOLDOWN] Duplicate trigger suppressed.")
    except Exception as exc:
        return _fail(step, title, exc)

    # --- STEP 5 ---
    step, title = 5, "EARLY DEPARTURE ANOMALY (<20 MINS)"
    try:
        _banner(step, title)
        before_calls = len(_event_calls(bot)) if dry_run else 0
        result = record_session_completion(
            subject_id="guest_white_shirt",
            branch_id=BRANCH,
            db_path=db_path,
            telegram=bot,
        )
        _assert(
            result.get("is_early_departure") is True,
            f"expected early departure, got {result!r}",
        )
        if dry_run:
            new_calls = _event_calls(bot)[before_calls:]
            _assert(
                any(c and c[0] == "early_departure" for c in new_calls),
                f"early_departure not dispatched: {new_calls!r}",
            )
        mins = int(round(float(result.get("duration_seconds") or 0) / 60.0))
        _ok(f"[ALERT] Early departure anomaly detected ({mins}m < 20m threshold).")
    except Exception as exc:
        return _fail(step, title, exc)

    # --- STEP 6 ---
    step, title = 6, "STANDARD COMPLETED 75-MIN SESSION & SLA BOTTLENECK"
    try:
        _banner(step, title)
        started_reg = _iso(now - timedelta(minutes=90))
        ended_reg = _iso(now - timedelta(minutes=15))
        _insert_completed_visit(
            db_path,
            "guest_regular_02",
            started_reg,
            ended_reg,
            4500.0,
        )
        _insert_bottleneck(db_path, _iso(now))
        _ok("Inserted 75-min completed visit + wait_bottleneck event.")
    except Exception as exc:
        return _fail(step, title, exc)

    # --- STEP 7 ---
    step, title = 7, "SCORECARD & CHURN VERIFICATION"
    try:
        _banner(step, title)
        day = now.date().isoformat()
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            counts = query_daily_counts(conn, day)
        finally:
            conn.close()

        _assert(counts["total_visits"] == 2, f"total_visits={counts['total_visits']}")
        _assert(
            counts["completed_sessions"] == 2,
            f"completed_sessions={counts['completed_sessions']}",
        )
        _assert(counts["unique_guests"] == 2, f"unique_guests={counts['unique_guests']}")
        _assert(counts["walk_aways"] == 1, f"walk_aways={counts['walk_aways']}")
        _assert(counts["bottlenecks"] == 1, f"bottlenecks={counts['bottlenecks']}")
        avg = float(counts["avg_duration_minutes"] or 0.0)
        _assert(40.0 <= avg <= 50.0, f"avg_duration_minutes={avg} (expected ~45)")

        scorecard = build_daily_scorecard(db_path=db_path, branch_id=BRANCH, day=day)
        risks = compute_churn_risks(db_path=db_path)
        churn_text = format_churn_watchlist(risks, branch_id=BRANCH)

        print()
        print(scorecard)
        print()
        print(churn_text)
        print()
        _ok("Scorecard metrics matched expected simulation totals.")
    except Exception as exc:
        return _fail(step, title, exc)

    print(f"\n{GREEN}{BOLD}ALL 7 STEPS PASSED{RESET}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Champei entrance lifecycle simulation harness",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Mock Telegram dispatch (default)",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Dispatch via real TelegramOut / env vars",
    )
    parser.add_argument(
        "--db",
        default="",
        help=f"SQLite path (default: {DEFAULT_DB})",
    )
    args = parser.parse_args(argv)

    dry_run = not bool(args.live)
    custom_db = bool(str(args.db or "").strip())
    db_path = Path(args.db).expanduser() if custom_db else DEFAULT_DB

    if not custom_db and db_path.exists():
        db_path.unlink()

    return run_simulation(dry_run=dry_run, db_path=db_path)


if __name__ == "__main__":
    raise SystemExit(main())
