"""Champei Spa edge production runner.

Starts inbound Telegram controller + daily scorecard scheduler + weekly
customer activity brief, then either runs a mock heartbeat (--mock) or wraps
launcher.start_unified_server (default).
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from datetime import datetime, time as dt_time, timedelta, timezone, tzinfo as dt_tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

# Ensure edge/ is on sys.path when launched as a script.
_EDGE = Path(__file__).resolve().parent
if str(_EDGE) not in sys.path:
    sys.path.insert(0, str(_EDGE))

from db import connect  # noqa: E402
from paths import data_dir  # noqa: E402
from telegram_controller import TelegramController  # noqa: E402


def _verify_wal(db_path: Path) -> str:
    conn = connect(db_path)
    try:
        mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0] or "").lower()
        if mode != "wal":
            raise SystemExit(f"[run_champei] expected WAL journal_mode, got {mode!r}")
        print(f"[DB] SQLite WAL mode confirmed: {mode}", flush=True)
        return mode
    finally:
        conn.close()


def _scorecard_loop(
    stop_event: threading.Event,
    *,
    branch_id: str,
    db_path: Path,
    fire_at: dt_time = dt_time(21, 0),
) -> None:
    """Fire send_daily_scorecard once per local calendar day at fire_at."""
    from analytics.scorecard import send_daily_scorecard

    last_sent_day: str | None = None
    while not stop_event.is_set():
        now = datetime.now()
        day = now.date().isoformat()
        if (
            last_sent_day != day
            and (now.hour, now.minute) >= (fire_at.hour, fire_at.minute)
        ):
            try:
                ok = send_daily_scorecard(db_path=db_path, branch_id=branch_id)
                print(
                    f"[run_champei] daily scorecard {'sent' if ok else 'skipped/failed'} day={day}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[run_champei] scorecard error: {exc}", flush=True)
            last_sent_day = day
        stop_event.wait(30.0)


def _local_tz() -> dt_tzinfo:
    name = os.environ.get("TZ", "").strip() or "Asia/Phnom_Penh"
    try:
        return ZoneInfo(name)
    except Exception:
        return datetime.now().astimezone().tzinfo or timezone.utc


def _parse_fire_at(value: str, default: dt_time = dt_time(21, 0)) -> dt_time:
    text = str(value or "").strip()
    if not text:
        return default
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    return default


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def _weekly_brief_loop(
    stop_event: threading.Event,
    *,
    branch_id: str,
    db_path: Path,
    fire_at: dt_time = dt_time(21, 0),
    tz: dt_tzinfo | None = None,
    now_fn=None,
    send_fn=None,
    poll_seconds: float = 30.0,
) -> None:
    """Fire send_weekly_customer_brief once per local week on Sunday at fire_at."""
    from analytics.weekly_customer_brief import send_weekly_customer_brief

    if not _env_flag("WEEKLY_CUSTOMER_BRIEF_ENABLED", True):
        print("[run_champei] weekly customer brief disabled", flush=True)
        return

    zone = tz if tz is not None else _local_tz()
    clock = now_fn if now_fn is not None else (lambda: datetime.now(zone))
    dispatch = send_fn if send_fn is not None else send_weekly_customer_brief

    last_week_key: str | None = None
    while not stop_event.is_set():
        now = clock()
        local = now.astimezone(zone) if now.tzinfo is not None else now.replace(tzinfo=zone)
        if local.weekday() == 6 and (local.hour, local.minute) >= (fire_at.hour, fire_at.minute):
            week_key = (local.date() - timedelta(days=local.weekday())).isoformat()
            if last_week_key != week_key:
                try:
                    ok = dispatch(db_path=db_path, branch_id=branch_id, as_of=local.date())
                    print(
                        f"[run_champei] weekly customer brief "
                        f"{'sent' if ok else 'skipped/failed'} week={week_key}",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[run_champei] weekly customer brief error: {exc}", flush=True)
                last_week_key = week_key
        stop_event.wait(poll_seconds)


def _mock_loop(stop_event: threading.Event) -> None:
    print("[run_champei] mock mode — heartbeat (Ctrl-C / SIGTERM to stop)", flush=True)
    while not stop_event.is_set():
        print(
            f"[run_champei] heartbeat {datetime.now().isoformat(timespec='seconds')}",
            flush=True,
        )
        stop_event.wait(60.0)


def main(argv: list[str] | None = None) -> int:
    default_branch = os.environ.get("BRANCH_ID", "champei-pp-01").strip() or "champei-pp-01"
    try:
        default_port = int(os.environ.get("HUB_PORT", "8000") or "8000")
    except ValueError:
        default_port = 8000

    parser = argparse.ArgumentParser(description="Champei Spa edge intelligence daemon")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Skip camera hub; run heartbeat + Telegram controller only",
    )
    parser.add_argument("--port", type=int, default=default_port, help="Hub HTTP port")
    parser.add_argument(
        "--branch",
        default=default_branch,
        help="Branch id for scorecards",
    )
    args = parser.parse_args(argv)

    # Import launcher early so load_dotenv_files() populates TELEGRAM_* from .env.
    import launcher  # noqa: F401

    db_path = data_dir() / "events.db"
    _verify_wal(db_path)

    stop_event = threading.Event()

    def _stop_background(*_args: object) -> None:
        stop_event.set()

    if args.mock:
        signal.signal(signal.SIGINT, _stop_background)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _stop_background)

    controller = TelegramController(db_path=db_path, branch_id=args.branch)
    tg_thread = threading.Thread(
        target=controller.run_loop,
        kwargs={"stop_event": stop_event},
        name="telegram-controller",
        daemon=True,
    )
    score_thread = threading.Thread(
        target=_scorecard_loop,
        kwargs={
            "stop_event": stop_event,
            "branch_id": args.branch,
            "db_path": db_path,
        },
        name="daily-scorecard",
        daemon=True,
    )
    weekly_thread = threading.Thread(
        target=_weekly_brief_loop,
        kwargs={
            "stop_event": stop_event,
            "branch_id": args.branch,
            "db_path": db_path,
            "fire_at": _parse_fire_at(os.environ.get("WEEKLY_CUSTOMER_BRIEF_AT", "")),
        },
        name="weekly-customer-brief",
        daemon=True,
    )
    tg_thread.start()
    print("[run_champei] Telegram controller started", flush=True)
    score_thread.start()
    print("[run_champei] Scorecard scheduler started", flush=True)
    weekly_thread.start()
    print("[run_champei] Weekly customer brief scheduler started", flush=True)

    try:
        if args.mock:
            _mock_loop(stop_event)
        else:
            # Hub owns SIGINT/SIGTERM; we join background threads after it returns.
            from launcher import start_unified_server

            start_unified_server(port=args.port, open_browser=False)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        tg_thread.join(timeout=2.0)
        score_thread.join(timeout=2.0)
        weekly_thread.join(timeout=2.0)
        print("[run_champei] shutdown complete", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
