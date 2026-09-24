"""Champei Spa edge production runner.

Starts inbound Telegram controller + daily scorecard scheduler, then either
runs a mock heartbeat (--mock) or wraps launcher.start_unified_server (default).
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from datetime import datetime, time as dt_time
from pathlib import Path

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
        print(f"[run_champei] sqlite journal_mode={mode} path={db_path}", flush=True)
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


def _mock_loop(stop_event: threading.Event) -> None:
    print("[run_champei] mock mode — heartbeat (Ctrl-C / SIGTERM to stop)", flush=True)
    while not stop_event.is_set():
        print(f"[run_champei] heartbeat {datetime.now().isoformat(timespec='seconds')}", flush=True)
        stop_event.wait(60.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Champei Spa edge intelligence daemon")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Skip camera hub; run heartbeat + Telegram controller only",
    )
    parser.add_argument("--port", type=int, default=8765, help="Hub HTTP port")
    parser.add_argument("--branch", default="champei-pp-01", help="Branch id for scorecards")
    args = parser.parse_args(argv)

    # Import launcher early so load_dotenv_files() populates TELEGRAM_* from .env.
    import launcher  # noqa: F401

    db_path = data_dir() / "events.db"
    _verify_wal(db_path)

    stop_event = threading.Event()
    controller = TelegramController(db_path=db_path, branch_id=args.branch)
    tg_thread = threading.Thread(
        target=controller.run_loop,
        kwargs={"stop_event": stop_event},
        name="telegram-controller",
        daemon=False,
    )
    score_thread = threading.Thread(
        target=_scorecard_loop,
        kwargs={
            "stop_event": stop_event,
            "branch_id": args.branch,
            "db_path": db_path,
        },
        name="daily-scorecard",
        daemon=False,
    )
    tg_thread.start()
    score_thread.start()
    print("[run_champei] Telegram controller + scorecard scheduler started", flush=True)

    def _stop_background(*_args: object) -> None:
        stop_event.set()

    try:
        if args.mock:
            signal.signal(signal.SIGINT, _stop_background)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, _stop_background)
            _mock_loop(stop_event)
        else:
            # Hub owns SIGINT/SIGTERM; we join background threads after it returns.
            from launcher import start_unified_server

            start_unified_server(port=args.port, open_browser=False)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        tg_thread.join(timeout=5.0)
        score_thread.join(timeout=5.0)
        print("[run_champei] shutdown complete", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
