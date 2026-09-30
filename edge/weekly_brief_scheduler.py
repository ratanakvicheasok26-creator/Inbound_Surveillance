"""Background scheduler for massage weekly customer Telegram briefs.

Epoch-aligned 7-day windows (default start 2026-09-25). Fires once at the end
of each completed period (last day at fire_at, with catch-up on later days).
Never touches the camera / video worker loop.
"""

from __future__ import annotations

import json
import threading
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from paths import data_dir
from weekly_customer_brief import (
    DEFAULT_EPOCH,
    DEFAULT_PERIOD_DAYS,
    parse_epoch,
    period_key,
    period_window,
    send_weekly_customer_brief,
)


def parse_fire_at(value: Any, default: dt_time = dt_time(21, 0)) -> dt_time:
    text = str(value or "").strip()
    if not text:
        return default
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    return default


def env_or_cfg_flag(raw: Any, default: bool = True) -> bool:
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def _local_tz(name: str | None = None):
    zone_name = (name or "").strip() or "Asia/Phnom_Penh"
    try:
        return ZoneInfo(zone_name)
    except Exception:
        return datetime.now().astimezone().tzinfo or timezone.utc


def state_path(root: Path | None = None) -> Path:
    base = Path(root) if root is not None else data_dir()
    return base / "weekly_customer_brief_state.json"


def load_sent_periods(path: Path) -> set[str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if isinstance(raw, dict):
        items = raw.get("sent_periods") or []
    elif isinstance(raw, list):
        items = raw
    else:
        return set()
    return {str(item) for item in items if item}


def mark_period_sent(path: Path, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sent = load_sent_periods(path)
    sent.add(key)
    payload = {"sent_periods": sorted(sent), "updated_at": datetime.now().isoformat(timespec="seconds")}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def completed_period_ready(
    now: datetime,
    *,
    epoch: date | str | None = None,
    period_days: int = DEFAULT_PERIOD_DAYS,
    fire_at: dt_time = dt_time(21, 0),
) -> tuple[date, date, int] | None:
    """Return the latest fully completed period that is eligible to send."""
    local = now
    start_epoch = parse_epoch(epoch)
    length = max(1, int(period_days))
    # A period ending on day D is ready at fire_at on D, or any time after D.
    candidate_day = local.date()
    window = period_window(candidate_day, epoch=start_epoch, period_days=length)
    if window is None:
        return None
    start, end, idx = window
    if candidate_day < end:
        # Still inside an unfinished period — previous period may still need catch-up.
        if idx <= 0:
            return None
        prev_start = start_epoch + timedelta(days=(idx - 1) * length)
        prev_end = prev_start + timedelta(days=length - 1)
        return prev_start, prev_end, idx - 1
    if candidate_day == end and (local.hour, local.minute) < (fire_at.hour, fire_at.minute):
        if idx <= 0:
            return None
        prev_start = start_epoch + timedelta(days=(idx - 1) * length)
        prev_end = prev_start + timedelta(days=length - 1)
        return prev_start, prev_end, idx - 1
    return start, end, idx


def weekly_brief_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    root = cfg if isinstance(cfg, dict) else {}
    section = root.get("weekly_customer_brief")
    section = section if isinstance(section, dict) else {}
    return {
        "enabled": env_or_cfg_flag(section.get("enabled", True), True),
        "epoch": parse_epoch(section.get("epoch") or DEFAULT_EPOCH),
        "period_days": max(1, int(section.get("period_days") or DEFAULT_PERIOD_DAYS)),
        "fire_at": parse_fire_at(section.get("fire_at") or "21:00"),
        "lookback_days": max(1, int(section.get("lookback_days") or 90)),
        "ai_summary": env_or_cfg_flag(section.get("ai_summary", True), True),
        "ollama_model": str(section.get("ollama_model") or root.get("ollama_model") or "qwen2.5:3b"),
        "ollama_host": str(
            section.get("ollama_host")
            or (root.get("complaint_monitoring") or {}).get("ollama_host")
            or "http://localhost:11434"
        ),
        "tz": str(section.get("tz") or root.get("tz") or "Asia/Phnom_Penh"),
    }


class WeeklyCustomerBriefScheduler:
    """Daemon-friendly poller that sends each completed 7-day massage brief once."""

    def __init__(
        self,
        *,
        get_cfg: Callable[[], dict[str, Any]],
        get_bot: Callable[[], Any],
        db_path: Path | None = None,
        state_file: Path | None = None,
        poll_seconds: float = 30.0,
        now_fn: Callable[[], datetime] | None = None,
        send_fn: Callable[..., bool] | None = None,
        workplace_fn: Callable[[], str] | None = None,
    ) -> None:
        self._get_cfg = get_cfg
        self._get_bot = get_bot
        self._db_path = Path(db_path) if db_path is not None else data_dir() / "events.db"
        self._state_file = Path(state_file) if state_file is not None else state_path()
        self._poll_seconds = max(1.0, float(poll_seconds))
        self._now_fn = now_fn
        self._send_fn = send_fn or send_weekly_customer_brief
        self._workplace_fn = workplace_fn
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="weekly-customer-brief",
            daemon=True,
        )
        self._thread.start()
        print("[weekly-brief] scheduler started", flush=True)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    def _workplace(self) -> str:
        if self._workplace_fn is not None:
            return str(self._workplace_fn() or "").strip().lower()
        cfg = self._get_cfg() or {}
        return str(cfg.get("workplace_type") or "").strip().lower()

    def _clock(self, tz_name: str) -> datetime:
        zone = _local_tz(tz_name)
        if self._now_fn is not None:
            now = self._now_fn()
            if now.tzinfo is None:
                return now.replace(tzinfo=zone)
            return now.astimezone(zone)
        return datetime.now(zone)

    def tick(self) -> bool:
        """Run one eligibility check. Returns True if a send was attempted."""
        if self._workplace() != "massage":
            return False
        cfg = weekly_brief_cfg(self._get_cfg())
        if not cfg["enabled"]:
            return False
        now = self._clock(cfg["tz"])
        ready = completed_period_ready(
            now,
            epoch=cfg["epoch"],
            period_days=cfg["period_days"],
            fire_at=cfg["fire_at"],
        )
        if ready is None:
            return False
        start, end, _idx = ready
        key = period_key(start, end)
        if key in load_sent_periods(self._state_file):
            return False

        venue = str(
            (self._get_cfg() or {}).get("venue")
            or (self._get_cfg() or {}).get("garage_name")
            or "Massage"
        )
        bot = self._get_bot()
        try:
            ok = bool(
                self._send_fn(
                    db_path=self._db_path,
                    venue=venue,
                    as_of=end,
                    epoch=cfg["epoch"],
                    period_days=cfg["period_days"],
                    lookback_days=cfg["lookback_days"],
                    ai_summary=cfg["ai_summary"],
                    ollama_model=cfg["ollama_model"],
                    ollama_host=cfg["ollama_host"],
                    telegram=bot,
                )
            )
        except Exception as exc:
            print(f"[weekly-brief] send error period={key}: {exc}", flush=True)
            return True

        if ok:
            mark_period_sent(self._state_file, key)
            print(f"[weekly-brief] sent period={key}", flush=True)
        else:
            print(f"[weekly-brief] skipped/failed period={key}", flush=True)
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:
                print(f"[weekly-brief] loop error: {exc}", flush=True)
            self._stop.wait(self._poll_seconds)
