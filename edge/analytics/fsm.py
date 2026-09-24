"""Pure-Python visitor finite state machine for Champei Spa.

Vision layers emit VisitorEvent values; this module owns transitions and
hospitality side effects (arrival notify, visit start, walk-away, completion,
wait SLA). No OpenCV / torch / workplace imports.
"""

from __future__ import annotations

import secrets
import sqlite3
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from analytics.sessions import record_session_completion, record_walk_away
from db import connect as db_connect
from db import start_customer_visit
from telegram_out import TelegramOut, format_alert_header
from visitor_registry import notify_guest_arrival

WAIT_SLA_SECONDS = 180.0


class VisitorState(str, Enum):
    ENTERED_LOBBY = "ENTERED_LOBBY"
    WAITING_BENCH = "WAITING_BENCH"
    SHOE_SWAPPING = "SHOE_SWAPPING"
    IN_SERVICE = "IN_SERVICE"
    COMPLETED = "COMPLETED"
    WALK_AWAY = "WALK_AWAY"
    ENTOURAGE = "ENTOURAGE"


class VisitorEvent(str, Enum):
    DOOR_ENTER = "DOOR_ENTER"
    BENCH_SEAT = "BENCH_SEAT"
    SHOE_SWAP = "SHOE_SWAP"
    HALLWAY_ENTER = "HALLWAY_ENTER"
    DOOR_EXIT = "DOOR_EXIT"
    TICK = "TICK"


def _iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(float(ts)).isoformat(timespec="seconds")


class VisitorFSM:
    """Per-subject visitor journey with entourage filter and skipped-detection recovery."""

    def __init__(
        self,
        subject_id: str,
        branch_id: str = "champei-pp-01",
        *,
        telegram: TelegramOut | None = None,
        db_path: Path | str | None = None,
        conn: sqlite3.Connection | None = None,
        visit_count: int = 1,
        wait_sla_seconds: float = WAIT_SLA_SECONDS,
    ) -> None:
        self.subject_id = str(subject_id or "").strip()
        self.branch_id = str(branch_id or "champei-pp-01").strip() or "champei-pp-01"
        self.state: VisitorState | None = None
        self.entry_ts: float | None = None
        self.bench_ts: float | None = None
        self.has_shoe_swapped: bool = False
        self.sla_alerted: bool = False
        self.service_started_ts: float | None = None
        self.visit_id: str | None = None

        self._telegram = telegram
        self._db_path = Path(db_path) if db_path is not None else None
        self._conn = conn
        self._visit_count = max(1, int(visit_count or 1))
        self._wait_sla_seconds = float(wait_sla_seconds)

        self.last_completion: dict[str, Any] | None = None
        self.last_side_effects: list[str] = []

    def handle(self, event: VisitorEvent | str, ts: float) -> VisitorState | None:
        ev = event if isinstance(event, VisitorEvent) else VisitorEvent(str(event))
        t = float(ts)

        if ev is VisitorEvent.DOOR_ENTER:
            self._on_door_enter(t)
        elif ev is VisitorEvent.BENCH_SEAT:
            self._on_bench_seat(t)
        elif ev is VisitorEvent.SHOE_SWAP:
            self._on_shoe_swap(t)
        elif ev is VisitorEvent.HALLWAY_ENTER:
            self._on_hallway_enter(t)
        elif ev is VisitorEvent.DOOR_EXIT:
            self._on_door_exit(t)
        elif ev is VisitorEvent.TICK:
            self._on_tick(t)
        return self.state

    def _on_door_enter(self, ts: float) -> None:
        if self.state is not None and self.state not in (
            VisitorState.COMPLETED,
            VisitorState.WALK_AWAY,
            VisitorState.ENTOURAGE,
        ):
            return
        self.state = VisitorState.ENTERED_LOBBY
        self.entry_ts = ts
        self.bench_ts = None
        self.has_shoe_swapped = False
        self.sla_alerted = False
        self.service_started_ts = None
        self.visit_id = None
        self.last_completion = None

    def _on_bench_seat(self, ts: float) -> None:
        if self.state is None or self.state is VisitorState.ENTERED_LOBBY:
            self.state = VisitorState.WAITING_BENCH
            self.bench_ts = ts
            if self.entry_ts is None:
                self.entry_ts = ts

    def _on_shoe_swap(self, ts: float) -> None:
        if self.state not in (VisitorState.ENTERED_LOBBY, VisitorState.WAITING_BENCH, None):
            if self.state is VisitorState.SHOE_SWAPPING:
                self.has_shoe_swapped = True
            return
        if self.state is None:
            self.entry_ts = ts
        self.state = VisitorState.SHOE_SWAPPING
        self.has_shoe_swapped = True
        self._emit("notify_guest_arrival")
        notify_guest_arrival(
            self.subject_id,
            self._visit_count,
            branch_id=self.branch_id,
            zone="shoe_lounge",
            telegram=self._telegram,
            conn=self._conn,
            db_path=self._db_path,
            now=ts,
        )

    def _on_hallway_enter(self, ts: float) -> None:
        if self.state in (
            VisitorState.IN_SERVICE,
            VisitorState.COMPLETED,
            VisitorState.WALK_AWAY,
            VisitorState.ENTOURAGE,
        ):
            return
        # Fault tolerance: promote from lobby/bench/shoe (or cold start) even if
        # shoe-swap posture was missed by the camera.
        if self.state is None:
            self.entry_ts = ts
        self.state = VisitorState.IN_SERVICE
        self.service_started_ts = ts
        if not self.visit_id:
            self.visit_id = f"visit_{secrets.token_hex(6)}"
        self._start_visit(ts)

    def _on_door_exit(self, ts: float) -> None:
        if self.state is VisitorState.ENTERED_LOBBY:
            self.state = VisitorState.WALK_AWAY
            self._emit("record_walk_away")
            record_walk_away(
                branch_id=self.branch_id,
                ts=_iso_from_ts(ts),
                db_path=self._db_path,
            )
            return

        if self.state is VisitorState.WAITING_BENCH:
            if not self.has_shoe_swapped:
                self.state = VisitorState.ENTOURAGE
                self._emit("entourage")
                return
            self._complete_session(ts)
            return

        if self.state is VisitorState.SHOE_SWAPPING:
            self._complete_session(ts)
            return

        if self.state is VisitorState.IN_SERVICE:
            self._complete_session(ts)
            return

    def _on_tick(self, ts: float) -> None:
        if self.state is not VisitorState.WAITING_BENCH:
            return
        if self.bench_ts is None or self.sla_alerted:
            return
        if not self.has_shoe_swapped:
            return
        if (ts - self.bench_ts) <= self._wait_sla_seconds:
            return
        self.sla_alerted = True
        text = (
            f"{format_alert_header(self.branch_id, 'WAIT BOTTLENECK')}: "
            f"Guest waiting >= {int(self._wait_sla_seconds)}s at Front Desk "
            f"(subject={self.subject_id}). Action: Greet / assist."
        )
        bot = self._telegram if self._telegram is not None else TelegramOut()
        self._emit("wait_bottleneck")
        try:
            bot.send_alert("wait_bottleneck", text)
        except Exception as exc:
            print(f"[fsm] wait_bottleneck send failed: {exc}", flush=True)

    def _start_visit(self, ts: float) -> None:
        own = self._conn is None
        conn = self._conn
        if conn is None:
            if self._db_path is None:
                return
            conn = db_connect(self._db_path)
        try:
            assert self.visit_id is not None
            self._emit("start_customer_visit")
            start_customer_visit(
                conn,
                self.visit_id,
                self.subject_id,
                "hallway",
                datetime.fromtimestamp(ts),
            )
        finally:
            if own and conn is not None:
                conn.close()

    def _complete_session(self, ts: float) -> None:
        self.state = VisitorState.COMPLETED
        self._emit("record_session_completion")
        self.last_completion = record_session_completion(
            self.subject_id,
            ended_at=_iso_from_ts(ts),
            db_path=self._db_path,
            branch_id=self.branch_id,
            telegram=self._telegram,
        )

    def _emit(self, name: str) -> None:
        self.last_side_effects.append(name)
