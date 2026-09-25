"""Unit tests for analytics.fsm.VisitorFSM (no vision deps)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from analytics.fsm import VisitorEvent, VisitorFSM, VisitorState
from analytics.sessions import ensure_session_columns
from db import connect
from visitor_registry import ensure_visitor_meta, reset_notify_cooldowns


class VisitorFSMTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "events.db"
        self.conn = connect(self.db_path)
        ensure_session_columns(self.conn)
        ensure_visitor_meta(self.conn)
        reset_notify_cooldowns()
        self.bot = MagicMock()
        self.bot.send_alert.return_value = True
        self.t0 = 1_700_000_000.0

    def tearDown(self) -> None:
        reset_notify_cooldowns()
        self.conn.close()
        self.tmp.cleanup()

    def _fsm(self, subject: str = "visitor_fsm_1") -> VisitorFSM:
        return VisitorFSM(
            subject,
            branch_id="champei-pp-01",
            telegram=self.bot,
            db_path=self.db_path,
            conn=self.conn,
            visit_count=1,
        )

    def test_happy_path_full_service(self) -> None:
        fsm = self._fsm()
        t = self.t0
        self.assertEqual(fsm.handle(VisitorEvent.DOOR_ENTER, t), VisitorState.ENTERED_LOBBY)
        self.assertEqual(fsm.handle(VisitorEvent.BENCH_SEAT, t + 10), VisitorState.WAITING_BENCH)
        self.assertEqual(fsm.handle(VisitorEvent.SHOE_SWAP, t + 40), VisitorState.SHOE_SWAPPING)
        self.assertTrue(fsm.has_shoe_swapped)
        self.assertEqual(fsm.handle(VisitorEvent.HALLWAY_ENTER, t + 90), VisitorState.IN_SERVICE)
        self.assertIsNotNone(fsm.visit_id)
        self.assertEqual(
            fsm.handle(VisitorEvent.DOOR_EXIT, t + 90 + 3600),
            VisitorState.COMPLETED,
        )
        self.assertIsNotNone(fsm.last_completion)
        assert fsm.last_completion is not None
        self.assertFalse(fsm.last_completion["is_early_departure"])
        self.assertGreaterEqual(fsm.last_completion["duration_seconds"], 3500)
        self.assertIn("notify_guest_arrival", fsm.last_side_effects)
        self.assertIn("start_customer_visit", fsm.last_side_effects)
        self.assertIn("record_session_completion", fsm.last_side_effects)
        # Arrival notify used send_alert (vip/guest), not wait_bottleneck.
        event_types = [c.args[0] for c in self.bot.send_alert.call_args_list]
        self.assertTrue(any(e in ("vip_arrival", "guest_arrival") for e in event_types))

    def test_early_departure(self) -> None:
        fsm = self._fsm("visitor_early")
        t = self.t0
        fsm.handle(VisitorEvent.SHOE_SWAP, t)
        fsm.handle(VisitorEvent.HALLWAY_ENTER, t + 30)
        fsm.handle(VisitorEvent.DOOR_EXIT, t + 30 + 900)  # 15 minutes in service
        self.assertEqual(fsm.state, VisitorState.COMPLETED)
        assert fsm.last_completion is not None
        self.assertTrue(fsm.last_completion["is_early_departure"])
        self.assertEqual(fsm.last_completion["duration_seconds"], 900)
        early_calls = [
            c for c in self.bot.send_alert.call_args_list if c.args and c.args[0] == "early_departure"
        ]
        self.assertEqual(len(early_calls), 1)

    def test_lobby_walk_away(self) -> None:
        fsm = self._fsm("visitor_bounce")
        t = self.t0
        fsm.handle(VisitorEvent.DOOR_ENTER, t)
        fsm.handle(VisitorEvent.DOOR_EXIT, t + 30)
        self.assertEqual(fsm.state, VisitorState.WALK_AWAY)
        self.assertIn("record_walk_away", fsm.last_side_effects)
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE event_type = 'walk_away'"
        ).fetchone()
        self.assertEqual(int(row["n"]), 1)

    def test_entourage_bench_exit(self) -> None:
        fsm = self._fsm("visitor_friend")
        t = self.t0
        fsm.handle(VisitorEvent.BENCH_SEAT, t)
        # 10 minutes of waiting without shoe swap — SLA must stay quiet.
        fsm.handle(VisitorEvent.TICK, t + 700)
        self.assertFalse(fsm.sla_alerted)
        fsm.handle(VisitorEvent.DOOR_EXIT, t + 700)
        self.assertEqual(fsm.state, VisitorState.ENTOURAGE)
        self.assertNotIn("wait_bottleneck", fsm.last_side_effects)
        self.assertNotIn("record_walk_away", fsm.last_side_effects)
        self.assertNotIn("record_session_completion", fsm.last_side_effects)
        wait_calls = [
            c for c in self.bot.send_alert.call_args_list if c.args and c.args[0] == "wait_bottleneck"
        ]
        self.assertEqual(len(wait_calls), 0)

    def test_skipped_detection_recovery(self) -> None:
        fsm = self._fsm("visitor_skip_shoe")
        t = self.t0
        fsm.handle(VisitorEvent.DOOR_ENTER, t)
        # Camera missed shoe-swap; hallway still starts service.
        fsm.handle(VisitorEvent.HALLWAY_ENTER, t + 45)
        self.assertEqual(fsm.state, VisitorState.IN_SERVICE)
        self.assertIsNotNone(fsm.service_started_ts)
        self.assertIn("start_customer_visit", fsm.last_side_effects)
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM customer_visits WHERE subject_id = ?",
            ("visitor_skip_shoe",),
        ).fetchone()
        self.assertEqual(int(row["n"]), 1)


if __name__ == "__main__":
    unittest.main()
