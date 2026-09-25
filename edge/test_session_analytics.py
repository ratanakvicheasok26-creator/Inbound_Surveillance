"""Session completion, early departure, and walk-away analytics tests."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from analytics.scorecard import build_daily_scorecard
from analytics.sessions import (
    ensure_session_columns,
    record_session_completion,
    record_walk_away,
)
from visitor_registry import set_visitor_alias


def _seed_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE customer_visits (
            id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL,
            zone_id TEXT NOT NULL DEFAULT 'waiting',
            started_at TEXT NOT NULL,
            ended_at TEXT,
            source TEXT NOT NULL DEFAULT 'edge'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            event_type TEXT NOT NULL,
            abs_path TEXT,
            branch_id TEXT,
            camera_role TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE visitor_meta (
            visitor_id TEXT PRIMARY KEY,
            alias TEXT,
            vip_tier TEXT DEFAULT 'Standard',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    ensure_session_columns(conn)
    return conn


class SessionAnalyticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "events.db"
        self.conn = _seed_db(self.db_path)
        self.conn.close()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _insert_open(self, visit_id: str, subject_id: str, started_at: str) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "INSERT INTO customer_visits (id, subject_id, started_at) VALUES (?, ?, ?)",
            (visit_id, subject_id, started_at),
        )
        conn.commit()
        conn.close()

    def test_session_completion_writes_duration_and_status(self) -> None:
        self._insert_open("s1", "visitor_done", "2026-09-23T10:00:00")
        result = record_session_completion(
            "visitor_done",
            ended_at="2026-09-23T11:05:00",
            db_path=self.db_path,
            telegram=MagicMock(),
        )
        self.assertEqual(result["duration_seconds"], 3900)
        self.assertFalse(result["is_early_departure"])

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT ended_at, duration_seconds, status FROM customer_visits WHERE id = 's1'"
        ).fetchone()
        conn.close()
        self.assertEqual(row["ended_at"], "2026-09-23T11:05:00")
        self.assertEqual(int(row["duration_seconds"]), 3900)
        self.assertEqual(row["status"], "completed")

    def test_early_departure_triggers_alert_with_alias(self) -> None:
        self._insert_open("s2", "visitor_early", "2026-09-23T10:00:00")
        set_visitor_alias(
            "visitor_early",
            "Lok Chumteav Sophy",
            vip_tier="Gold",
            db_path=self.db_path,
        )
        bot = MagicMock()
        bot.send_alert.return_value = True
        result = record_session_completion(
            "visitor_early",
            ended_at="2026-09-23T10:15:00",  # 15 minutes
            db_path=self.db_path,
            branch_id="champei-pp-01",
            telegram=bot,
        )
        self.assertEqual(result["duration_seconds"], 900)
        self.assertTrue(result["is_early_departure"])
        bot.send_alert.assert_called_once()
        event_type, text = bot.send_alert.call_args[0][:2]
        self.assertEqual(event_type, "early_departure")
        self.assertIn("EARLY DEPARTURE ALERT", text)
        self.assertIn("Lok Chumteav Sophy", text)
        self.assertIn("Duration: 15 mins", text)

    def test_long_session_no_early_alert(self) -> None:
        self._insert_open("s3", "visitor_long", "2026-09-23T09:00:00")
        bot = MagicMock()
        result = record_session_completion(
            "visitor_long",
            ended_at="2026-09-23T10:05:00",  # 65 minutes
            db_path=self.db_path,
            telegram=bot,
        )
        self.assertEqual(result["duration_seconds"], 3900)
        self.assertFalse(result["is_early_departure"])
        bot.send_alert.assert_not_called()

    def test_walk_away_increments_scorecard(self) -> None:
        day = "2026-09-23"
        self._insert_open("s4", "visitor_bounce", f"{day}T08:00:00")
        ok = record_walk_away(
            branch_id="champei-pp-01",
            ts=f"{day}T08:10:00",
            db_path=self.db_path,
        )
        self.assertTrue(ok)
        text = build_daily_scorecard(
            db_path=self.db_path,
            branch_id="champei-pp-01",
            day=day,
        )
        self.assertIn("• Lobby Bounces / Walk-Aways: 1", text)
        self.assertIn("• Total Visits: 1", text)


class TelegramEarlyDepartureRoutingTests(unittest.TestCase):
    def test_early_departure_in_both_event_sets(self) -> None:
        from telegram_out import OWNER_EVENTS, STAFF_EVENTS

        self.assertIn("early_departure", STAFF_EVENTS)
        self.assertIn("early_departure", OWNER_EVENTS)


if __name__ == "__main__":
    unittest.main()
