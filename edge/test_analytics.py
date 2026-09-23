"""Daily scorecard and silent churn analytics tests."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from analytics.churn import (
    compute_churn_risks,
    format_churn_watchlist,
    send_weekly_churn,
)
from analytics.scorecard import build_daily_scorecard, send_daily_scorecard
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
    return conn


class ScorecardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "events.db"
        self.conn = _seed_db(self.db_path)
        day = "2026-09-23"
        self.conn.execute(
            "INSERT INTO customer_visits (id, subject_id, started_at) VALUES (?, ?, ?)",
            ("v1", "visitor_a", f"{day}T09:00:00"),
        )
        self.conn.execute(
            "INSERT INTO customer_visits (id, subject_id, started_at) VALUES (?, ?, ?)",
            ("v2", "visitor_a", f"{day}T11:00:00"),
        )
        self.conn.execute(
            "INSERT INTO customer_visits (id, subject_id, started_at) VALUES (?, ?, ?)",
            ("v3", "visitor_b", f"{day}T14:00:00"),
        )
        self.conn.execute(
            "INSERT INTO events (ts, event_type, branch_id) VALUES (?, ?, ?)",
            (f"{day}T10:05:00", "wait_bottleneck", "champei-pp-01"),
        )
        self.conn.commit()
        self.conn.close()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_scorecard_counts_and_format(self) -> None:
        text = build_daily_scorecard(
            db_path=self.db_path,
            branch_id="champei-pp-01",
            day="2026-09-23",
        )
        self.assertIn("[champei-pp-01] DAILY OPERATIONS SCORECARD", text)
        self.assertIn("Date: 2026-09-23", text)
        self.assertIn("• Total Visits: 3", text)
        self.assertIn("• Unique Guests: 2", text)
        self.assertIn("• Front-Desk Bottlenecks (>3m): 1", text)

    def test_send_daily_scorecard_routes_owner_event(self) -> None:
        bot = MagicMock()
        bot.send_alert.return_value = True
        ok = send_daily_scorecard(
            db_path=self.db_path,
            branch_id="champei-pp-01",
            day="2026-09-23",
            telegram=bot,
        )
        self.assertTrue(ok)
        bot.send_alert.assert_called_once()
        event_type, text = bot.send_alert.call_args[0][:2]
        self.assertEqual(event_type, "daily_scorecard")
        self.assertIn("Total Visits: 3", text)


class ChurnTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "events.db"
        self.conn = _seed_db(self.db_path)
        self.day0 = date(2026, 1, 1)
        self.conn.execute(
            "INSERT INTO customer_visits (id, subject_id, started_at) VALUES (?, ?, ?)",
            ("c1", "visitor_risk", f"{self.day0.isoformat()}T10:00:00"),
        )
        self.conn.execute(
            "INSERT INTO customer_visits (id, subject_id, started_at) VALUES (?, ?, ?)",
            ("c2", "visitor_risk", f"{(self.day0 + timedelta(days=10)).isoformat()}T10:00:00"),
        )
        # Single-visit subject — must be excluded
        self.conn.execute(
            "INSERT INTO customer_visits (id, subject_id, started_at) VALUES (?, ?, ?)",
            ("c3", "visitor_once", f"{self.day0.isoformat()}T12:00:00"),
        )
        # On-track visitor: last visit recent relative to cadence
        self.conn.execute(
            "INSERT INTO customer_visits (id, subject_id, started_at) VALUES (?, ?, ?)",
            ("c4", "visitor_ok", f"{(self.day0 + timedelta(days=20)).isoformat()}T10:00:00"),
        )
        self.conn.execute(
            "INSERT INTO customer_visits (id, subject_id, started_at) VALUES (?, ?, ?)",
            ("c5", "visitor_ok", f"{(self.day0 + timedelta(days=30)).isoformat()}T10:00:00"),
        )
        self.conn.commit()
        self.conn.close()
        set_visitor_alias(
            "visitor_risk",
            "Lok Chumteav Sophy",
            vip_tier="Gold",
            notes="Prefers Room 3",
            db_path=self.db_path,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_churn_risk_and_alias_enrichment(self) -> None:
        as_of = self.day0 + timedelta(days=35)  # last visit day10 → overdue 25
        risks = compute_churn_risks(
            db_path=self.db_path,
            threshold_multiplier=2.0,
            min_days_overdue=21,
            as_of=as_of,
        )
        ids = {r["visitor_id"] for r in risks}
        self.assertIn("visitor_risk", ids)
        self.assertNotIn("visitor_once", ids)
        self.assertNotIn("visitor_ok", ids)

        risk = next(r for r in risks if r["visitor_id"] == "visitor_risk")
        self.assertEqual(risk["days_overdue"], 25)
        self.assertEqual(risk["avg_cadence_days"], 10.0)
        self.assertTrue(risk["is_named"])
        self.assertEqual(risk["display_name"], "Lok Chumteav Sophy")

        text = format_churn_watchlist(risks, branch_id="champei-pp-01")
        self.assertIn("SILENT CHURN WATCHLIST", text)
        self.assertIn("Lok Chumteav Sophy", text)
        self.assertIn("overdue 25d", text)
        self.assertIn("avg cadence 10d", text)

    def test_empty_watchlist_copy(self) -> None:
        text = format_churn_watchlist([], branch_id="champei-pp-01")
        self.assertIn("No at-risk guests detected this cycle.", text)

    def test_send_weekly_churn_routes_owner_event(self) -> None:
        bot = MagicMock()
        bot.send_alert.return_value = True
        ok = send_weekly_churn(
            db_path=self.db_path,
            branch_id="champei-pp-01",
            as_of=self.day0 + timedelta(days=35),
            telegram=bot,
        )
        self.assertTrue(ok)
        event_type, text = bot.send_alert.call_args[0][:2]
        self.assertEqual(event_type, "silent_churn")
        self.assertIn("SILENT CHURN WATCHLIST", text)


if __name__ == "__main__":
    unittest.main()
