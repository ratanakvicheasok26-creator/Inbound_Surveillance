"""Daily scorecard, weekly customer brief, and silent churn analytics tests."""

from __future__ import annotations

import io
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from analytics.churn import (
    compute_churn_risks,
    format_churn_watchlist,
    send_weekly_churn,
)
from analytics.scorecard import build_daily_scorecard, send_daily_scorecard
from analytics.weekly_customer_brief import (
    EVENT_TYPE,
    TELEGRAM_CHUNK_LIMIT,
    build_weekly_customer_brief,
    compute_customer_activity_brief,
    format_customer_activity_brief,
    main,
    send_weekly_customer_brief,
    week_window,
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
        self.assertIn("• Completed Sessions: 0", text)
        self.assertIn("• Unique Guests: 2", text)
        self.assertIn("• Avg Session Duration: 0 mins", text)
        self.assertIn("• Lobby Bounces / Walk-Aways: 0", text)
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


class WeeklyCustomerBriefTests(unittest.TestCase):
    AS_OF = "2026-09-27"

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "events.db"
        self.conn = _seed_db(self.db_path)
        self.conn.execute(
            """
            CREATE TABLE anonymous_subjects (
                id TEXT PRIMARY KEY,
                local_track_key TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                alias TEXT
            )
            """
        )
        self.conn.commit()
        self.conn.close()
        self._seed_visits()
        set_visitor_alias("customer_01", "Sokha", db_path=self.db_path)
        self._set_anonymous_alias("visitor_02", "Dara")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _insert(
        self,
        visit_id: str,
        subject_id: str,
        started_at: str,
        zone_id: str = "entrance",
    ) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "INSERT INTO customer_visits (id, subject_id, zone_id, started_at) VALUES (?, ?, ?, ?)",
            (visit_id, subject_id, zone_id, started_at),
        )
        conn.commit()
        conn.close()

    def _set_anonymous_alias(self, subject_id: str, alias: str) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "INSERT INTO anonymous_subjects (id, local_track_key, first_seen_at, last_seen_at, alias) "
            "VALUES (?, ?, ?, ?, ?)",
            (subject_id, subject_id, "2026-01-01T00:00:00", "2026-09-27T23:00:00", alias),
        )
        conn.commit()
        conn.close()

    def _seed_visits(self) -> None:
        self._insert("z1", "customer_01", "2026-09-21T13:30:00", "entrance")
        self._insert("z2", "customer_01", "2026-09-21T13:35:00", "waiting")
        self._insert("z3", "customer_01", "2026-09-23T13:45:00", "entrance")
        self._insert("z4", "customer_01", "2026-09-27T13:10:00", "entrance")
        self._insert("d1", "visitor_02", "2026-09-21T09:00:00", "entrance")
        self._insert("d2", "visitor_02", "2026-09-22T09:30:00", "entrance")
        self._insert("d3", "visitor_02", "2026-09-23T10:00:00", "entrance")
        self._insert("g1", "customer_03", "2026-09-23T16:00:00", "entrance")
        self._insert("g2", "customer_03", "2026-09-26T16:30:00", "entrance")
        self._insert("s1", "staff_01", "2026-09-21T13:00:00", "staff_room")
        self._insert("old1", "customer_old", "2026-01-05T09:00:00", "entrance")
        self._insert("prev1", "customer_prev", "2026-09-20T10:00:00", "entrance")
        self._insert("next1", "customer_next", "2026-10-05T09:00:00", "entrance")

    def _brief(self, **kwargs):
        params = {"db_path": self.db_path, "branch_id": "champei-pp-01", "as_of": self.AS_OF}
        params.update(kwargs)
        return compute_customer_activity_brief(**params)

    def test_week_window_contains_as_of(self) -> None:
        start, end = week_window(date(2026, 9, 27))
        self.assertEqual(start.isoformat(), "2026-09-21")
        self.assertEqual(end.isoformat(), "2026-09-27")

    def test_metrics_use_one_arrival_per_customer_day(self) -> None:
        brief = self._brief()
        self.assertEqual(brief["total_visits"], 8)
        self.assertEqual(brief["unique_customers"], 3)
        self.assertEqual(
            [c["customer_id"] for c in brief["customers"]],
            ["visitor_02", "customer_01", "customer_03"],
        )
        ids = {c["customer_id"] for c in brief["customers"]}
        self.assertNotIn("staff_01", ids)
        self.assertNotIn("customer_prev", ids)
        self.assertNotIn("customer_next", ids)
        self.assertNotIn("customer_old", ids)

    def test_usual_visit_time_per_customer(self) -> None:
        customers = {c["customer_id"]: c for c in self._brief()["customers"]}
        self.assertEqual(customers["customer_01"]["usual_time"], "13:00-13:59")
        self.assertEqual(customers["customer_01"]["usual_share"], "3 of 3")
        self.assertEqual(customers["customer_01"]["week_visits"], 3)
        self.assertEqual(customers["visitor_02"]["usual_time"], "09:00-09:59")
        self.assertEqual(customers["visitor_02"]["usual_share"], "2 of 3")
        self.assertEqual(customers["customer_03"]["usual_time"], "16:00-16:59")

    def test_usual_time_ties_are_all_reported(self) -> None:
        self._insert("t1", "visitor_02", "2026-09-25T10:45:00", "entrance")
        customers = {c["customer_id"]: c for c in self._brief()["customers"]}
        self.assertEqual(
            customers["visitor_02"]["usual_time"],
            "09:00-09:59 / 10:00-10:59",
        )
        self.assertEqual(customers["visitor_02"]["usual_share"], "2 of 4")

    def test_most_frequent_customer_ties(self) -> None:
        top = self._brief()["top_customers"]
        self.assertEqual(
            {c["customer_id"] for c in top},
            {"customer_01", "visitor_02"},
        )
        self.assertTrue(all(int(c["visit_days"]) == 3 for c in top))

    def test_busiest_day(self) -> None:
        self.assertEqual(
            self._brief()["busiest_days"],
            [{"date": "2026-09-23", "weekday": "Wednesday", "visits": 3}],
        )

    def test_busiest_day_ties(self) -> None:
        self._insert("x1", "customer_04", "2026-09-21T10:00:00", "entrance")
        self._insert("x2", "customer_04", "2026-09-22T10:00:00", "entrance")
        busiest = self._brief()["busiest_days"]
        self.assertEqual([d["date"] for d in busiest], ["2026-09-21", "2026-09-23"])
        self.assertTrue(all(int(d["visits"]) == 3 for d in busiest))

    def test_alias_sources_and_precedence(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "INSERT INTO anonymous_subjects (id, local_track_key, first_seen_at, last_seen_at, alias) "
            "VALUES ('customer_01', 'customer_01', '2026-01-01T00:00:00', '2026-09-27T23:00:00', 'Shadow')"
        )
        conn.commit()
        conn.close()
        customers = {c["customer_id"]: c for c in self._brief()["customers"]}
        self.assertEqual(customers["customer_01"]["display_name"], "Sokha")
        self.assertTrue(customers["customer_01"]["is_named"])
        self.assertEqual(customers["visitor_02"]["display_name"], "Dara")
        self.assertEqual(customers["customer_03"]["display_name"], "customer_03")
        self.assertFalse(customers["customer_03"]["is_named"])

    def test_formatted_brief_text(self) -> None:
        text = format_customer_activity_brief(self._brief())
        self.assertIn("[champei-pp-01] WEEKLY CUSTOMER ACTIVITY BRIEF", text)
        self.assertIn("Week: 2026-09-21 to 2026-09-27", text)
        self.assertIn("- Customer visits this week: 8", text)
        self.assertIn("- Unique customers: 3", text)
        self.assertIn("Most frequent customer", text)
        self.assertIn("Sokha (3 visit days)", text)
        self.assertIn("Dara (3 visit days)", text)
        self.assertIn("Busiest day: Wednesday 2026-09-23 (3 visits)", text)
        self.assertIn("Usual visit times", text)
        self.assertIn("- Sokha - 13:00-13:59 (3 of 3 visit days)", text)
        self.assertIn("- Dara - 09:00-09:59 (2 of 3 visit days)", text)
        self.assertNotIn("staff_01", text)
        self.assertNotIn("customer_old", text)

    def test_build_helper_matches_formatter(self) -> None:
        self.assertEqual(
            build_weekly_customer_brief(db_path=self.db_path, as_of=self.AS_OF),
            format_customer_activity_brief(self._brief()),
        )

    def test_empty_database(self) -> None:
        with tempfile.TemporaryDirectory() as empty_dir:
            empty_db = Path(empty_dir) / "events.db"
            brief = compute_customer_activity_brief(
                db_path=empty_db,
                as_of=self.AS_OF,
            )
        self.assertEqual(brief["total_visits"], 0)
        self.assertEqual(brief["customers"], [])
        self.assertEqual(brief["busiest_days"], [])
        self.assertEqual(brief["top_customers"], [])
        self.assertIn(
            "No customer visits recorded this week.",
            format_customer_activity_brief(brief),
        )

    def test_partial_schema_is_tolerated(self) -> None:
        with tempfile.TemporaryDirectory() as partial_dir:
            partial_db = Path(partial_dir) / "events.db"
            conn = sqlite3.connect(str(partial_db))
            conn.execute("CREATE TABLE customer_visits (id TEXT PRIMARY KEY)")
            conn.commit()
            conn.close()
            brief = compute_customer_activity_brief(db_path=partial_db, as_of=self.AS_OF)
        self.assertEqual(brief["total_visits"], 0)

    def test_send_routes_owner_event(self) -> None:
        bot = MagicMock()
        bot.send_alert.return_value = True
        ok = send_weekly_customer_brief(
            db_path=self.db_path,
            branch_id="champei-pp-01",
            as_of=self.AS_OF,
            telegram=bot,
        )
        self.assertTrue(ok)
        bot.send_alert.assert_called_once()
        event_type, text = bot.send_alert.call_args[0][:2]
        self.assertEqual(event_type, EVENT_TYPE)
        self.assertIn("WEEKLY CUSTOMER ACTIVITY BRIEF", text)

    def test_send_fails_when_dispatch_fails(self) -> None:
        bot = MagicMock()
        bot.send_alert.return_value = False
        self.assertFalse(
            send_weekly_customer_brief(
                db_path=self.db_path,
                as_of=self.AS_OF,
                telegram=bot,
            )
        )

    def test_long_brief_is_split_into_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as busy_dir:
            busy_db = Path(busy_dir) / "events.db"
            conn = _seed_db(busy_db)
            for index in range(200):
                subject = f"customer_bulk_{index:03d}"
                conn.execute(
                    "INSERT INTO customer_visits (id, subject_id, started_at) VALUES (?, ?, ?)",
                    (f"bulk{index}", subject, "2026-09-25T15:30:00"),
                )
            conn.commit()
            conn.close()
            bot = MagicMock()
            bot.send_alert.return_value = True
            ok = send_weekly_customer_brief(
                db_path=busy_db,
                as_of=self.AS_OF,
                telegram=bot,
            )
        self.assertTrue(ok)
        self.assertGreater(bot.send_alert.call_count, 1)
        for call in bot.send_alert.call_args_list:
            event_type, chunk = call.args[0], call.args[1]
            self.assertEqual(event_type, EVENT_TYPE)
            self.assertLessEqual(len(chunk), TELEGRAM_CHUNK_LIMIT + 40)
        self.assertIn("(continued 2/", bot.send_alert.call_args_list[1].args[1])

    def test_cli_dry_run_prints_without_sending(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer), patch(
            "analytics.weekly_customer_brief.TelegramOut"
        ) as bot_cls:
            code = main(
                [
                    "--dry-run",
                    "--db",
                    str(self.db_path),
                    "--branch",
                    "champei-pp-01",
                    "--as-of",
                    self.AS_OF,
                ]
            )
        self.assertEqual(code, 0)
        bot_cls.assert_not_called()
        self.assertIn("WEEKLY CUSTOMER ACTIVITY BRIEF", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
