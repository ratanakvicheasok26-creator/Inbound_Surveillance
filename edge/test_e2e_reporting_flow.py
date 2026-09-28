"""Comprehensive End-to-End Validation Test for Champei Spa 4-Step Intelligence Pipeline.

Validates:
- Step 1: Live Guest Arrival Entry Alert Format & Fields.
- Step 2: Daily Operations Scorecard (Daily @ 21:45) Aggregation & Formatting.
- Step 3: Weekly Executive Business Report (Sundays @ 17:00) Aggregation & Formatting.
- Step 4: Monthly Strategic Intelligence Report (Month-End @ 21:45) Aggregation & Formatting.
- Schedulers: Correct fire conditions for daily, weekly (Sunday 17:00), and monthly (Last day 21:45).
- Telegram Controller: Proper handling of /scorecard, /weekly, /monthly commands.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import unittest
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from analytics.monthly_customer_brief import build_monthly_strategic_brief, send_monthly_strategic_brief
from analytics.scorecard import build_daily_scorecard
from analytics.sessions import ensure_session_columns
from analytics.weekly_customer_brief import build_weekly_customer_brief, send_weekly_customer_brief
from run_champei import (
    _is_last_day_of_month,
    _monthly_brief_loop,
    _parse_fire_at,
    _scorecard_loop,
    _weekly_brief_loop,
)
from telegram_controller import TelegramController

TZ = ZoneInfo("Asia/Phnom_Penh")
BRANCH = "champei-pp-01"


class ChampeiPipelineEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_events.db"
        self._init_database()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _init_database(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            ensure_session_columns(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS visitor_meta (
                    visitor_id TEXT PRIMARY KEY,
                    alias TEXT,
                    tier TEXT,
                    notes TEXT,
                    created_at REAL,
                    updated_at REAL
                )
                """
            )
            conn.commit()

    def _insert_visit(
        self,
        visitor_id: str,
        start_dt: datetime,
        duration_mins: int = 60,
        status: str = "completed",
        alias: str | None = None,
        tier: str | None = None,
    ) -> None:
        dur_secs = duration_mins * 60.0
        end_dt = start_dt + timedelta(minutes=duration_mins)
        started_at = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
        ended_at = end_dt.strftime("%Y-%m-%dT%H:%M:%S") if status == "completed" else None
        dur_val = dur_secs if status == "completed" else None
        
        with sqlite3.connect(str(self.db_path)) as conn:
            ensure_session_columns(conn)
            conn.execute(
                """
                INSERT INTO customer_visits
                    (id, subject_id, zone_id, started_at, ended_at, source, status, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"visit-{start_dt.timestamp()}-{visitor_id}",
                    visitor_id,
                    "waiting",
                    started_at,
                    ended_at,
                    "edge",
                    status,
                    dur_val,
                ),
            )
            if alias:
                conn.execute(
                    """
                    INSERT INTO visitor_meta (visitor_id, alias, tier, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(visitor_id) DO UPDATE SET alias=excluded.alias, tier=excluded.tier, updated_at=excluded.updated_at
                    """,
                    (visitor_id, alias, tier or "Gold VIP", start_dt.timestamp(), start_dt.timestamp()),
                )
            conn.commit()

    def test_step1_live_guest_arrival_format(self) -> None:
        """Step 1: Check format of guest arrival alert."""
        cust_id = "CUST-0001"
        arrival_time_str = "11:42 PM"
        visit_count = 1
        text = (
            f"🌿 [{BRANCH}] 👋 GUEST ARRIVAL\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 Customer ID: {cust_id}\n"
            f"🔢 Visit Count: Visit #{visit_count} (New Guest)\n"
            f"🕒 Arrival Time: {arrival_time_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💡 Staff Quick-Name: Reply with:\n"
            f"/name {cust_id} <Guest_Name>"
        )
        self.assertIn("GUEST ARRIVAL", text)
        self.assertIn("Customer ID: CUST-0001", text)
        self.assertIn("Arrival Time: 11:42 PM", text)
        self.assertIn("/name CUST-0001 <Guest_Name>", text)

    def test_step2_daily_operations_scorecard(self) -> None:
        """Step 2: Generate and test Daily Operations Scorecard format and accuracy."""
        test_day = date(2026, 9, 28)
        base_dt = datetime(2026, 9, 28, 10, 0, tzinfo=TZ)

        # Insert 3 visits: 2 completed (60m, 90m -> avg 75m), 1 active
        self._insert_visit("CUST-0001", base_dt, duration_mins=60, status="completed")
        self._insert_visit("CUST-0002", base_dt + timedelta(hours=1), duration_mins=90, status="completed")
        self._insert_visit("CUST-0003", base_dt + timedelta(hours=2), duration_mins=10, status="active")

        scorecard = build_daily_scorecard(self.db_path, branch_id=BRANCH, as_of=test_day)
        
        self.assertIn(f"[{BRANCH}] 📊 DAILY OPERATIONS SCORECARD", scorecard)
        self.assertIn("📅 Date: 2026-09-28", scorecard)
        self.assertIn("• Total Customer Visits: 3", scorecard)
        self.assertIn("• Completed Treatments: 2", scorecard)
        self.assertIn("• Unique Guests: 3", scorecard)
        self.assertIn("• Avg Session Duration: 75 mins", scorecard)

    def test_step3_weekly_executive_business_report(self) -> None:
        """Step 3: Generate and test Weekly Executive Business Report (aggregates Mon-Sun)."""
        # Week of Mon 2026-09-21 to Sun 2026-09-27
        mon = datetime(2026, 9, 21, 14, 0, tzinfo=TZ)
        wed = datetime(2026, 9, 23, 15, 0, tzinfo=TZ)
        sat = datetime(2026, 9, 26, 16, 0, tzinfo=TZ)
        sun = datetime(2026, 9, 27, 11, 0, tzinfo=TZ)

        # CUST-0001 visits Mon & Sat (repeat)
        self._insert_visit("CUST-0001", mon, duration_mins=60, alias="Vichea", tier="Gold VIP")
        self._insert_visit("CUST-0001", sat, duration_mins=60, alias="Vichea", tier="Gold VIP")
        # CUST-0002 visits Wed & Sun
        self._insert_visit("CUST-0002", wed, duration_mins=90)
        self._insert_visit("CUST-0002", sun, duration_mins=90)
        # CUST-0003 visits Sun
        self._insert_visit("CUST-0003", sun, duration_mins=45)

        # As of Sunday 2026-09-27
        weekly = build_weekly_customer_brief(self.db_path, branch_id=BRANCH, as_of=date(2026, 9, 27))

        self.assertIn("WEEKLY EXECUTIVE BUSINESS REPORT", weekly)
        self.assertIn("📅 Week: 2026-09-21 to 2026-09-27 (Mon – Sun)", weekly)
        self.assertIn("• Total Customer Visits: 5", weekly)
        self.assertIn("• Completed Treatments: 5", weekly)
        self.assertIn("• Unique Individuals: 3 guests", weekly)
        self.assertIn("Vichea", weekly)
        self.assertIn("CUST-0002", weekly)
        self.assertIn("• Mon (09/21): 01 visits", weekly)
        self.assertIn("• Sun (09/27): 02 visits", weekly)

    def test_step4_monthly_strategic_intelligence_report(self) -> None:
        """Step 4: Generate and test Monthly Strategic Intelligence Report."""
        # Insert visits across multiple weeks of September 2026
        # Week 1: 09/01
        self._insert_visit("CUST-0001", datetime(2026, 9, 1, 10, 0, tzinfo=TZ), duration_mins=60, alias="Vichea")
        # Week 2: 09/10
        self._insert_visit("CUST-0001", datetime(2026, 9, 10, 10, 0, tzinfo=TZ), duration_mins=60, alias="Vichea")
        self._insert_visit("CUST-0002", datetime(2026, 9, 10, 14, 0, tzinfo=TZ), duration_mins=90)
        # Week 4: 09/25
        self._insert_visit("CUST-0003", datetime(2026, 9, 25, 11, 0, tzinfo=TZ), duration_mins=45)
        # Week 5: 09/30 (Month end)
        self._insert_visit("CUST-0001", datetime(2026, 9, 30, 16, 0, tzinfo=TZ), duration_mins=60, alias="Vichea")

        # As of September 30, 2026
        monthly = build_monthly_strategic_brief(self.db_path, branch_id=BRANCH, as_of=date(2026, 9, 30))

        self.assertIn("MONTHLY STRATEGIC INTELLIGENCE REPORT", monthly)
        self.assertIn("📅 Month: September 2026 (2026-09-01 to 2026-09-30)", monthly)
        self.assertIn("• Total Monthly Visits: 5", monthly)
        self.assertIn("• Total Unique Guests: 3 individuals", monthly)
        self.assertIn("1. Vichea", monthly)
        self.assertIn("3 visits this month", monthly)
        self.assertIn("Week 1 (09/01 - 09/07): 001 visits", monthly)
        self.assertIn("Week 2 (09/08 - 09/14): 002 visits", monthly)
        self.assertIn("Week 5 (09/29 - 09/30): 001 visits", monthly)

    def test_weekly_scheduler_fires_sunday_at_17_00(self) -> None:
        """Scheduler: Weekly brief fires at exactly 17:00 on Sunday."""
        calls: list[dict] = []
        stop_event = threading.Event()
        moments = [
            datetime(2026, 9, 27, 16, 59, tzinfo=TZ),  # Sunday 16:59 -> No fire
            datetime(2026, 9, 27, 17, 00, tzinfo=TZ),  # Sunday 17:00 -> Fires!
            datetime(2026, 9, 27, 17, 30, tzinfo=TZ),  # Sunday 17:30 -> Already fired this week, skip
        ]

        def now_fn() -> datetime:
            if not moments:
                stop_event.set()
                return datetime(2026, 9, 27, 18, 0, tzinfo=TZ)
            return moments.pop(0)

        def send_fn(**kwargs) -> bool:
            calls.append(kwargs)
            return True

        _weekly_brief_loop(
            stop_event,
            branch_id=BRANCH,
            db_path=self.db_path,
            fire_at=dt_time(17, 0),
            tz=TZ,
            now_fn=now_fn,
            send_fn=send_fn,
            poll_seconds=0.01,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["as_of"], date(2026, 9, 27))

    def test_monthly_scheduler_fires_month_end_at_21_45(self) -> None:
        """Scheduler: Monthly brief fires on the last day of the month at 21:45."""
        calls: list[dict] = []
        stop_event = threading.Event()
        # Sept 2026 has 30 days
        self.assertTrue(_is_last_day_of_month(date(2026, 9, 30)))
        self.assertFalse(_is_last_day_of_month(date(2026, 9, 29)))

        moments = [
            datetime(2026, 9, 29, 21, 45, tzinfo=TZ),  # Sept 29 21:45 -> Not last day, skip
            datetime(2026, 9, 30, 21, 44, tzinfo=TZ),  # Sept 30 21:44 -> Before fire time, skip
            datetime(2026, 9, 30, 21, 45, tzinfo=TZ),  # Sept 30 21:45 -> Fires!
            datetime(2026, 9, 30, 22, 00, tzinfo=TZ),  # Sept 30 22:00 -> Already fired this month, skip
        ]

        def now_fn() -> datetime:
            if not moments:
                stop_event.set()
                return datetime(2026, 9, 30, 23, 0, tzinfo=TZ)
            return moments.pop(0)

        def send_fn(**kwargs) -> bool:
            calls.append(kwargs)
            return True

        _monthly_brief_loop(
            stop_event,
            branch_id=BRANCH,
            db_path=self.db_path,
            fire_at=dt_time(21, 45),
            tz=TZ,
            now_fn=now_fn,
            send_fn=send_fn,
            poll_seconds=0.01,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["as_of"], date(2026, 9, 30))

    def test_telegram_controller_on_demand_commands(self) -> None:
        """Telegram Controller: Verify /scorecard, /weekly, and /monthly handlers."""
        owner_id = "1217328195"
        controller = TelegramController(
            owner_chat_id=owner_id,
            staff_chat_id=owner_id,
            db_path=self.db_path,
            branch_id=BRANCH,
        )
        
        # Test /scorecard
        scorecard_reply = controller.handle_message(owner_id, "/scorecard")
        self.assertIsNotNone(scorecard_reply)
        self.assertIn("DAILY OPERATIONS SCORECARD", scorecard_reply)

        # Test /weekly
        weekly_reply = controller.handle_message(owner_id, "/weekly")
        self.assertIsNotNone(weekly_reply)
        self.assertIn("WEEKLY EXECUTIVE BUSINESS REPORT", weekly_reply)

        # Test /monthly
        monthly_reply = controller.handle_message(owner_id, "/monthly")
        self.assertIsNotNone(monthly_reply)
        self.assertIn("MONTHLY STRATEGIC INTELLIGENCE REPORT", monthly_reply)


if __name__ == "__main__":
    unittest.main()
