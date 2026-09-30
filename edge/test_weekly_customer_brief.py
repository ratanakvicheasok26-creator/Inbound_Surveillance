"""Tests for massage weekly customer brief + epoch scheduler."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, time as dt_time
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

from db import connect
from weekly_brief_scheduler import (
    WeeklyCustomerBriefScheduler,
    completed_period_ready,
    load_sent_periods,
    parse_fire_at,
    weekly_brief_cfg,
)
from weekly_customer_brief import (
    compute_customer_activity_brief,
    format_customer_activity_brief,
    generate_ai_summary,
    period_window,
    send_weekly_customer_brief,
)

TZ = ZoneInfo("Asia/Phnom_Penh")
EPOCH = date(2026, 9, 25)


def _moment(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=TZ)


class PeriodWindowTests(unittest.TestCase):
    def test_first_seven_days_from_sept_25(self) -> None:
        window = period_window(date(2026, 9, 25), epoch=EPOCH)
        self.assertEqual(window, (date(2026, 9, 25), date(2026, 10, 1), 0))
        window = period_window(date(2026, 10, 1), epoch=EPOCH)
        self.assertEqual(window, (date(2026, 9, 25), date(2026, 10, 1), 0))

    def test_second_period_starts_oct_2(self) -> None:
        window = period_window(date(2026, 10, 2), epoch=EPOCH)
        self.assertEqual(window, (date(2026, 10, 2), date(2026, 10, 8), 1))

    def test_before_epoch_returns_none(self) -> None:
        self.assertIsNone(period_window(date(2026, 9, 24), epoch=EPOCH))


class CompletedPeriodReadyTests(unittest.TestCase):
    def test_not_ready_mid_period(self) -> None:
        self.assertIsNone(
            completed_period_ready(
                _moment("2026-09-28T22:00:00"),
                epoch=EPOCH,
                fire_at=dt_time(21, 0),
            )
        )

    def test_ready_on_last_day_after_fire_at(self) -> None:
        ready = completed_period_ready(
            _moment("2026-10-01T21:00:00"),
            epoch=EPOCH,
            fire_at=dt_time(21, 0),
        )
        self.assertEqual(ready, (date(2026, 9, 25), date(2026, 10, 1), 0))

    def test_not_ready_on_last_day_before_fire_at(self) -> None:
        self.assertIsNone(
            completed_period_ready(
                _moment("2026-10-01T20:59:00"),
                epoch=EPOCH,
                fire_at=dt_time(21, 0),
            )
        )

    def test_catch_up_next_day_sends_previous_period(self) -> None:
        ready = completed_period_ready(
            _moment("2026-10-02T10:00:00"),
            epoch=EPOCH,
            fire_at=dt_time(21, 0),
        )
        # Oct 2 is inside period 1, so catch-up should surface period 0.
        self.assertEqual(ready, (date(2026, 9, 25), date(2026, 10, 1), 0))


class BriefComputeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "events.db"
        self.conn = connect(self.db_path)
        rows = [
            ("v1", "visitor_aaa", "2026-09-25T10:00:00"),
            ("v2", "visitor_aaa", "2026-09-26T10:15:00"),
            ("v3", "visitor_aaa", "2026-09-27T09:45:00"),
            ("v4", "visitor_bbb", "2026-09-25T18:00:00"),
            ("v5", "visitor_bbb", "2026-09-25T18:30:00"),  # same day second zone
            ("v6", "staff_xyz", "2026-09-25T11:00:00"),
            ("v7", "visitor_ccc", "2026-10-03T12:00:00"),  # next period
        ]
        for vid, subject, started in rows:
            self.conn.execute(
                "INSERT INTO customer_visits (id, subject_id, zone_id, started_at) VALUES (?, ?, 'entrance', ?)",
                (vid, subject, started),
            )
        self.conn.execute(
            """
            INSERT INTO anonymous_subjects (id, local_track_key, first_seen_at, last_seen_at, alias)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "visitor_aaa",
                "track_aaa",
                "2026-09-25T10:00:00",
                "2026-09-27T09:45:00",
                "Regular Guest",
            ),
        )
        self.conn.commit()

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def test_compute_counts_unique_and_top_visitor(self) -> None:
        brief = compute_customer_activity_brief(
            db_path=self.db_path,
            venue="Champei",
            as_of=date(2026, 10, 1),
            epoch=EPOCH,
        )
        self.assertEqual(brief["unique_customers"], 2)
        # visitor_aaa: 3 canonical days; visitor_bbb: 1 (two same-day zone rows collapse)
        self.assertEqual(brief["total_visits"], 4)
        top_ids = {c["customer_id"] for c in brief["top_customers"]}
        self.assertEqual(top_ids, {"visitor_aaa"})
        self.assertEqual(brief["top_customers"][0]["display_name"], "Regular Guest")
        self.assertNotIn("staff_xyz", {c["customer_id"] for c in brief["customers"]})
        self.assertNotIn("visitor_ccc", {c["customer_id"] for c in brief["customers"]})

    def test_format_includes_unique_id(self) -> None:
        brief = compute_customer_activity_brief(
            db_path=self.db_path,
            venue="Champei",
            as_of=date(2026, 10, 1),
            epoch=EPOCH,
        )
        text = format_customer_activity_brief(brief)
        self.assertIn("visitor_aaa", text)
        self.assertIn("WEEKLY CUSTOMER ACTIVITY BRIEF", text)

    def test_ai_summary_uses_ollama_response(self) -> None:
        brief = compute_customer_activity_brief(
            db_path=self.db_path,
            as_of=date(2026, 10, 1),
            epoch=EPOCH,
        )

        class _Resp:
            ok = True

            def json(self):
                return {"response": "Top guest visitor_aaa led the week."}

        summary = generate_ai_summary(
            brief,
            enabled=True,
            request_fn=lambda *a, **k: _Resp(),
        )
        self.assertIn("visitor_aaa", summary)

    def test_send_uses_telegram_out(self) -> None:
        bot = MagicMock()
        bot.send_message.return_value = True
        ok = send_weekly_customer_brief(
            db_path=self.db_path,
            venue="Champei",
            as_of=date(2026, 10, 1),
            epoch=EPOCH,
            ai_summary=False,
            telegram=bot,
        )
        self.assertTrue(ok)
        self.assertTrue(bot.send_message.called)
        payload = bot.send_message.call_args[0][0]
        self.assertIn("visitor_aaa", payload)


class SchedulerTests(unittest.TestCase):
    def test_parse_fire_at(self) -> None:
        self.assertEqual(parse_fire_at("09:30"), dt_time(9, 30))
        self.assertEqual(parse_fire_at("bad"), dt_time(21, 0))

    def test_cfg_defaults(self) -> None:
        cfg = weekly_brief_cfg({"venue": "X", "weekly_customer_brief": {"enabled": False}})
        self.assertFalse(cfg["enabled"])
        self.assertEqual(cfg["epoch"], EPOCH)

    def test_scheduler_sends_once_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.json"
            calls: list[dict] = []
            moments = [
                _moment("2026-10-01T21:05:00"),
                _moment("2026-10-01T21:30:00"),
            ]
            queue = list(moments)

            def now_fn() -> datetime:
                return queue[0] if queue else moments[-1]

            def send_fn(**kwargs):
                calls.append(kwargs)
                return True

            sched = WeeklyCustomerBriefScheduler(
                get_cfg=lambda: {
                    "workplace_type": "massage",
                    "venue": "Spa",
                    "weekly_customer_brief": {"enabled": True, "epoch": "2026-09-25"},
                },
                get_bot=lambda: MagicMock(),
                db_path=Path(tmp) / "events.db",
                state_file=state,
                now_fn=now_fn,
                send_fn=send_fn,
                workplace_fn=lambda: "massage",
            )
            self.assertTrue(sched.tick())
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["as_of"], date(2026, 10, 1))
            # second tick same period must not resend
            queue.pop(0)
            self.assertFalse(sched.tick())
            self.assertEqual(len(calls), 1)
            sent = load_sent_periods(state)
            self.assertIn("2026-09-25_2026-10-01", sent)

    def test_garage_workplace_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sched = WeeklyCustomerBriefScheduler(
                get_cfg=lambda: {"workplace_type": "garage"},
                get_bot=lambda: MagicMock(),
                state_file=Path(tmp) / "state.json",
                now_fn=lambda: _moment("2026-10-01T21:05:00"),
                send_fn=lambda **k: True,
                workplace_fn=lambda: "garage",
            )
            self.assertFalse(sched.tick())


if __name__ == "__main__":
    unittest.main()
