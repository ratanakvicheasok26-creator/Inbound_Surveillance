"""Sunday weekly-customer-brief scheduler tests for run_champei.py."""

from __future__ import annotations

import os
import threading
import unittest
from datetime import date, datetime, time as dt_time
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from run_champei import _env_flag, _parse_fire_at, _weekly_brief_loop

TZ = ZoneInfo("Asia/Phnom_Penh")
BRANCH = "champei-pp-01"
DB_PATH = Path("/tmp/champei-weekly-brief-test.db")


def _moment(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=TZ)


class WeeklyBriefSchedulerTests(unittest.TestCase):
    def _run_loop(
        self,
        moments: list[datetime],
        *,
        fire_at: dt_time = dt_time(21, 0),
        env: dict[str, str] | None = None,
        expect_loop_exit: bool = True,
    ) -> list[dict]:
        calls: list[dict] = []
        stop_event = threading.Event()
        queue = list(moments)

        def now_fn() -> datetime:
            if not queue:
                stop_event.set()
                return moments[-1]
            return queue.pop(0)

        def send_fn(**kwargs) -> bool:
            calls.append(kwargs)
            return True

        with patch.dict(os.environ, env or {}, clear=False):
            _weekly_brief_loop(
                stop_event,
                branch_id=BRANCH,
                db_path=DB_PATH,
                fire_at=fire_at,
                tz=TZ,
                now_fn=now_fn,
                send_fn=send_fn,
                poll_seconds=0.01,
            )
        if expect_loop_exit:
            self.assertTrue(stop_event.is_set(), "loop did not terminate")
        return calls

    def test_sends_once_on_sunday_after_fire_time(self) -> None:
        calls = self._run_loop(
            [
                _moment("2026-09-27T20:59:00"),
                _moment("2026-09-27T21:00:00"),
                _moment("2026-09-27T21:30:00"),
            ]
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["branch_id"], BRANCH)
        self.assertEqual(calls[0]["db_path"], DB_PATH)
        self.assertEqual(calls[0]["as_of"], date(2026, 9, 27))

    def test_no_send_before_sunday_fire_time(self) -> None:
        calls = self._run_loop([_moment("2026-09-27T18:00:00")])
        self.assertEqual(calls, [])

    def test_no_send_on_other_weekdays(self) -> None:
        calls = self._run_loop(
            [
                _moment("2026-09-25T22:00:00"),
                _moment("2026-09-26T22:00:00"),
                _moment("2026-09-28T22:00:00"),
            ]
        )
        self.assertEqual(calls, [])

    def test_no_second_send_in_same_week(self) -> None:
        calls = self._run_loop(
            [
                _moment("2026-09-27T21:05:00"),
                _moment("2026-09-27T22:00:00"),
                _moment("2026-09-27T23:30:00"),
            ]
        )
        self.assertEqual(len(calls), 1)

    def test_sends_again_the_following_sunday(self) -> None:
        calls = self._run_loop(
            [
                _moment("2026-09-27T21:05:00"),
                _moment("2026-10-04T21:05:00"),
            ]
        )
        self.assertEqual(
            [call["as_of"] for call in calls],
            [date(2026, 9, 27), date(2026, 10, 4)],
        )

    def test_custom_fire_time_from_loop_argument(self) -> None:
        calls = self._run_loop(
            [
                _moment("2026-09-27T08:00:00"),
                _moment("2026-09-27T09:00:00"),
            ],
            fire_at=dt_time(9, 0),
        )
        self.assertEqual(len(calls), 1)

    def test_disabled_by_environment(self) -> None:
        calls = self._run_loop(
            [_moment("2026-09-27T22:00:00")],
            env={"WEEKLY_CUSTOMER_BRIEF_ENABLED": "false"},
            expect_loop_exit=False,
        )
        self.assertEqual(calls, [])

    def test_naive_clock_is_localized_to_tz(self) -> None:
        calls = self._run_loop(
            [
                datetime.fromisoformat("2026-09-27T21:10:00"),
                datetime.fromisoformat("2026-09-27T21:20:00"),
            ]
        )
        self.assertEqual(len(calls), 1)


class FireAtParsingTests(unittest.TestCase):
    def test_blank_uses_default(self) -> None:
        self.assertEqual(_parse_fire_at(""), dt_time(21, 0))
        self.assertEqual(_parse_fire_at("  "), dt_time(21, 0))

    def test_parses_hh_mm_and_hh_mm_ss(self) -> None:
        self.assertEqual(_parse_fire_at("09:30"), dt_time(9, 30))
        self.assertEqual(_parse_fire_at("18:05:00"), dt_time(18, 5))

    def test_invalid_uses_default(self) -> None:
        self.assertEqual(_parse_fire_at("not-a-time"), dt_time(21, 0))
        self.assertEqual(_parse_fire_at("25:99"), dt_time(21, 0))


class EnvFlagTests(unittest.TestCase):
    def test_default_true_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(_env_flag("WEEKLY_CUSTOMER_BRIEF_ENABLED", True))

    def test_falsy_values(self) -> None:
        for raw in ("0", "false", "no", "off", "FALSE"):
            with patch.dict(os.environ, {"WEEKLY_CUSTOMER_BRIEF_ENABLED": raw}, clear=True):
                self.assertFalse(_env_flag("WEEKLY_CUSTOMER_BRIEF_ENABLED", True))

    def test_truthy_values(self) -> None:
        for raw in ("1", "true", "yes", "on"):
            with patch.dict(os.environ, {"WEEKLY_CUSTOMER_BRIEF_ENABLED": raw}, clear=True):
                self.assertTrue(_env_flag("WEEKLY_CUSTOMER_BRIEF_ENABLED", False))


if __name__ == "__main__":
    unittest.main()
