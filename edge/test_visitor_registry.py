"""Visitor registry + arrival Telegram handshake (no camera/pose deps)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from visitor_registry import (
    format_guest_arrival_message,
    get_visitor_display_name,
    notify_guest_arrival,
    reset_notify_cooldowns,
    set_visitor_alias,
)


class _FakeBot:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def send_alert(self, event_type: str, text: str, photo: bytes | None = None) -> bool:
        self.calls.append((event_type, text))
        return True


class VisitorRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "events.db"
        reset_notify_cooldowns()

    def tearDown(self) -> None:
        reset_notify_cooldowns()
        self.tmp.cleanup()

    def test_fallback_when_no_alias(self) -> None:
        info = get_visitor_display_name("visitor_a1f3", db_path=self.db_path)
        self.assertFalse(info["is_named"])
        self.assertEqual(info["display_name"], "visitor_a1f3")
        self.assertEqual(info["tier"], "Standard")
        self.assertEqual(info["notes"], "")

    def test_upsert_alias_changes_display_and_vip_message(self) -> None:
        set_visitor_alias(
            "visitor_a1f3",
            "Lok Chumteav Sophy",
            vip_tier="Gold",
            notes="Prefers Room 3",
            db_path=self.db_path,
        )
        info = get_visitor_display_name("visitor_a1f3", db_path=self.db_path)
        self.assertTrue(info["is_named"])
        self.assertEqual(info["display_name"], "Lok Chumteav Sophy")
        self.assertEqual(info["tier"], "Gold")
        self.assertEqual(info["notes"], "Prefers Room 3")

        event, text = format_guest_arrival_message(
            "visitor_a1f3",
            4,
            branch_id="champei-pp-01",
            zone="shoe_lounge",
            meta=info,
        )
        self.assertEqual(event, "vip_arrival")
        self.assertIn("VIP ARRIVAL", text)
        self.assertIn("Lok Chumteav Sophy", text)
        self.assertIn("Visit #4", text)
        self.assertIn("Gold", text)

        bot = _FakeBot()
        ok = notify_guest_arrival(
            "visitor_a1f3",
            4,
            branch_id="champei-pp-01",
            zone="shoe_lounge",
            telegram=bot,
            db_path=self.db_path,
            now=1000.0,
        )
        self.assertTrue(ok)
        self.assertEqual(len(bot.calls), 1)
        self.assertEqual(bot.calls[0][0], "vip_arrival")
        self.assertIn("Lok Chumteav Sophy", bot.calls[0][1])

    def test_new_and_returning_guest_copy(self) -> None:
        _, new_msg = format_guest_arrival_message(
            "visitor_x",
            1,
            branch_id="champei-pp-01",
            zone="shoe_lounge",
        )
        self.assertIn("NEW GUEST DETECTED", new_msg)
        self.assertIn("shoe_lounge", new_msg)

        _, ret_msg = format_guest_arrival_message(
            "visitor_x",
            2,
            branch_id="champei-pp-01",
            zone="shoe_lounge",
        )
        self.assertIn("RETURNING GUEST", ret_msg)
        self.assertIn("visitor_x", ret_msg)
        self.assertIn("Visit #2", ret_msg)

    def test_cooldown_suppresses_second_dispatch(self) -> None:
        bot = _FakeBot()
        first = notify_guest_arrival(
            "visitor_cd",
            1,
            telegram=bot,
            db_path=self.db_path,
            now=10.0,
            cooldown_seconds=3600.0,
        )
        second = notify_guest_arrival(
            "visitor_cd",
            2,
            telegram=bot,
            db_path=self.db_path,
            now=10.0 + 60.0,
            cooldown_seconds=3600.0,
        )
        third = notify_guest_arrival(
            "visitor_cd",
            3,
            telegram=bot,
            db_path=self.db_path,
            now=10.0 + 3601.0,
            cooldown_seconds=3600.0,
        )
        self.assertTrue(first)
        self.assertFalse(second)
        self.assertTrue(third)
        self.assertEqual(len(bot.calls), 2)


class TelegramRoutingTests(unittest.TestCase):
    def test_staff_and_owner_routes(self) -> None:
        import os

        from telegram_out import TelegramOut

        prev = {
            k: os.environ.get(k)
            for k in (
                "TELEGRAM_BOT_TOKEN",
                "TELEGRAM_CHAT_ID",
                "TELEGRAM_STAFF_CHAT_ID",
                "TELEGRAM_OWNER_CHAT_ID",
            )
        }
        try:
            os.environ["TELEGRAM_BOT_TOKEN"] = "tok"
            os.environ["TELEGRAM_CHAT_ID"] = "primary"
            os.environ["TELEGRAM_STAFF_CHAT_ID"] = "staff"
            os.environ["TELEGRAM_OWNER_CHAT_ID"] = "owner"
            bot = TelegramOut()
            self.assertEqual(bot.resolve_chat_for_event("wait_bottleneck"), "staff")
            self.assertEqual(bot.resolve_chat_for_event("vip_arrival"), "staff")
            self.assertEqual(bot.resolve_chat_for_event("guest_arrival"), "staff")
            self.assertEqual(bot.resolve_chat_for_event("daily_scorecard"), "owner")
            self.assertEqual(bot.resolve_chat_for_event("silent_churn"), "owner")
        finally:
            for key, value in prev.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
