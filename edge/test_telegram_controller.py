"""RBAC + /name parsing tests for telegram_controller (no live Telegram)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from telegram_controller import (
    DENIED,
    TelegramController,
    parse_name_args,
)
from visitor_registry import get_visitor_display_name


class ParseNameArgsTests(unittest.TestCase):
    def test_alias_only_defaults_tier(self) -> None:
        parsed = parse_name_args("visitor_a1f3 Lok Chumteav Sophy")
        self.assertIsNotNone(parsed)
        assert parsed is not None
        vid, alias, tier, notes = parsed
        self.assertEqual(vid, "visitor_a1f3")
        self.assertEqual(alias, "Lok Chumteav Sophy")
        self.assertEqual(tier, "Standard")
        self.assertEqual(notes, "")

    def test_tier_and_notes(self) -> None:
        parsed = parse_name_args("visitor_x Alice | Gold | Prefers Room 3")
        self.assertEqual(
            parsed,
            ("visitor_x", "Alice", "Gold", "Prefers Room 3"),
        )

    def test_invalid_missing_alias(self) -> None:
        self.assertIsNone(parse_name_args("visitor_only"))


class TelegramControllerRbacTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "events.db"
        self.session = MagicMock()
        self.ctrl = TelegramController(
            token="test-token",
            staff_chat_id="111",
            owner_chat_id="222",
            db_path=self.db_path,
            branch_id="champei-pp-01",
            session=self.session,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_stranger_ignored(self) -> None:
        reply = self.ctrl.handle_message("999", "/scorecard")
        self.assertIsNone(reply)
        reply2 = self.ctrl.handle_message("999", "/name visitor_a Alice")
        self.assertIsNone(reply2)

    def test_staff_denied_scorecard_and_churn(self) -> None:
        self.assertEqual(self.ctrl.handle_message("111", "/scorecard"), DENIED)
        self.assertEqual(self.ctrl.handle_message("111", "/churn"), DENIED)
        self.assertEqual(self.ctrl.handle_message("111", "/weekly"), DENIED)

    def test_owner_permitted_scorecard(self) -> None:
        reply = self.ctrl.handle_message("222", "/scorecard")
        self.assertIsNotNone(reply)
        assert reply is not None
        self.assertIn("DAILY OPERATIONS SCORECARD", reply)
        self.assertIn("champei-pp-01", reply)

    def test_owner_permitted_churn(self) -> None:
        reply = self.ctrl.handle_message("222", "/churn")
        self.assertIsNotNone(reply)
        assert reply is not None
        self.assertIn("SILENT CHURN WATCHLIST", reply)

    def test_owner_permitted_weekly_brief(self) -> None:
        reply = self.ctrl.handle_message("222", "/weekly")
        self.assertIsNotNone(reply)
        assert reply is not None
        self.assertIn("WEEKLY CUSTOMER ACTIVITY BRIEF", reply)
        self.assertIn("champei-pp-01", reply)

    def test_staff_name_upsert(self) -> None:
        reply = self.ctrl.handle_message(
            "111",
            "/name visitor_a1f3 Lok Chumteav Sophy | Gold | Prefers Room 3",
        )
        self.assertIsNotNone(reply)
        assert reply is not None
        self.assertIn("✅ Saved profile", reply)
        self.assertIn("visitor_a1f3", reply)
        self.assertIn("Lok Chumteav Sophy", reply)
        self.assertIn("Gold", reply)

        info = get_visitor_display_name("visitor_a1f3", db_path=self.db_path)
        self.assertTrue(info["is_named"])
        self.assertEqual(info["display_name"], "Lok Chumteav Sophy")
        self.assertEqual(info["tier"], "Gold")
        self.assertEqual(info["notes"], "Prefers Room 3")

    def test_info_after_name(self) -> None:
        self.ctrl.handle_message("111", "/name visitor_b Bob | Standard |")
        reply = self.ctrl.handle_message("111", "/info visitor_b")
        self.assertIsNotNone(reply)
        assert reply is not None
        self.assertIn("Bob", reply)
        self.assertIn("Standard", reply)

    def test_staff_help_omits_owner_commands(self) -> None:
        reply = self.ctrl.handle_message("111", "/help")
        assert reply is not None
        self.assertIn("/name", reply)
        self.assertNotIn("/scorecard", reply)
        self.assertNotIn("/weekly", reply)

    def test_owner_help_includes_owner_commands(self) -> None:
        reply = self.ctrl.handle_message("222", "/help")
        assert reply is not None
        self.assertIn("/scorecard", reply)
        self.assertIn("/churn", reply)
        self.assertIn("/weekly", reply)


if __name__ == "__main__":
    unittest.main()
