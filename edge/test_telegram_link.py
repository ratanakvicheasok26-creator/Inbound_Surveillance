from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telegram_link import TelegramLinkService
from telegram_out import TelegramOut, normalize_chat_id, resolve_telegram_credentials


class TelegramLinkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.svc = TelegramLinkService()
        self.svc._token = "123:ABC"
        self.svc._bot_username = "GarageBot"

    def test_begin_link_builds_deep_link(self) -> None:
        result = self.svc.begin_link("11111111-2222-3333-4444-555555555555", "Alex")
        self.assertTrue(result["ok"])
        self.assertTrue(result["deep_link"].startswith("https://t.me/GarageBot?start="))
        self.assertTrue(result["app_link"].startswith("tg://resolve?domain=GarageBot&start="))
        self.assertIn(result["code"], self.svc._pending)

    def test_start_payload_links_chat(self) -> None:
        begun = self.svc.begin_link("user-42", "Alex")
        code = begun["code"]
        handled = self.svc._handle_update(
            {
                "update_id": 1,
                "message": {
                    "text": f"/start {code}",
                    "chat": {"id": 998877},
                    "from": {"id": 55, "username": "alex_op"},
                },
            }
        )
        self.assertTrue(handled)
        link = self.svc.consume_link_for_user("user-42")
        self.assertIsNotNone(link)
        assert link is not None
        self.assertEqual(link["chat_id"], "998877")
        self.assertEqual(link["telegram_username"], "alex_op")

    def test_bare_start_uses_single_pending(self) -> None:
        self.svc.begin_link("user-7", "Sam")
        handled = self.svc._handle_update(
            {
                "update_id": 2,
                "message": {
                    "text": "/start",
                    "chat": {"id": 42},
                    "from": {"id": 9},
                },
            }
        )
        self.assertTrue(handled)
        link = self.svc.consume_link_for_user("user-7")
        self.assertEqual(link["chat_id"], "42")

    @patch("telegram_link.requests.get")
    def test_poll_once_advances_offset(self, mock_get: MagicMock) -> None:
        self.svc.begin_link("user-1", "Alex")
        code = next(iter(self.svc._pending))
        mock_get.return_value.ok = True
        mock_get.return_value.json.return_value = {
            "ok": True,
            "result": [
                {
                    "update_id": 10,
                    "message": {
                        "text": f"/start {code}",
                        "chat": {"id": 77},
                        "from": {"id": 1},
                    },
                }
            ],
        }
        with patch.object(self.svc, "_reply"):
            handled = self.svc.poll_once()
        self.assertEqual(handled, 1)
        self.assertEqual(self.svc._offset, 11)

    def test_repeated_start_retains_linked_message(self) -> None:
        begun = self.svc.begin_link("user-99", "HourMeng")
        code = begun["code"]
        # First /start links the chat
        with patch.object(self.svc, "_reply") as mock_reply:
            handled1 = self.svc._handle_update(
                {
                    "update_id": 1,
                    "message": {
                        "text": f"/start {code}",
                        "chat": {"id": 123456},
                        "from": {"id": 10, "username": "hourmeng"},
                    },
                }
            )
            self.assertTrue(handled1)
            mock_reply.assert_called_with(
                "123456",
                "Linked to HourMeng. Alerts and daily summaries will arrive here. "
                "If you connect a different Telegram later, this chat will stop receiving alerts.",
            )

        # Second /start from the same chat should recognize that it is already linked
        with patch.object(self.svc, "_reply") as mock_reply:
            handled2 = self.svc._handle_update(
                {
                    "update_id": 2,
                    "message": {
                        "text": "/start",
                        "chat": {"id": 123456},
                        "from": {"id": 10, "username": "hourmeng"},
                    },
                }
            )
            self.assertTrue(handled2)
            mock_reply.assert_called_with(
                "123456",
                "This chat is already linked to HourMeng. Alerts and daily summaries will arrive here.",
            )

    def test_active_chat_recognized_on_start(self) -> None:
        self.svc.set_active_chat("777888", "AutoFix Garage")
        with patch.object(self.svc, "_reply") as mock_reply:
            handled = self.svc._handle_update(
                {
                    "update_id": 3,
                    "message": {
                        "text": "/start",
                        "chat": {"id": 777888},
                        "from": {"id": 20},
                    },
                }
            )
            self.assertTrue(handled)
            mock_reply.assert_called_with(
                "777888",
                "This chat is already linked to AutoFix Garage. Alerts and daily summaries will arrive here.",
            )


    def test_relink_forgets_previous_chat(self) -> None:
        first = self.svc.begin_link("user-1", "Alex")
        with patch.object(self.svc, "_reply") as mock_reply:
            self.svc._handle_update(
                {
                    "update_id": 1,
                    "message": {
                        "text": f"/start {first['code']}",
                        "chat": {"id": 111},
                        "from": {"id": 11, "username": "first_tg"},
                    },
                }
            )
        first_link = self.svc.consume_link_for_user("user-1")
        self.assertEqual(first_link["chat_id"], "111")
        self.svc.set_active_chat("111", "Alex")

        second = self.svc.begin_link("user-1", "Alex")
        with patch.object(self.svc, "_reply") as mock_reply:
            self.svc._handle_update(
                {
                    "update_id": 2,
                    "message": {
                        "text": f"/start {second['code']}",
                        "chat": {"id": 222},
                        "from": {"id": 22, "username": "second_tg"},
                    },
                }
            )
            texts = [call.args[1] for call in mock_reply.call_args_list]
            self.assertTrue(any("unlinked" in text.lower() for text in texts))
            self.assertTrue(any(call.args[0] == "111" for call in mock_reply.call_args_list))

        latest = self.svc.consume_link_for_user("user-1")
        self.assertEqual(latest["chat_id"], "222")
        self.assertNotIn("111", self.svc._known_chats)
        self.assertIn("222", self.svc._known_chats)
        self.assertEqual(self.svc._active_chat_id, "222")

        with patch.object(self.svc, "_reply") as mock_reply:
            handled = self.svc._handle_update(
                {
                    "update_id": 3,
                    "message": {
                        "text": "/start",
                        "chat": {"id": 111},
                        "from": {"id": 11},
                    },
                }
            )
            self.assertFalse(handled)
            mock_reply.assert_called_with(
                "111",
                "Open Integrations in Inbound Surveillance and tap Connect Telegram, "
                "then press Start here so we can link this chat to your account.",
            )


class TelegramOutNormalizeTests(unittest.TestCase):
    def test_normalize_keeps_latest_chat_id(self) -> None:
        self.assertEqual(normalize_chat_id("222"), "222")
        self.assertEqual(normalize_chat_id("111,222"), "222")
        self.assertEqual(normalize_chat_id(["111", "222"]), "222")
        self.assertEqual(normalize_chat_id('["111","222"]'), "222")
        bot = TelegramOut("token", ["111", "222"])
        self.assertEqual(bot.chat_id, "222")
        bot.set_chat("111,333")
        self.assertEqual(bot.chat_id, "333")

    def test_resolve_prefers_environment(self) -> None:
        import os

        prev_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        prev_chat = os.environ.get("TELEGRAM_CHAT_ID")
        try:
            os.environ["TELEGRAM_BOT_TOKEN"] = "env-token"
            os.environ["TELEGRAM_CHAT_ID"] = "999"
            token, chat = resolve_telegram_credentials("cfg-token", "111")
            self.assertEqual(token, "env-token")
            self.assertEqual(chat, "999")
            del os.environ["TELEGRAM_BOT_TOKEN"]
            del os.environ["TELEGRAM_CHAT_ID"]
            token, chat = resolve_telegram_credentials("cfg-token", "111")
            self.assertEqual(token, "cfg-token")
            self.assertEqual(chat, "111")
        finally:
            if prev_token is None:
                os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            else:
                os.environ["TELEGRAM_BOT_TOKEN"] = prev_token
            if prev_chat is None:
                os.environ.pop("TELEGRAM_CHAT_ID", None)
            else:
                os.environ["TELEGRAM_CHAT_ID"] = prev_chat


if __name__ == "__main__":
    unittest.main()
