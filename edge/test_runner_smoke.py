"""Smoke-test run_champei.py --mock: WAL banner, thread start, clean SIGINT exit."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path

EDGE_DIR = Path(__file__).resolve().parent
RUNNER = EDGE_DIR / "run_champei.py"

MARKER_WAL = "[DB] SQLite WAL mode confirmed: wal"
MARKER_TG = "Telegram controller started"
MARKER_SCORE = "Scorecard scheduler started"


class RunChampeiSmokeTests(unittest.TestCase):
    def test_mock_runner_starts_and_exits_cleanly_on_sigint(self) -> None:
        self.assertTrue(RUNNER.is_file(), f"missing runner: {RUNNER}")
        env = os.environ.copy()
        # Avoid loading a real bot token during smoke (controller still starts).
        env.setdefault("TELEGRAM_BOT_TOKEN", "")
        env.setdefault("TELEGRAM_STAFF_CHAT_ID", "")
        env.setdefault("TELEGRAM_OWNER_CHAT_ID", "")

        proc = subprocess.Popen(
            [sys.executable, str(RUNNER), "--mock"],
            cwd=str(EDGE_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        collected: list[str] = []
        deadline = time.monotonic() + 20.0
        try:
            while time.monotonic() < deadline:
                line = proc.stdout.readline()
                if line:
                    collected.append(line)
                    blob = "".join(collected)
                    if (
                        MARKER_WAL in blob
                        and MARKER_TG in blob
                        and MARKER_SCORE in blob
                    ):
                        break
                elif proc.poll() is not None:
                    break
            blob = "".join(collected)
            self.assertIn(MARKER_WAL, blob, f"WAL marker missing in:\n{blob}")
            self.assertIn(MARKER_TG, blob, f"Telegram start missing in:\n{blob}")
            self.assertIn(MARKER_SCORE, blob, f"Scorecard start missing in:\n{blob}")

            proc.send_signal(signal.SIGINT)
            try:
                code = proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGTERM)
                try:
                    code = proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
                    self.fail("run_champei.py --mock did not exit within 5s after SIGINT")
            self.assertEqual(code, 0, f"expected exit 0, got {code}; output:\n{blob}")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2.0)
            if proc.stdout is not None:
                proc.stdout.close()


if __name__ == "__main__":
    unittest.main()
