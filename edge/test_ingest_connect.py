"""Ingest contract: connect preview must not wait on YOLO or reconnect storms."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from launcher import CameraStreamWorker, LiveStreamEngine


def _worker(source="0", protocol="webcam") -> CameraStreamWorker:
    return CameraStreamWorker(
        "cam-1",
        {"id": "cam-1", "name": "Cam", "source": source, "protocol": protocol, "enabled": True},
    )


class IngestReconnectTests(unittest.TestCase):
    def test_connecting_does_not_reconnect_even_if_latest_ts_is_zero(self) -> None:
        worker = _worker()
        worker.grabber.connection_state = "CONNECTING"
        worker.grabber._adapter = None
        worker.latest_ts = 0.0
        now = time.time() + 30
        self.assertFalse(worker._should_reconnect(now))
        worker._reconnect = MagicMock()
        self.assertEqual(worker._maybe_reconnect(now, now - 10), now - 10)
        worker._reconnect.assert_not_called()

    def test_failed_and_standby_do_reconnect(self) -> None:
        worker = _worker()
        worker.latest_ts = 0.0
        worker.grabber.connection_state = "FAILED"
        self.assertTrue(worker._should_reconnect(time.time()))
        worker.grabber.connection_state = "STANDBY"
        self.assertTrue(worker._should_reconnect(time.time()))

    def test_stall_after_real_frames_reconnects(self) -> None:
        worker = _worker("rtsp://192.168.1.10/stream", "rtsp")
        worker.grabber.connection_state = "RECONNECTING"
        worker.latest_ts = time.time() - 5.0
        self.assertTrue(worker._should_reconnect(time.time()))
        worker._reconnect = MagicMock()
        now = time.time()
        self.assertEqual(worker._maybe_reconnect(now, now - 3.0), now)
        worker._reconnect.assert_called_once()

    def test_connected_without_frames_yet_does_not_reconnect(self) -> None:
        worker = _worker()
        worker.grabber.connection_state = "CONNECTED"
        worker.latest_ts = 0.0
        self.assertFalse(worker._should_reconnect(time.time() + 30))


class IngestPreviewWithoutYoloTests(unittest.TestCase):
    def test_get_camera_frame_uses_worker_jpeg_before_inference(self) -> None:
        engine = LiveStreamEngine.__new__(LiveStreamEngine)
        engine.lock = MagicMock()
        engine.cfg = {
            "active_camera_id": "cam-1",
            "rotate": 0,
            "flip": "none",
            "cameras": [{"id": "cam-1", "name": "Webcam", "enabled": True, "source": "0"}],
        }
        engine._camera_frame_cache = {}
        engine.current_frame_jpeg = None
        engine._fallback_grabber = MagicMock()
        engine._fallback_grabber.peek_latest_frame.return_value = None

        worker = _worker()
        worker.latest_jpeg = b"RAW_PREVIEW_JPEG"
        worker.latest_ts = time.time()
        engine.camera_pool = MagicMock()
        engine.camera_pool.get_worker.return_value = worker

        frame, mime = engine.get_camera_frame("cam-1")
        self.assertEqual(frame, b"RAW_PREVIEW_JPEG")
        self.assertEqual(mime, "image/jpeg")


if __name__ == "__main__":
    unittest.main()
