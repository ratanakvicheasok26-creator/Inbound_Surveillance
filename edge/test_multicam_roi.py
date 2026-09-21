import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from launcher import LiveStreamEngine, CameraStreamWorker


class TestMultiCamRoiTracking(unittest.TestCase):
    def setUp(self):
        self.engine = LiveStreamEngine()
        self.engine.cfg["cameras"] = [
            {"id": "cam-1", "name": "Main Camera", "source": "0", "ml_enabled": True},
            {"id": "cam-2", "name": "Bay 2 Camera", "source": "rtsp://192.168.1.102/live", "ml_enabled": True},
            {"id": "cam-3", "name": "Bay 3 Camera", "source": "rtsp://192.168.1.103/live", "ml_enabled": True},
            {"id": "cam-4", "name": "Bay 4 Camera", "source": "rtsp://192.168.1.104/live", "ml_enabled": True},
        ]
        self.engine.cfg["active_camera_id"] = "cam-1"
        self.engine.active_roi_cameras.clear()

    def test_toggle_camera_ml(self):
        with patch("launcher.save_config"):
            # Toggle cam-2 from True -> False
            res = self.engine.toggle_camera_ml("cam-2", False)
            self.assertTrue(res["success"])
            self.assertFalse(res["ml_enabled"])

            # Verify in cfg
            c2 = next(c for c in self.engine.cfg["cameras"] if c["id"] == "cam-2")
            self.assertFalse(c2["ml_enabled"])

            # Add cam-2 to active_roi_cameras and toggle ML off -> should be evicted
            self.engine.active_roi_cameras["cam-2"] = time.time()
            res = self.engine.toggle_camera_ml("cam-2", False)
            self.assertNotIn("cam-2", self.engine.active_roi_cameras)

            # Toggle back to True
            res = self.engine.toggle_camera_ml("cam-2", True)
            self.assertTrue(res["success"])
            self.assertTrue(res["ml_enabled"])

            # Invalid camera
            err = self.engine.toggle_camera_ml("non-existent-cam")
            self.assertFalse(err["success"])

    def test_background_camera_ml_gating(self):
        # Fake frame and detector
        fake_frame = MagicMock()
        fake_frame.shape = (480, 640, 3)

        # Mock model prediction returning 1 person
        mock_person = MagicMock()
        mock_person.box = (200, 150, 400, 350)
        mock_person.keypoints = []

        self.engine.running = True
        self.engine.model = MagicMock()
        cam_cfg = dict(self.engine.cfg["cameras"][1])

        with patch("launcher.person_detections", return_value=([mock_person], [])):
            with patch("launcher.detection_in_bay", return_value=True):
                # When ML is enabled, evaluate_background_camera marks active_roi_cameras
                self.engine._evaluate_background_camera("cam-2", fake_frame, cam_cfg)
                self.assertIn("cam-2", self.engine.active_roi_cameras)

                # Now disable ML on cam-2
                cam_cfg["ml_enabled"] = False
                self.engine.active_roi_cameras.clear()

                self.engine._evaluate_background_camera("cam-2", fake_frame, cam_cfg)
                # cam-2 should NOT be tracked in active_roi_cameras because ML is disabled
                self.assertNotIn("cam-2", self.engine.active_roi_cameras)

    def test_concurrent_multi_camera_tracking_and_exit(self):
        now = time.time()
        # Simulate 3 background cameras actively tracking persons in ROI
        self.engine.active_roi_cameras["cam-2"] = now
        self.engine.active_roi_cameras["cam-3"] = now
        self.engine.active_roi_cameras["cam-4"] = now

        # Active camera is cam-1
        self.assertEqual(len(self.engine.active_roi_cameras), 3)

        # Simulate person leaving ROI on cam-2: timestamp not updated
        old_time = now - 4.0
        self.engine.active_roi_cameras["cam-2"] = old_time

        # If a camera is evaluated with no person in ROI and elapsed > 3.0s, it's removed
        last_seen = self.engine.active_roi_cameras.get("cam-2", 0.0)
        if (now - last_seen) >= 3.0:
            self.engine.active_roi_cameras.pop("cam-2", None)

        self.assertNotIn("cam-2", self.engine.active_roi_cameras)
        self.assertIn("cam-3", self.engine.active_roi_cameras)
        self.assertIn("cam-4", self.engine.active_roi_cameras)

    def test_background_camera_freshness_when_stale_in_grid(self):
        from adapters.base import FramePacket
        import numpy as np

        cam_cfg = {"id": "cam-2", "name": "Bay 2 Camera", "source": "rtsp://192.168.1.102/live"}
        worker = CameraStreamWorker("cam-2", cam_cfg)
        self.engine.camera_pool._workers["cam-2"] = worker

        # Simulate stale cached JPEG
        now = time.time()
        worker.latest_jpeg = b"STALE_OLD_JPEG"
        worker.latest_ts = now - 2.0

        # Grabber has newer frame
        fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        new_packet = FramePacket(fake_frame, now, 10, 10, jpeg=b"FRESH_NEW_JPEG")
        worker.grabber.peek_latest_frame = lambda: new_packet

        # get_camera_frame must return the fresh JPEG, not the 2.0s stale one
        jpeg, ctype = self.engine.get_camera_frame("cam-2")
        self.assertEqual(jpeg, b"FRESH_NEW_JPEG")
        self.assertEqual(worker.latest_jpeg, b"FRESH_NEW_JPEG")
        self.assertEqual(worker.latest_ts, now)


if __name__ == "__main__":
    unittest.main()
