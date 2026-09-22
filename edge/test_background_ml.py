import sys
from pathlib import Path
import unittest
import numpy as np
import time

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from launcher import (
    _normalize_cameras,
    upsert_camera,
    _box_intersects_roi,
    CameraStreamWorker,
    CameraStreamPool,
    LiveStreamEngine,
)
from person import Detection
from vehicle import VehicleDetection


class TestBackgroundML(unittest.TestCase):
    def test_camera_trigger_mode_config(self):
        cfg = {"cameras": []}
        cam = upsert_camera(cfg, {"id": "cam-1", "name": "Entrance", "source": "0"})
        self.assertEqual(cam.get("trigger_mode"), "roi_state_change")

        cam2 = upsert_camera(cfg, {"id": "cam-2", "name": "Side Bay", "source": "1", "trigger_mode": "any_detection"})
        self.assertEqual(cam2.get("trigger_mode"), "any_detection")

        normalized = _normalize_cameras([cam, cam2])
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized[0]["trigger_mode"], "roi_state_change")
        self.assertEqual(normalized[1]["trigger_mode"], "any_detection")

    def test_box_intersects_roi(self):
        roi = [0.2, 0.2, 0.4, 0.4]  # [200, 200, 400, 400] on 1000x1000
        frame_w, frame_h = 1000, 1000

        # Vehicle right inside ROI: [250, 250, 350, 350]
        inside_box = (250, 250, 350, 350)
        self.assertTrue(_box_intersects_roi(inside_box, roi, frame_w, frame_h))

        # Vehicle completely outside ROI: [700, 700, 800, 800]
        outside_box = (700, 700, 800, 800)
        self.assertFalse(_box_intersects_roi(outside_box, roi, frame_w, frame_h))

    def test_motion_gating_skips_static_scene(self):
        called = []
        def eval_cb(cid, frame, cfg):
            called.append(cid)

        worker = CameraStreamWorker(
            "cam-test",
            {"source": "0", "trigger_mode": "roi_state_change"},
            eval_callback=eval_cb,
        )

        # Simulate two identical static frames
        static_frame1 = np.full((120, 160, 3), 100, dtype=np.uint8)
        static_frame2 = np.full((120, 160, 3), 100, dtype=np.uint8)

        # Feed frame 1
        worker._prev_gray = static_frame1[:, :, 0]
        worker._last_bg_eval = 0.0

        # Evaluate with identical frame 2: diff is 0, so motion_score is 0
        diff = np.abs(static_frame1[:, :, 0].astype(int) - static_frame2[:, :, 0].astype(int))
        motion_score = float(np.mean(diff))
        self.assertEqual(motion_score, 0.0)

        # Non-identical frame with motion
        motion_frame = np.full((120, 160, 3), 200, dtype=np.uint8)
        diff_motion = np.abs(static_frame1[:, :, 0].astype(int) - motion_frame[:, :, 0].astype(int))
        motion_score_high = float(np.mean(diff_motion))
        self.assertGreater(motion_score_high, 1.2)

    def test_evaluate_background_camera_and_acknowledge(self):
        engine = LiveStreamEngine()
        engine.running = True

        # Mock models to avoid requiring heavy neural net weights in unit test
        class DummyResult:
            boxes = None
            keypoints = None
        class DummyModel:
            def predict(self, *args, **kwargs):
                return [DummyResult()]

        engine.model = DummyModel()
        engine.vehicle_model = None

        cam_cfg = {
            "id": "cam-bg",
            "name": "Background Camera 1",
            "trigger_mode": "any_detection",
            "roi": [0.1, 0.1, 0.5, 0.5],
        }

        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test event triggering with any_detection
        import launcher
        orig_person_detections = launcher.person_detections
        try:
            fake_det = Detection(
                x1=100.0,
                y1=100.0,
                x2=200.0,
                y2=300.0,
                conf=0.85,
                identity="Mechanic John",
                is_staff=True,
            )
            launcher.person_detections = lambda *a, **k: ([fake_det], [])

            engine._evaluate_background_camera("cam-bg", fake_frame, cam_cfg)
            self.assertIsNotNone(engine.latest_camera_event)
            self.assertEqual(engine.latest_camera_event["camera_id"], "cam-bg")
            self.assertEqual(engine.latest_camera_event["camera_name"], "Background Camera 1")
            self.assertIn("person detected", engine.latest_camera_event["event"])
            self.assertFalse(engine.latest_camera_event["handled"])

            # Test telemetry propagation
            telem = engine.garage_telemetry()
            self.assertIn("latest_camera_event", telem)
            self.assertEqual(telem["latest_camera_event"]["camera_id"], "cam-bg")

            # Test event acknowledgement
            event_id = engine.latest_camera_event["event_id"]
            engine.acknowledge_event(event_id)
            self.assertTrue(engine.latest_camera_event["handled"])
        finally:
            launcher.person_detections = orig_person_detections


if __name__ == "__main__":
    unittest.main()
