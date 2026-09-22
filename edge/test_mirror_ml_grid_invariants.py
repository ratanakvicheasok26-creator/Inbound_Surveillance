import unittest
import numpy as np
import cv2
from unittest.mock import MagicMock

from person import (
    is_upper_body_pose,
    is_human_pose,
    is_crouch_or_sit_pose,
    is_face_closeup,
    HEAD_POINTS,
    NOSE,
    L_EYE,
    R_EYE,
    L_EAR,
    R_EAR,
    L_SHOULDER,
    R_SHOULDER,
    L_ELBOW,
    R_ELBOW,
    L_WRIST,
    R_WRIST,
    L_HIP,
    R_HIP,
    L_KNEE,
    R_KNEE,
    L_ANKLE,
    R_ANKLE,
)
from occupancy import is_admissible_bay_occupant
from launcher import (
    resolve_orient,
    orient_frame,
    parse_flip,
    is_auto_rotate,
    transform_point_hflip,
    transform_point_vflip,
    LiveStreamEngine,
)


def _empty_kpts():
    return [(0.0, 0.0, 0.0) for _ in range(17)]


class TestMirrorOrientInvariants(unittest.TestCase):
    """Guards camera orientation and mirror flip persistence."""

    def test_flip_persistence_with_auto_rotate(self):
        engine = LiveStreamEngine.__new__(LiveStreamEngine)
        engine.lock = MagicMock()
        engine.cfg = {
            "rotate": "auto",
            "flip": "none",
            "active_camera_id": "cam-1",
            "cameras": [{"id": "cam-1", "name": "Bay 1", "rotate": "auto", "flip": "none"}],
            "bays": [{"id": "bay-1", "name": "Bay 1", "roi": [0.1, 0.1, 0.4, 0.4]}],
        }
        engine.bay_manager = MagicMock()
        engine.bay_manager.telemetry.return_value = {}

        # 1. Setting flip='h' while rotate='auto'
        res = engine.set_orient(rotate="auto", flip="h")
        self.assertEqual(engine.cfg.get("rotate"), "auto")
        self.assertEqual(engine.cfg.get("flip"), "h")
        self.assertEqual(engine.cfg["cameras"][0].get("flip"), "h")

        # 2. Subsequent reconnect/orient call with rotate='auto' and flip=None must NOT wipe flip
        res2 = engine.set_orient(rotate="auto")
        self.assertEqual(engine.cfg.get("rotate"), "auto")
        self.assertEqual(engine.cfg.get("flip"), "h")
        self.assertEqual(engine.cfg["cameras"][0].get("flip"), "h")

    def test_flip_bay_geometry_transformation(self):
        engine = LiveStreamEngine.__new__(LiveStreamEngine)
        engine.lock = MagicMock()
        engine.cfg = {
            "rotate": "auto",
            "flip": "none",
            "active_camera_id": "cam-1",
            "cameras": [{"id": "cam-1", "name": "Bay 1", "rotate": "auto", "flip": "none"}],
            "bays": [{"id": "bay-1", "name": "Bay 1", "roi": [0.1, 0.2, 0.3, 0.4]}],
        }
        engine.bay_manager = MagicMock()
        engine.bay_manager.telemetry.return_value = {}

        # Changing flip to 'h' while rotate is 'auto' must mirror ROI horizontally: [1 - (0.1+0.3), 0.2, 0.3, 0.4] -> [0.6, 0.2, 0.3, 0.4]
        engine.set_orient(rotate="auto", flip="h")
        bays = engine.cfg.get("bays")
        self.assertIsNotNone(bays)
        roi = bays[0]["roi"]
        self.assertAlmostEqual(roi[0], 0.6, places=2)
        self.assertAlmostEqual(roi[1], 0.2, places=2)
        self.assertAlmostEqual(roi[2], 0.3, places=2)
        self.assertAlmostEqual(roi[3], 0.4, places=2)

    def test_orient_frame_horizontal_flip(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[20:40, 10:30] = 255  # Box on the left
        flipped = orient_frame(img, 0, "h")
        # In flipped image, the box must be on the right (x: 70 to 90)
        self.assertEqual(flipped[30, 80, 0], 255)
        self.assertEqual(flipped[30, 20, 0], 0)


class TestMachineLearningTrackingInvariants(unittest.TestCase):
    """Guards ML person tracking, upper-body desk worker framing, and bay occupancy."""

    def test_tinypose_desk_worker_upper_body_pose(self):
        # Desk worker: strong head keypoints (>0.60), lower shoulder keypoints (0.12 - 0.18)
        kpts = _empty_kpts()
        kpts[NOSE] = (300.0, 150.0, 0.75)
        kpts[L_EYE] = (290.0, 140.0, 0.70)
        kpts[R_EYE] = (310.0, 140.0, 0.72)
        kpts[L_EAR] = (280.0, 145.0, 0.65)
        kpts[R_EAR] = (320.0, 145.0, 0.68)
        # Shoulders confidence around 0.16 (below standard 0.35, but above adaptive 0.15)
        kpts[L_SHOULDER] = (260.0, 220.0, 0.16)
        kpts[R_SHOULDER] = (340.0, 220.0, 0.17)

        # Upper body pose should accept this desk worker
        self.assertTrue(
            is_upper_body_pose(200, 100, 400, 350, kpts, 480, kpt_conf=0.35)
        )
        # Human pose with box_conf=0.75 should accept this
        self.assertTrue(
            is_human_pose(200, 100, 400, 350, kpts, 480, kpt_conf=0.35, box_conf=0.75)
        )

    def test_face_closeup_desk_worker_pose(self):
        # Extreme close-up where shoulders are cut off at bottom of frame
        kpts = _empty_kpts()
        kpts[NOSE] = (320.0, 200.0, 0.85)
        kpts[L_EYE] = (300.0, 180.0, 0.82)
        kpts[R_EYE] = (340.0, 180.0, 0.83)
        kpts[L_EAR] = (270.0, 190.0, 0.65)
        kpts[R_EAR] = (370.0, 190.0, 0.67)

        self.assertTrue(is_face_closeup(kpts, kpt_conf=0.35))
        self.assertTrue(
            is_upper_body_pose(250, 150, 390, 320, kpts, 480, kpt_conf=0.35)
        )
        self.assertTrue(
            is_human_pose(250, 150, 390, 320, kpts, 480, kpt_conf=0.35, box_conf=0.80)
        )

    def test_occupancy_admits_desk_worker(self):
        # Test that is_admissible_bay_occupant accepts upper body & desk poses
        kpts = _empty_kpts()
        kpts[NOSE] = (300.0, 150.0, 0.75)
        kpts[L_EYE] = (290.0, 140.0, 0.70)
        kpts[R_EYE] = (310.0, 140.0, 0.72)
        kpts[L_SHOULDER] = (260.0, 220.0, 0.16)
        kpts[R_SHOULDER] = (340.0, 220.0, 0.17)

        det = MagicMock()
        det.clutter = False
        det.hits = 3
        det.is_staff = False
        det.keypoints = kpts
        det.box.return_value = (200, 100, 400, 350)
        det.liveness = 5.0
        det.jitter = 0.05
        det.motion = 8.0

        bay = {"id": "bay-1", "name": "Bay 1", "roi": [0.1, 0.1, 0.8, 0.8]}
        self.assertTrue(
            is_admissible_bay_occupant(det, bay, frame_h=480, kpt_conf=0.35)
        )

    def test_occupancy_rejects_frozen_engine_block(self):
        from person import hood_lean_keypoints

        kpts = hood_lean_keypoints()
        det = MagicMock()
        det.clutter = False
        det.hits = 8
        det.is_staff = False
        det.keypoints = kpts
        det.box.return_value = (90, 40, 190, 280)
        det.liveness = 0.4
        det.jitter = 0.002
        det.motion = 0.3

        bay = {"id": "bay-1", "name": "Bay 1", "roi": [0.1, 0.1, 0.8, 0.8], "type": "vehicle_bay"}
        self.assertFalse(
            is_admissible_bay_occupant(det, bay, frame_h=480, kpt_conf=0.12)
        )


class TestCameraGridFrameDelivery(unittest.TestCase):
    """Guards active camera frame delivery in grid mode to prevent blank tiles (HTTP 204)."""

    def test_active_camera_encodes_raw_grabber_frame_without_204(self):
        engine = LiveStreamEngine.__new__(LiveStreamEngine)
        engine.lock = MagicMock()
        engine.cfg = {
            "active_camera_id": "cam-1",
            "rotate": "auto",
            "flip": "none",
            "cameras": [{"id": "cam-1", "name": "Webcam", "enabled": True}],
        }
        engine._camera_frame_cache = {}
        engine.current_frame_jpeg = None  # Not yet set by inference loop

        # Grabber has a raw BGR frame but no pre-encoded hardware JPEG
        raw_bgr = np.zeros((240, 320, 3), dtype=np.uint8)
        pkt = MagicMock()
        pkt.frame = raw_bgr
        pkt.jpeg = None
        pkt.timestamp = 100.0

        engine._fallback_grabber = MagicMock()
        engine._fallback_grabber.peek_latest_frame.return_value = pkt
        engine.camera_pool = MagicMock()
        engine.camera_pool.get_worker.return_value = None

        # Requesting active camera frame should encode the raw frame and return valid JPEG bytes
        jpeg_bytes, mime = engine.get_camera_frame("cam-1")
        self.assertIsNotNone(jpeg_bytes)
        self.assertEqual(mime, "image/jpeg")
        # Check JPEG magic header 0xFF 0xD8
        self.assertTrue(jpeg_bytes.startswith(b"\xff\xd8"))
        # Check cache is updated
        self.assertIn("cam-1", engine._camera_frame_cache)


if __name__ == "__main__":
    unittest.main()
