"""Unit tests for the RTMPose person engine duck-type and loader."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bay_zoom import detections_from_result
from person import person_detections, standing_person_keypoints
from rtmpose import (
    RTMPoseEngine,
    load_person_pose_model,
    resolve_models_dir,
    resolve_rtmpose_mode,
)
from runtime import resolve_pose_engine, resolve_runtime


class _FakeDet:
    def __init__(self, boxes):
        self.score_thr = 0.25
        self._boxes = np.asarray(boxes, dtype=np.float32)

    def __call__(self, image):
        return self._boxes


class _FakePose:
    def __init__(self, keypoints):
        self._keypoints = np.asarray(keypoints, dtype=np.float32)

    def __call__(self, image, bboxes=None):
        n = 0 if bboxes is None else len(bboxes)
        xy = self._keypoints[:, :2]
        sc = self._keypoints[:, 2]
        kpts = np.repeat(xy[None, ...], max(n, 1), axis=0)[: max(n, 1)]
        scores = np.repeat(sc[None, ...], max(n, 1), axis=0)[: max(n, 1)]
        if n == 0:
            return np.zeros((0, 17, 2), dtype=np.float32), np.zeros((0, 17), dtype=np.float32)
        return kpts, scores


class _FakeYOLO:
    task = "pose"

    def __init__(self, path, task="pose"):
        self.path = path
        self.task = task


class RTMPoseRuntimeTests(unittest.TestCase):
    def test_default_engine_is_rtmpose(self):
        self.assertEqual(resolve_pose_engine({}), "rtmpose")
        self.assertEqual(resolve_pose_engine({"pose_engine": "rtmlib"}), "rtmpose")
        self.assertEqual(resolve_pose_engine({"pose_engine": "yolo"}), "yolo")
        self.assertEqual(resolve_pose_engine({"pose_engine": "tinypose"}), "tinypose")
        profile = resolve_runtime({})
        self.assertEqual(profile.pose_engine, "rtmpose")

    def test_cpu_defaults_to_lightweight_mode(self):
        profile = resolve_runtime({"runtime": "cpu"})
        self.assertEqual(resolve_rtmpose_mode({}, profile), "lightweight")
        self.assertEqual(resolve_rtmpose_mode({"rtmpose_mode": "balanced"}, profile), "balanced")


class RTMPoseEngineTests(unittest.TestCase):
    def test_empty_frame_returns_duck_typed_result(self):
        engine = RTMPoseEngine(
            auto_download=False,
            det_model=_FakeDet([]),
            pose_model=_FakePose(standing_person_keypoints()),
        )
        results = engine.predict(np.zeros((480, 640, 3), dtype=np.uint8), conf=0.25)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0].boxes), 0)
        self.assertEqual(results[0].keypoints.data.shape, (0, 17, 3))

    def test_boxes_and_coco17_skeletons_feed_person_pipeline(self):
        kpts = standing_person_keypoints()
        engine = RTMPoseEngine(
            auto_download=False,
            det_model=_FakeDet([[100.0, 80.0, 180.0, 300.0]]),
            pose_model=_FakePose(kpts),
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = engine.predict(frame, conf=0.25)
        self.assertEqual(engine.task, "pose")
        res = results[0]
        self.assertEqual(len(res.boxes), 1)
        self.assertEqual(res.boxes[0].xyxy.shape, (1, 4))
        self.assertEqual(res.keypoints.data.shape, (1, 17, 3))
        accepted, rejected = person_detections(res, 480, conf_min=0.25, kpt_conf=0.35)
        self.assertTrue(accepted or rejected)
        dets = detections_from_result(res, conf_min=0.25)
        self.assertEqual(len(dets), 1)
        self.assertEqual(len(dets[0].keypoints), 17)

    def test_loader_uses_injected_yolo_when_engine_is_yolo(self):
        profile = resolve_runtime({"pose_engine": "yolo"})
        model = load_person_pose_model(
            profile,
            models_dir=ROOT / "models",
            weights_path="yolo11n-pose.pt",
            cfg={"pose_engine": "yolo"},
            yolo_cls=_FakeYOLO,
        )
        self.assertIsInstance(model, _FakeYOLO)

    def test_resolve_models_dir_creates_writable_fallback(self):
        dest = resolve_models_dir(lambda rel: ROOT / "does-not-exist" / rel, ROOT)
        self.assertEqual(dest, ROOT / "models")
        self.assertTrue(dest.is_dir())


if __name__ == "__main__":
    unittest.main()
