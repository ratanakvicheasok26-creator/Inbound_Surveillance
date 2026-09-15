"""Unit tests for PP-TinyPose and PP-PicoDet ONNX inference pipeline."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from bay_zoom import detections_from_result, zoom_empty_bays
from person import Detection, person_detections
from runtime import RuntimeProfile, resolve_pose_engine, resolve_runtime
from tinypose import (
    PaddlePoseEngine,
    PicoDetDetector,
    TinyPoseEstimator,
    ensure_tinypose_models,
)

MODELS_DIR = Path(__file__).resolve().parent / "models"


class TestTinyPosePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pico_path, cls.pose_path = ensure_tinypose_models(MODELS_DIR, download=False)

    def test_01_models_present(self):
        self.assertTrue(self.pico_path.is_file(), f"PicoDet ONNX missing: {self.pico_path}")
        self.assertTrue(self.pose_path.is_file(), f"TinyPose ONNX missing: {self.pose_path}")
        self.assertGreater(self.pico_path.stat().st_size, 1_000_000)
        self.assertGreater(self.pose_path.stat().st_size, 1_000_000)

    def test_02_picodet_detector(self):
        detector = PicoDetDetector(self.pico_path)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Empty frame should produce 0 detections at standard confidence
        boxes = detector.detect(frame, conf_min=0.30)
        self.assertIsInstance(boxes, list)

        # Frame with synthetic person silhouette
        cv2.rectangle(frame, (200, 100), (320, 420), (255, 255, 255), -1)
        boxes = detector.detect(frame, conf_min=0.15)
        for b in boxes:
            self.assertEqual(len(b), 5)
            x1, y1, x2, y2, conf = b
            self.assertGreaterEqual(x1, 0.0)
            self.assertGreaterEqual(y1, 0.0)
            self.assertLessEqual(x2, 640.0)
            self.assertLessEqual(y2, 480.0)
            self.assertGreater(conf, 0.0)

    def test_03_tinypose_estimator(self):
        estimator = TinyPoseEstimator(self.pose_path)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200, 100), (320, 420), (255, 255, 255), -1)
        test_boxes = [(200.0, 100.0, 320.0, 420.0, 0.85)]
        kpts_list = estimator.estimate(frame, test_boxes)

        self.assertEqual(len(kpts_list), 1)
        person_kpts = kpts_list[0]
        self.assertEqual(len(person_kpts), 17)  # 17 COCO keypoints
        for x, y, conf in person_kpts:
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 640.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 480.0)
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)

    def test_04_paddle_pose_engine_duck_typing(self):
        engine = PaddlePoseEngine(MODELS_DIR, auto_download=False)
        self.assertEqual(engine.task, "pose")

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (180, 80), (300, 400), (255, 255, 255), -1)

        results = engine.predict(frame, conf=0.15)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        res = results[0]

        # Verify boxes interface
        self.assertTrue(hasattr(res, "boxes"))
        self.assertTrue(hasattr(res.boxes, "data"))
        for b in res.boxes:
            self.assertEqual(len(b.conf), 1)
            self.assertEqual(b.xyxy.shape, (1, 4))

        # Verify keypoints interface
        self.assertTrue(hasattr(res, "keypoints"))
        self.assertTrue(hasattr(res.keypoints, "data"))
        self.assertEqual(res.keypoints.data.ndim, 3)
        self.assertEqual(res.keypoints.data.shape[1], 17)
        self.assertEqual(res.keypoints.data.shape[2], 3)

        # Integration with person_detections() in person.py
        accepted, rejected = person_detections(res, 480, conf_min=0.15)
        self.assertIsInstance(accepted, list)
        self.assertIsInstance(rejected, list)

        # Integration with bay_zoom.py
        dets = detections_from_result(res, conf_min=0.15)
        self.assertIsInstance(dets, list)

        bays = [{"id": "bay_1", "name": "Bay 1", "roi": [0.1, 0.1, 0.6, 0.8], "type": "vehicle_bay"}]
        zoom_acc, zoom_rej = zoom_empty_bays(
            engine,
            frame,
            bays,
            accepted,
            rejected,
            imgsz=320,
            device=None,
            person_conf=0.15,
            min_height_frac=0.05,
            min_aspect=1.1,
            min_keypoints=3,
            kpt_conf=0.30,
        )
        self.assertIsInstance(zoom_acc, list)
        self.assertIsInstance(zoom_rej, list)

    def test_05_runtime_profile_integration(self):
        cfg_yolo = {"pose_engine": "yolo"}
        self.assertEqual(resolve_pose_engine(cfg_yolo), "yolo")

        cfg_tiny = {"pose_engine": "tinypose"}
        self.assertEqual(resolve_pose_engine(cfg_tiny), "tinypose")

        cfg_paddle = {"engine": "paddle_onnx"}
        self.assertEqual(resolve_pose_engine(cfg_paddle), "tinypose")

        profile = resolve_runtime(cfg_tiny)
        self.assertEqual(profile.pose_engine, "tinypose")

    def test_06_benchmark_tinypose_speed(self):
        engine = PaddlePoseEngine(MODELS_DIR, auto_download=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200, 100), (320, 420), (255, 255, 255), -1)

        # Warm-up
        engine.predict(frame, conf=0.15)

        iters = 10
        t0 = time.perf_counter()
        for _ in range(iters):
            engine.predict(frame, conf=0.15)
        t1 = time.perf_counter()
        avg_ms = (t1 - t0) / iters * 1000.0
        fps = 1000.0 / avg_ms
        print(f"\n[Benchmark] PaddlePoseEngine average latency: {avg_ms:.2f} ms ({fps:.1f} FPS)")
        self.assertLess(avg_ms, 120.0, "PaddlePoseEngine should run in under 120ms on CPU")


if __name__ == "__main__":
    unittest.main()
