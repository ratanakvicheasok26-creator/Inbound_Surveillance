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
    RTMPOSE_MODES,
    RTMPoseEngine,
    load_person_pose_model,
    resolve_models_dir,
    resolve_rtmpose_mode,
)
from runtime import (
    DEFAULT_NUM_THREADS,
    physical_core_count,
    resolve_num_threads,
    resolve_pose_engine,
    resolve_runtime,
)


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


_DET_ONNX = ROOT / "models" / "yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx"
_POSE_ONNX = ROOT / "models" / "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx"


class _FakeRtmlibModel:
    """Stands in for a rtmlib YOLOX/RTMPose wrapper: a real ONNX path plus the
    default (untuned) session rtmlib would have built."""

    def __init__(self):
        import onnxruntime as ort

        if not _DET_ONNX.is_file():
            raise unittest.SkipTest(f"{_DET_ONNX.name} not present")
        self.onnx_model = str(_DET_ONNX)
        # Mirrors rtmlib/tools/base.py: no SessionOptions at all.
        self.session = ort.InferenceSession(self.onnx_model, providers=["CPUExecutionProvider"])

    def __call__(self, *args, **kwargs):
        return np.zeros((0, 5), dtype=np.float32)


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


class ThreadBudgetTests(unittest.TestCase):
    """rtmlib builds its ORT session with no SessionOptions, so the intra-op pool
    defaults to the *logical* CPU count. On an SMT box that is the slowest
    measured setting, so the engine has to pin itself to physical cores."""

    def test_thread_default_never_exceeds_physical_cores(self):
        import os

        phys = physical_core_count()
        logical = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        self.assertGreaterEqual(phys, 1)
        self.assertLessEqual(
            phys,
            logical,
            "physical_core_count must never exceed the CPUs actually available",
        )
        self.assertEqual(DEFAULT_NUM_THREADS, phys)

    def test_num_threads_config_overrides_autodetect(self):
        self.assertEqual(resolve_num_threads({}), physical_core_count())
        self.assertEqual(resolve_num_threads({"num_threads": 1}), 1)
        self.assertEqual(resolve_num_threads({"num_threads": "3"}), 3)

    def test_invalid_num_threads_falls_back_instead_of_crashing(self):
        for bad in ("x", "", None, []):
            self.assertEqual(resolve_num_threads({"num_threads": bad}), physical_core_count())

    def test_engine_applies_thread_budget_to_rtmlib_sessions(self):
        """Injected models still get re-tuned, so tests and prod agree."""
        det = _FakeRtmlibModel()
        pose = _FakeRtmlibModel()
        profile = resolve_runtime({"pose_engine": "rtmpose", "runtime": "cpu"})
        engine = RTMPoseEngine(
            models_dir=ROOT / "models",
            runtime_profile=profile,
            det_model=det,
            pose_model=pose,
            num_threads=1,
        )
        self.assertEqual(engine.num_threads, 1)
        self.assertEqual(det.session.get_session_options().intra_op_num_threads, 1)
        self.assertEqual(pose.session.get_session_options().intra_op_num_threads, 1)

    def test_tuning_never_raises(self):
        class _Broken:
            onnx_model = ""

        det, pose = _Broken(), _Broken()
        engine = RTMPoseEngine(
            models_dir=ROOT / "models",
            runtime_profile=resolve_runtime({"runtime": "cpu"}),
            det_model=det,
            pose_model=pose,
            num_threads=2,
        )
        self.assertIs(engine.det_model, det, "a bad model must be left alone, not replaced")

    def test_openvino_backend_keeps_rtmlib_wiring(self):
        """The openvino branch of _tune_sessions had no coverage at all.

        It rebuilds the compiled model by hand, so if it disagrees with
        rtmlib/tools/base.py about the device name or the output count,
        predict() breaks only for `runtime: openvino` boxes -- the config
        example explicitly tells operators that runtime is a valid choice.
        """
        try:
            from rtmlib import RTMPose, YOLOX
        except Exception as exc:  # pragma: no cover - rtmlib always present in prod
            self.skipTest(f"rtmlib unavailable: {exc}")

        if not _DET_ONNX.is_file() or not _POSE_ONNX.is_file():
            self.skipTest("RTMPose ONNX checkpoints not present")

        # Build the detector exactly the way rtmpose.py does, then let the
        # engine tune it. Both models are injected so the engine takes the
        # injected-model path instead of re-resolving the checkpoints.
        spec = RTMPOSE_MODES["lightweight"]
        det = YOLOX(
            str(_DET_ONNX),
            model_input_size=spec["det_input_size"],
            backend="openvino",
            device="cpu",
            score_thr=0.25,
            nms_thr=0.45,
        )
        pose = RTMPose(
            str(_POSE_ONNX),
            model_input_size=spec["pose_input_size"],
            backend="openvino",
            device="cpu",
            to_openpose=False,
        )
        expected_outputs = list(det._ov_outputs)
        self.assertTrue(expected_outputs, "rtmlib produced no output layers")
        baseline_compiled = det.compiled_model

        engine = RTMPoseEngine(
            models_dir=ROOT / "models",
            runtime_profile=resolve_runtime({"runtime": "openvino"}),
            det_model=det,
            pose_model=pose,
            num_threads=1,
        )
        self.assertEqual((engine.backend, engine.device), ("openvino", "cpu"))

        det = engine.det_model
        self.assertIsNot(det.compiled_model, baseline_compiled, "model was not recompiled")
        self.assertEqual(len(det._ov_outputs), len(expected_outputs))
        self.assertIsNotNone(det.input_layer)
        # The property must land on the same device rtmlib compiled for. OpenVINO
        # 2026 coerces INFERENCE_NUM_THREADS to int on read-back, older builds
        # return the string it was set with.
        self.assertEqual(
            int(det.core.get_property("CPU", "INFERENCE_NUM_THREADS")),
            1,
            "thread budget never reached the OpenVINO core",
        )
        # And the recompiled detector still runs.
        result = engine.predict(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
