"""OpenMMLab RTMPose person engine via rtmlib.

Replaces YOLO-pose / PP-TinyPose for person boxes + COCO-17 skeletons.
Vehicle detection stays on Ultralytics YOLO.

Uses official RTMPose checkpoints (Apache-2.0) from
https://github.com/open-mmlab/mmpose and the lightweight rtmlib runtime
https://github.com/Tau-J/rtmlib — YOLOX person detector, then RTMPose on
each box. Output is duck-typed to match ``ultralytics.YOLO.predict`` so
``person.py``, ``bay_zoom.py``, and ``occupancy.py`` stay unchanged.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from runtime import DEFAULT_NUM_THREADS, resolve_num_threads

# YOLOX-tiny + RTMPose-s (CPU) / YOLOX-m + RTMPose-m (GPU). URLs are the
# official OpenMMLab ONNX SDK zips; rtmlib also mirrors them on Hugging Face.
RTMPOSE_MODES: dict[str, dict[str, Any]] = {
    "performance": {
        "det": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip"
        ),
        "det_input_size": (640, 640),
        "pose": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip"
        ),
        "pose_input_size": (288, 384),
    },
    "lightweight": {
        "det": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip"
        ),
        "det_input_size": (416, 416),
        "pose": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip"
        ),
        "pose_input_size": (192, 256),
    },
    "balanced": {
        "det": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
        ),
        "det_input_size": (640, 640),
        "pose": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
        ),
        "pose_input_size": (192, 256),
    },
}


class PoseBox:
    """Duck-typed bounding box matching the Ultralytics box interface."""

    def __init__(self, x1: float, y1: float, x2: float, y2: float, conf: float):
        self.conf = np.array([conf], dtype=np.float32)
        self.xyxy = np.array([[x1, y1, x2, y2]], dtype=np.float32)


class PoseKeypoints:
    """Duck-typed keypoints container matching ``result.keypoints.data``."""

    def __init__(self, keypoint_array: np.ndarray):
        self.data = keypoint_array


class PoseBoxesList(list):
    """List of boxes with a ``.data`` matrix for Ultralytics compatibility."""

    def __init__(self, boxes: list[PoseBox]):
        super().__init__(boxes)
        if boxes:
            rows = []
            for b in boxes:
                x1, y1, x2, y2 = b.xyxy[0]
                rows.append([x1, y1, x2, y2, b.conf[0], 0.0])
            self.data = np.array(rows, dtype=np.float32)
        else:
            self.data = np.zeros((0, 6), dtype=np.float32)


class PoseResult:
    """Duck-typed inference result matching an Ultralytics Result."""

    def __init__(
        self,
        boxes: list[PoseBox],
        keypoints_data: np.ndarray,
        orig_shape: tuple[int, int],
    ):
        self.boxes = PoseBoxesList(boxes)
        self.keypoints = PoseKeypoints(keypoints_data)
        self.orig_shape = orig_shape


def resolve_models_dir(resource_path, data_dir: Path) -> Path:
    """Prefer bundled models/, otherwise the writable data dir."""
    try:
        bundled = Path(resource_path("models"))
        if bundled.is_dir():
            return bundled
    except Exception:
        pass
    dest = Path(data_dir) / "models"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def resolve_rtmpose_mode(cfg: dict | None, runtime_profile: Any = None) -> str:
    cfg = cfg or {}
    raw = str(cfg.get("rtmpose_mode") or "").strip().lower()
    if raw in RTMPOSE_MODES:
        return raw
    if runtime_profile is not None and getattr(runtime_profile, "is_gpu", False):
        return "balanced"
    return "lightweight"


def _backend_device(runtime_profile: Any) -> tuple[str, str]:
    name = getattr(runtime_profile, "name", "") if runtime_profile is not None else ""
    if name in ("cuda", "tensorrt"):
        return "onnxruntime", "cuda"
    if name == "openvino":
        return "openvino", "cpu"
    return "onnxruntime", "cpu"


def _as_xyxy_rows(bboxes: Any) -> np.ndarray:
    if bboxes is None:
        return np.zeros((0, 4), dtype=np.float32)
    arr = np.asarray(bboxes, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[:, :4]


def empty_pose_result(orig_shape: tuple[int, int] = (0, 0)) -> PoseResult:
    return PoseResult([], np.zeros((0, 17, 3), dtype=np.float32), orig_shape)


class RTMPoseEngine:
    """Duck-typed drop-in replacement for ``ultralytics.YOLO`` pose models."""

    task = "pose"

    def __init__(
        self,
        models_dir: Path | str | None = None,
        runtime_profile: Any = None,
        mode: str = "lightweight",
        auto_download: bool = True,
        det_model: Any = None,
        pose_model: Any = None,
        num_threads: int | None = None,
    ):
        if models_dir is None:
            models_dir = Path(__file__).resolve().parent / "models"
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.mode = mode if mode in RTMPOSE_MODES else "lightweight"
        self.backend, self.device = _backend_device(runtime_profile)
        self.num_threads = int(num_threads) if num_threads else DEFAULT_NUM_THREADS

        if det_model is not None and pose_model is not None:
            self.det_model = det_model
            self.pose_model = pose_model
            self._tune_sessions()
            return

        if not auto_download:
            raise FileNotFoundError(
                f"RTMPose models were not injected and auto_download is False ({self.models_dir})"
            )

        from rtmlib import RTMPose, YOLOX
        from rtmlib.tools.file import download_checkpoint

        spec = RTMPOSE_MODES[self.mode]
        det_path = download_checkpoint(spec["det"], dst_dir=str(self.models_dir))
        pose_path = download_checkpoint(spec["pose"], dst_dir=str(self.models_dir))
        self.det_model = YOLOX(
            det_path,
            model_input_size=spec["det_input_size"],
            backend=self.backend,
            device=self.device,
            score_thr=0.25,
            nms_thr=0.45,
        )
        self.pose_model = RTMPose(
            pose_path,
            model_input_size=spec["pose_input_size"],
            backend=self.backend,
            device=self.device,
            to_openpose=False,
        )
        self._tune_sessions()
        print(
            f"[RTMPoseEngine] Loaded YOLOX+RTMPose mode={self.mode} "
            f"backend={self.backend} device={self.device} threads={self.num_threads}",
            flush=True,
        )

    def _tune_sessions(self) -> None:
        """Rebuild rtmlib's inference sessions with an explicit thread budget.

        rtmlib constructs ``ort.InferenceSession`` with no SessionOptions, so
        ONNX Runtime defaults intra-op threads to the *logical* CPU count. On an
        SMT box that is the slowest setting measured (191ms vs 149ms for the
        YOLOX-tiny detector on 2c/4t). Pinning to physical cores is a free win
        and also stops the infer thread from fighting the capture and JPEG
        threads for the same cores.
        """
        if self.num_threads <= 0:
            return
        for name in ("det_model", "pose_model"):
            model = getattr(self, name, None)
            if model is None:
                continue
            try:
                if self.backend == "onnxruntime":
                    self._tune_ort_session(model)
                elif self.backend == "openvino":
                    self._tune_openvino_model(model)
            except Exception as ex:
                print(f"[RTMPoseEngine] thread tuning skipped for {name}: {ex}")

    def _tune_ort_session(self, model: Any) -> None:
        import onnxruntime as ort

        from runtime import configure_onnx_session_options

        path = getattr(model, "onnx_model", None)
        if not path:
            return
        # Same provider rtmlib resolved, via the same table, so tuning cannot
        # move the model to a different accelerator (or request one this box
        # does not have, which ORT raises on rather than falling back).
        from rtmlib.tools.base import RTMLIB_SETTINGS

        provider = RTMLIB_SETTINGS.get("onnxruntime", {}).get(
            self.device, "CPUExecutionProvider"
        )
        tuned = ort.InferenceSession(
            str(path),
            configure_onnx_session_options(self.num_threads),
            providers=[provider],
        )
        if [o.name for o in tuned.get_outputs()] != [o.name for o in model.session.get_outputs()]:
            raise RuntimeError("session output signature changed; keeping rtmlib default session")
        model.session = tuned

    def _tune_openvino_model(self, model: Any) -> None:
        import openvino as ov

        from runtime import configure_openvino_properties

        core = getattr(model, "core", None)
        if core is None:
            core = ov.Core()
            model.core = core
        configure_openvino_properties(core, self.num_threads)
        path = getattr(model, "onnx_model", None)
        if not path:
            return
        # Compile on the same device rtmlib picked, or tuning would silently
        # move the model off the accelerator it was loaded on. rtmlib resolves
        # it through RTMLIB_SETTINGS['openvino'] ('cpu'->CPU, 'gpu'->GPU, ...).
        from rtmlib.tools.base import RTMLIB_SETTINGS

        device = RTMLIB_SETTINGS.get("openvino", {}).get(self.device, self.device.upper())
        read_model = core.read_model(str(path))
        compiled = core.compile_model(read_model, device)
        model.compiled_model = compiled
        model.input_layer = compiled.input(0)
        # rtmlib indexes outputs off the *read* model's output count, not the
        # compiled one; mirror it so predict() finds the same layers.
        model._ov_outputs = [compiled.output(i) for i in range(len(read_model.outputs))]

    def predict(
        self,
        source: np.ndarray,
        imgsz: int = 640,
        conf: float = 0.25,
        device: Any = None,
        verbose: bool = False,
        **kwargs,
    ) -> list[PoseResult]:
        del imgsz, device, verbose, kwargs
        if source is None or not isinstance(source, np.ndarray) or source.size == 0:
            return [empty_pose_result()]

        h, w = source.shape[:2]
        with self.lock:
            if hasattr(self.det_model, "score_thr"):
                try:
                    self.det_model.score_thr = float(conf)
                except Exception:
                    pass
            raw_boxes = self.det_model(source)
            boxes_xyxy = _as_xyxy_rows(raw_boxes)
            if boxes_xyxy.shape[0] == 0:
                return [empty_pose_result((h, w))]

            keypoints, scores = self.pose_model(source, bboxes=boxes_xyxy)
            kpts = np.asarray(keypoints, dtype=np.float32)
            sc = np.asarray(scores, dtype=np.float32)
            if kpts.ndim == 2:
                kpts = kpts.reshape(1, -1, 2)
            if sc.ndim == 1:
                sc = sc.reshape(1, -1)
            if kpts.shape[0] != boxes_xyxy.shape[0]:
                n = min(kpts.shape[0], boxes_xyxy.shape[0])
                kpts = kpts[:n]
                sc = sc[:n]
                boxes_xyxy = boxes_xyxy[:n]
            if sc.shape[:2] != kpts.shape[:2]:
                sc = np.resize(sc, kpts.shape[:2])

            packed = np.concatenate([kpts, sc[..., None]], axis=-1)
            packed[:, :, 0] = np.clip(packed[:, :, 0], 0.0, float(w))
            packed[:, :, 1] = np.clip(packed[:, :, 1], 0.0, float(h))
            packed[:, :, 2] = np.clip(packed[:, :, 2], 0.0, 1.0)

            out_boxes: list[PoseBox] = []
            for i, xyxy in enumerate(boxes_xyxy):
                x1, y1, x2, y2 = (float(v) for v in xyxy[:4])
                joint_conf = float(np.mean(packed[i, :, 2])) if packed.shape[0] > i else float(conf)
                box_conf = float(np.clip(max(joint_conf, float(conf)), 0.0, 1.0))
                out_boxes.append(PoseBox(x1, y1, x2, y2, box_conf))
            return [PoseResult(out_boxes, packed, (h, w))]

    def __call__(self, source: np.ndarray, **kwargs) -> list[PoseResult]:
        return self.predict(source, **kwargs)


def load_person_pose_model(
    profile: Any,
    *,
    models_dir: Path,
    weights_path: str | Path,
    cfg: dict | None = None,
    yolo_cls: Any = None,
):
    """Load the configured person pose backend, falling back to YOLO-pose."""
    engine = getattr(profile, "pose_engine", "rtmpose")
    if engine == "rtmpose":
        try:
            mode = resolve_rtmpose_mode(cfg, profile)
            model = RTMPoseEngine(
                models_dir=models_dir,
                runtime_profile=profile,
                mode=mode,
                num_threads=resolve_num_threads(cfg),
            )
            print(f"[Pose] Initialized RTMPose engine mode={mode} ({getattr(profile, 'name', '')})", flush=True)
            return model
        except Exception as ex:
            print(f"[Pose] RTMPose init failed ({ex}); falling back to YOLO-pose", flush=True)
    elif engine == "tinypose":
        try:
            from tinypose import PaddlePoseEngine

            model = PaddlePoseEngine(models_dir=models_dir, runtime_profile=profile)
            print(f"[Pose] Initialized PP-TinyPose engine ({getattr(profile, 'name', '')})", flush=True)
            return model
        except Exception as ex:
            print(f"[Pose] PP-TinyPose init failed ({ex}); falling back to YOLO-pose", flush=True)

    if yolo_cls is None:
        from ultralytics import YOLO

        yolo_cls = YOLO
    model = yolo_cls(str(weights_path), task="pose")
    print(f"[Pose] Initialized YOLO-pose ({weights_path})", flush=True)
    return model
