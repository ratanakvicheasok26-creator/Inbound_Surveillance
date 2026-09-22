"""Dual-target inference profile: local edge CPU vs central GPU server.

``runtime`` in config.yaml selects the execution backend:

- ``cpu`` / ``openvino`` — ONNX Runtime / OpenVINO on a mini-PC (no GPU).
- ``cuda`` / ``tensorrt`` — NVIDIA GPU for many simultaneous RTSP streams.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard runtime dep
    cv2 = None  # type: ignore


RUNTIME_ALIASES = {
    "cpu": "cpu",
    "openvino": "openvino",
    "intel": "openvino",
    "cuda": "cuda",
    "gpu": "cuda",
    "tensorrt": "tensorrt",
    "trt": "tensorrt",
    "auto": "auto",
}

# Person tracking requires YOLO-pose keypoints. yolo11n_improved.pt is a
# detect checkpoint (vehicles / COCO boxes, no skeleton) and must never be
# used as the person model — person_detections() rejects every box without
# keypoints, so the overlay looks like "ML stopped tracking anyone".
EDGE_WEIGHTS = "yolo11n-pose.pt"
SERVER_WEIGHTS = "yolo11s-pose.pt"

# Person-detect defaults: keep far / crouched workers instead of shrinking
# 1080p to 640px and dropping anyone shorter than 12% of the frame.
DEFAULT_PERSON_CONF = 0.25
DEFAULT_MIN_PERSON_HEIGHT = 0.05
DEFAULT_MIN_ASPECT = 1.1
DEFAULT_MIN_KEYPOINTS = 3
DEFAULT_KPT_CONF = 0.35
DEFAULT_TINYPOSE_KPT_CONF = 0.12
DEFAULT_OCCUPY_CLEAR_SECONDS = 5.0
DEFAULT_UNDER_CAR_GRACE_SECONDS = 30.0
DEFAULT_IMGSZ = 640
DEFAULT_VEHICLE_IMGSZ = 512
GPU_IMGSZ = 1280
DEFAULT_BAY_ZOOM = True
DEFAULT_BAY_ZOOM_PAD = 0.08
DEFAULT_NUM_THREADS = min(4, max(1, os.cpu_count() or 4))


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    yolo_device: str | int
    weights_name: str
    dnn_backend: int
    dnn_target: int
    reid_enabled: bool
    track_max_age: int
    track_min_hits: int
    track_iou_threshold: float
    reid_match_threshold: float
    pose_engine: str = "rtmpose"

    @property
    def is_gpu(self) -> bool:
        return self.name in ("cuda", "tensorrt")


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _dnn_constants() -> tuple[int, int, int, int]:
    if cv2 is None:
        return 0, 0, 0, 0
    backend_opencv = int(getattr(cv2.dnn, "DNN_BACKEND_OPENCV", 0))
    backend_cuda = int(getattr(cv2.dnn, "DNN_BACKEND_CUDA", backend_opencv))
    target_cpu = int(getattr(cv2.dnn, "DNN_TARGET_CPU", 0))
    target_cuda = int(getattr(cv2.dnn, "DNN_TARGET_CUDA", target_cpu))
    return backend_opencv, backend_cuda, target_cpu, target_cuda


def normalize_runtime_name(raw: object) -> str:
    name = str(raw or "cpu").strip().lower()
    return RUNTIME_ALIASES.get(name, "cpu")


def resolve_person_conf(cfg: dict | None = None) -> float:
    cfg = cfg or {}
    value = cfg.get("person_conf")
    return float(value) if value is not None else DEFAULT_PERSON_CONF


def resolve_min_person_height(cfg: dict | None = None) -> float:
    cfg = cfg or {}
    value = cfg.get("min_person_height")
    return float(value) if value is not None else DEFAULT_MIN_PERSON_HEIGHT


def resolve_min_aspect(cfg: dict | None = None) -> float:
    cfg = cfg or {}
    value = cfg.get("min_aspect")
    return float(value) if value is not None else DEFAULT_MIN_ASPECT


def resolve_min_keypoints(cfg: dict | None = None) -> int:
    cfg = cfg or {}
    value = cfg.get("min_keypoints")
    return int(value) if value is not None else DEFAULT_MIN_KEYPOINTS


def resolve_kpt_conf(cfg: dict | None = None) -> float:
    cfg = cfg or {}
    value = cfg.get("kpt_conf")
    if value is not None:
        return float(value)
    if resolve_pose_engine(cfg) == "tinypose":
        return DEFAULT_TINYPOSE_KPT_CONF
    return DEFAULT_KPT_CONF


def resolve_occupy_clear_seconds(cfg: dict | None = None) -> float:
    cfg = cfg or {}
    value = cfg.get("occupy_clear_seconds")
    return float(value) if value is not None else DEFAULT_OCCUPY_CLEAR_SECONDS


def resolve_under_car_grace_seconds(cfg: dict | None = None) -> float:
    cfg = cfg or {}
    value = cfg.get("under_car_grace_seconds")
    return float(value) if value is not None else DEFAULT_UNDER_CAR_GRACE_SECONDS


def resolve_bay_zoom(cfg: dict | None = None) -> bool:
    cfg = cfg or {}
    if "bay_zoom" not in cfg:
        return DEFAULT_BAY_ZOOM
    return bool(cfg.get("bay_zoom"))


def resolve_bay_zoom_pad(cfg: dict | None = None) -> float:
    cfg = cfg or {}
    value = cfg.get("bay_zoom_pad")
    pad = float(value) if value is not None else DEFAULT_BAY_ZOOM_PAD
    return max(0.0, min(0.5, pad))


def resolve_imgsz(cfg: dict | None = None, profile: RuntimeProfile | None = None) -> int:
    """Letterboxed inference size. GPU can run 1280 when imgsz is omitted."""
    cfg = cfg or {}
    raw = cfg.get("imgsz")
    if raw is not None and str(raw).strip() != "":
        return max(32, int(raw) // 32 * 32)
    default = GPU_IMGSZ if (profile is not None and profile.is_gpu) else DEFAULT_IMGSZ
    return max(32, int(default) // 32 * 32)


def person_weights_name(raw: object | None) -> str:
    """Return a pose checkpoint name. Detect/vehicle files have no keypoints."""
    name = Path(str(raw or EDGE_WEIGHTS)).name.strip() or EDGE_WEIGHTS
    if "pose" in Path(name).stem.lower():
        return name
    return EDGE_WEIGHTS


DEFAULT_POSE_ENGINE = "rtmpose"


def resolve_pose_engine(cfg: dict | None = None) -> str:
    """Select pose inference backend: rtmpose (default), yolo, or tinypose."""
    cfg = cfg or {}
    raw = str(cfg.get("pose_engine") or cfg.get("engine") or DEFAULT_POSE_ENGINE).strip().lower()
    if raw in ("tinypose", "tiny_pose", "paddle", "paddle_onnx", "picodet_tinypose"):
        return "tinypose"
    if raw in ("yolo", "yolo11", "ultralytics", "yolov8", "yolov5"):
        return "yolo"
    return "rtmpose"


def resolve_runtime(cfg: dict | None = None) -> RuntimeProfile:
    """Pick an execution profile from config, falling back if GPU is missing."""
    cfg = cfg or {}
    requested = normalize_runtime_name(cfg.get("runtime") or "cpu")
    if requested == "auto":
        requested = "cuda" if cuda_available() else "cpu"

    name = requested
    if name in ("cuda", "tensorrt") and not cuda_available():
        print(f"[Runtime] {name} requested but CUDA is unavailable; using cpu")
        name = "cpu"

    backend_opencv, backend_cuda, target_cpu, target_cuda = _dnn_constants()
    if name in ("cuda", "tensorrt"):
        yolo_device: str | int = 0
        dnn_backend, dnn_target = backend_cuda, target_cuda
        weights_name = person_weights_name(cfg.get("server_weights") or cfg.get("weights") or SERVER_WEIGHTS)
    elif name == "openvino":
        yolo_device = "intel"
        dnn_backend, dnn_target = backend_opencv, target_cpu
        weights_name = person_weights_name(cfg.get("weights") or EDGE_WEIGHTS)
    else:
        yolo_device = "cpu"
        dnn_backend, dnn_target = backend_opencv, target_cpu
        weights_name = person_weights_name(cfg.get("weights") or EDGE_WEIGHTS)

    pose_engine = resolve_pose_engine(cfg)

    return RuntimeProfile(
        name=name,
        yolo_device=yolo_device,
        weights_name=weights_name.strip() or EDGE_WEIGHTS,
        dnn_backend=dnn_backend,
        dnn_target=dnn_target,
        reid_enabled=bool(cfg.get("enable_reid", True)),
        track_max_age=max(1, int(cfg.get("track_max_age") if cfg.get("track_max_age") is not None else 90)),
        track_min_hits=max(1, int(cfg.get("track_min_hits") if cfg.get("track_min_hits") is not None else 3)),
        track_iou_threshold=float(cfg.get("track_iou_threshold") if cfg.get("track_iou_threshold") is not None else 0.3),
        reid_match_threshold=float(
            cfg.get("reid_match_threshold") if cfg.get("reid_match_threshold") is not None else 0.50
        ),
        pose_engine=pose_engine,
    )


def apply_dnn_backend(net: Any, profile: RuntimeProfile | None) -> None:
    """Prefer CUDA on the server; OpenCV CPU/OpenVINO on the edge."""
    if net is None or profile is None or cv2 is None:
        return
    try:
        net.setPreferableBackend(profile.dnn_backend)
        net.setPreferableTarget(profile.dnn_target)
    except Exception:
        backend_opencv, _, target_cpu, _ = _dnn_constants()
        net.setPreferableBackend(backend_opencv)
        net.setPreferableTarget(target_cpu)


def resolve_weights_file(cfg: dict, resource_path, data_dir: Path) -> str:
    """Locate YOLO weights for the active runtime (engine > onnx > pt)."""
    profile = resolve_runtime(cfg)
    name = Path(profile.weights_name).name
    stem = Path(name).stem
    search_dirs = []
    try:
        bundled = Path(resource_path(name)).parent
        search_dirs.append(bundled)
    except Exception:
        pass
    search_dirs.append(Path(data_dir))

    preferred_exts: tuple[str, ...]
    if profile.name == "tensorrt":
        preferred_exts = (".engine", ".pt", ".onnx")
    elif profile.name == "openvino":
        preferred_exts = ("_openvino_model", ".onnx", ".pt")
    elif profile.name == "cuda":
        preferred_exts = (".pt", ".engine", ".onnx")
    else:
        preferred_exts = (".onnx", ".pt")

    candidates: list[str] = []
    for ext in preferred_exts:
        candidates.append(f"{stem}{ext}")
    if Path(profile.weights_name).suffix and name not in candidates:
        candidates.append(name)
    seen: set[str] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        for folder in search_dirs:
            path = folder / cand
            if path.is_file() or (cand.endswith("_openvino_model") and path.is_dir()):
                return str(path)
    return str(profile.weights_name)


def configure_onnx_session_options(num_threads: int = DEFAULT_NUM_THREADS):
    """Configures ONNX Runtime SessionOptions with thread pinning."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts


def configure_openvino_properties(core, num_threads: int = DEFAULT_NUM_THREADS):
    """Configures OpenVINO Core properties to prioritize low latency on edge CPUs."""
    core.set_property(
        "CPU",
        {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_NUM_THREADS": str(num_threads),
            "NUM_STREAMS": "1",
        },
    )


def benchmark_pose(model_path: str, imgsz: int = 640, runs: int = 60) -> float:
    """Executes warmup and benchmark inferences to verify execution latency.
    Returns: Average execution latency in milliseconds.
    """
    path = Path(model_path)
    if path.suffix == ".onnx":
        try:
            import onnxruntime as ort

            opts = configure_onnx_session_options()
            sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name
            dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
            for _ in range(5):
                sess.run(None, {input_name: dummy})
            t0 = time.perf_counter()
            for _ in range(runs):
                sess.run(None, {input_name: dummy})
            elapsed = time.perf_counter() - t0
            latency_ms = (elapsed / runs) * 1000.0
            fps = runs / elapsed
            print(f"[BENCHMARK] {model_path} ({imgsz}x{imgsz}) [ORT]: {latency_ms:.2f} ms/frame ({fps:.1f} FPS)")
            return latency_ms
        except Exception as ex:
            print(f"[BENCHMARK] Direct ORT benchmark skipped: {ex}")
    elif (path.is_dir() and (path / "yolo11n-pose.xml").exists()) or path.suffix == ".xml":
        try:
            import openvino as ov

            core = ov.Core()
            configure_openvino_properties(core)
            xml_path = str(path / "yolo11n-pose.xml") if path.is_dir() else str(path)
            model = core.read_model(xml_path)
            compiled = core.compile_model(model, "CPU")
            dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
            req = compiled.create_infer_request()
            for _ in range(5):
                req.infer([dummy])
            t0 = time.perf_counter()
            for _ in range(runs):
                req.infer([dummy])
            elapsed = time.perf_counter() - t0
            latency_ms = (elapsed / runs) * 1000.0
            fps = runs / elapsed
            print(f"[BENCHMARK] {model_path} ({imgsz}x{imgsz}) [OpenVINO]: {latency_ms:.2f} ms/frame ({fps:.1f} FPS)")
            return latency_ms
        except Exception as ex:
            print(f"[BENCHMARK] Direct OpenVINO benchmark skipped: {ex}")

    from ultralytics import YOLO

    model = YOLO(str(model_path), task="pose")
    dummy_frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        _ = model(dummy_frame, imgsz=imgsz, verbose=False)

    # Timed run
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model(dummy_frame, imgsz=imgsz, verbose=False)
    elapsed = time.perf_counter() - t0

    latency_ms = (elapsed / runs) * 1000.0
    fps = runs / elapsed
    print(f"[BENCHMARK] {model_path} ({imgsz}x{imgsz}): {latency_ms:.2f} ms/frame ({fps:.1f} FPS)")
    return latency_ms

