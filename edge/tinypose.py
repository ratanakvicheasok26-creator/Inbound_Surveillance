"""PP-TinyPose and PP-PicoDet inference engine via ONNX Runtime.

Provides high-efficiency CPU / edge keypoint estimation and pedestrian detection
developed by Baidu PaddlePaddle / PaddleDetection.

Exposes `PaddlePoseEngine` as a drop-in duck-typed replacement for `ultralytics.YOLO`,
producing results directly compatible with `person.py`, `bay_zoom.py`, and `occupancy.py`.
"""

from __future__ import annotations

import os
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

PICODET_FILENAME = "picodet_s_320_lcnet_pedestrian.onnx"
TINYPOSE_FILENAME = "tinypose_256_192.onnx"

PICODET_URLS = (
    "https://raw.githubusercontent.com/guojin-yan/Csharp_and_OpenVINO_deploy_PP-TinyPose/master/model/picodet_v2_s_320_pedestrian/picodet_s_320_lcnet_pedestrian.onnx",
)
TINYPOSE_URLS = (
    "https://raw.githubusercontent.com/guojin-yan/Csharp_and_OpenVINO_deploy_PP-TinyPose/master/model/tinypose_256_192/tinypose_256_192.onnx",
)

# Standard ImageNet normalization used by PaddleDetection
PADDLE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
PADDLE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# TinyPose input dimensions (Height=256, Width=192) -> Aspect ratio 4:3
TINYPOSE_INPUT_H = 256
TINYPOSE_INPUT_W = 192
TINYPOSE_HM_H = 64
TINYPOSE_HM_W = 48
TINYPOSE_SCALE_X = TINYPOSE_INPUT_W / TINYPOSE_HM_W  # 4.0
TINYPOSE_SCALE_Y = TINYPOSE_INPUT_H / TINYPOSE_HM_H  # 4.0

# PicoDet input dimensions (Height=320, Width=320)
PICODET_INPUT_SIZE = 320


def ensure_tinypose_models(models_dir: Path, download: bool = True) -> tuple[Path, Path]:
    """Verify presence of PicoDet and TinyPose ONNX models, downloading if missing."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    pico_path = models_dir / PICODET_FILENAME
    pose_path = models_dir / TINYPOSE_FILENAME

    if not pico_path.is_file() or pico_path.stat().st_size < 1_000_000:
        if not download:
            raise FileNotFoundError(f"PicoDet ONNX model missing at {pico_path}")
        print(f"[TinyPose] Downloading {PICODET_FILENAME} (~4.8 MB)...", flush=True)
        downloaded = False
        for url in PICODET_URLS:
            try:
                urllib.request.urlretrieve(url, pico_path)
                if pico_path.stat().st_size > 1_000_000:
                    downloaded = True
                    break
            except Exception as e:
                print(f"[TinyPose] Download error from {url}: {e}", flush=True)
        if not downloaded or not pico_path.is_file():
            raise RuntimeError(f"Failed to download {PICODET_FILENAME}")

    if not pose_path.is_file() or pose_path.stat().st_size < 1_000_000:
        if not download:
            raise FileNotFoundError(f"TinyPose ONNX model missing at {pose_path}")
        print(f"[TinyPose] Downloading {TINYPOSE_FILENAME} (~5.6 MB)...", flush=True)
        downloaded = False
        for url in TINYPOSE_URLS:
            try:
                urllib.request.urlretrieve(url, pose_path)
                if pose_path.stat().st_size > 1_000_000:
                    downloaded = True
                    break
            except Exception as e:
                print(f"[TinyPose] Download error from {url}: {e}", flush=True)
        if not downloaded or not pose_path.is_file():
            raise RuntimeError(f"Failed to download {TINYPOSE_FILENAME}")

    return pico_path, pose_path


class PicoDetDetector:
    """ONNX Runtime detector for PicoDet pedestrian model (320x320)."""

    def __init__(self, model_path: Path, providers: list[str] | None = None):
        if ort is None:
            raise ImportError("onnxruntime is required for PicoDet inference")
        self.model_path = Path(model_path)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), opts, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def detect(
        self,
        frame: np.ndarray,
        conf_min: float = 0.25,
        iou_thresh: float = 0.45,
    ) -> list[tuple[float, float, float, float, float]]:
        """Run pedestrian detection on an image.

        Returns:
            List of (x1, y1, x2, y2, conf) in original frame coordinates.
        """
        h_orig, w_orig = frame.shape[:2]
        if h_orig < 8 or w_orig < 8:
            return []

        # Resize to 320x320 and normalize
        resized = cv2.resize(frame, (PICODET_INPUT_SIZE, PICODET_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - PADDLE_MEAN) / PADDLE_STD
        blob = np.ascontiguousarray(np.transpose(norm, (2, 0, 1))[np.newaxis, :, :, :], dtype=np.float32)

        outputs = self.session.run(None, {self.input_name: blob})
        boxes = outputs[0][0]     # Shape (2125, 4) in 320x320 space
        scores = outputs[1][0, 0] # Shape (2125,) class 0 scores

        mask = scores >= conf_min
        if not np.any(mask):
            return []

        valid_boxes = boxes[mask]
        valid_scores = scores[mask]

        # Rescale boxes to original frame coordinates
        scale_x = w_orig / float(PICODET_INPUT_SIZE)
        scale_y = h_orig / float(PICODET_INPUT_SIZE)

        scaled_boxes_xywh = []
        scores_list = []
        for b, s in zip(valid_boxes, valid_scores):
            x1 = float(max(0.0, min(w_orig, b[0] * scale_x)))
            y1 = float(max(0.0, min(h_orig, b[1] * scale_y)))
            x2 = float(max(0.0, min(w_orig, b[2] * scale_x)))
            y2 = float(max(0.0, min(h_orig, b[3] * scale_y)))
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w > 1.0 and h > 1.0:
                scaled_boxes_xywh.append([int(x1), int(y1), int(w), int(h)])
                scores_list.append(float(s))

        if not scaled_boxes_xywh:
            return []

        indices = cv2.dnn.NMSBoxes(scaled_boxes_xywh, scores_list, conf_min, iou_thresh)
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, bw, bh = scaled_boxes_xywh[idx]
                results.append((float(bx), float(by), float(bx + bw), float(by + bh), scores_list[idx]))

        return results


class TinyPoseEstimator:
    """ONNX Runtime pose estimator for PP-TinyPose (256x192)."""

    def __init__(self, model_path: Path, providers: list[str] | None = None):
        if ort is None:
            raise ImportError("onnxruntime is required for TinyPose inference")
        self.model_path = Path(model_path)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), opts, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name

    def estimate(
        self,
        frame: np.ndarray,
        boxes: list[tuple[float, float, float, float, float]],
        kpt_conf_floor: float = 0.05,
    ) -> list[list[tuple[float, float, float]]]:
        """Run 17-keypoint pose estimation on detected person bounding boxes.

        Args:
            frame: Full BGR frame.
            boxes: List of (x1, y1, x2, y2, conf).
            kpt_conf_floor: Minimum confidence to consider a keypoint detected.

        Returns:
            List of 17 keypoints [(x, y, conf), ...] per person in frame coordinates.
        """
        frame_h, frame_w = frame.shape[:2]
        all_keypoints: list[list[tuple[float, float, float]]] = []

        for x1, y1, x2, y2, _ in boxes:
            bw = max(x2 - x1, 1.0)
            bh = max(y2 - y1, 1.0)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Expand with aspect ratio matching TinyPose 3:4 (width:height = 192:256)
            # Add 15% margin around the person to capture head & limbs without clipping
            aspect = TINYPOSE_INPUT_W / float(TINYPOSE_INPUT_H)  # 0.75
            if bw > bh * aspect:
                cw = bw * 1.15
                ch = cw / aspect
            else:
                ch = bh * 1.15
                cw = ch * aspect

            crop_x1 = max(0, int(round(cx - cw / 2.0)))
            crop_y1 = max(0, int(round(cy - ch / 2.0)))
            crop_x2 = min(frame_w, int(round(cx + cw / 2.0)))
            crop_y2 = min(frame_h, int(round(cy + ch / 2.0)))

            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            if crop_w < 8 or crop_h < 8:
                all_keypoints.append([(0.0, 0.0, 0.0)] * 17)
                continue

            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            resized_crop = cv2.resize(crop, (TINYPOSE_INPUT_W, TINYPOSE_INPUT_H), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            norm = (rgb - PADDLE_MEAN) / PADDLE_STD
            blob = np.ascontiguousarray(np.transpose(norm, (2, 0, 1))[np.newaxis, :, :, :], dtype=np.float32)

            outputs = self.session.run(None, {self.input_name: blob})
            heatmap = outputs[0][0]  # Shape (17, 64, 48)

            person_kpts: list[tuple[float, float, float]] = []
            for k in range(17):
                hm_slice = heatmap[k]
                idx = int(np.argmax(hm_slice))
                py, px = divmod(idx, TINYPOSE_HM_W)
                conf = float(hm_slice[py, px])

                # DarkPose subpixel refinement
                dx, dy = 0.0, 0.0
                if 0 < px < TINYPOSE_HM_W - 1:
                    dx = float(hm_slice[py, px + 1] - hm_slice[py, px - 1])
                if 0 < py < TINYPOSE_HM_H - 1:
                    dy = float(hm_slice[py + 1, px] - hm_slice[py - 1, px])

                refined_px = px + 0.25 * float(np.sign(dx))
                refined_py = py + 0.25 * float(np.sign(dy))

                crop_coord_x = refined_px * TINYPOSE_SCALE_X
                crop_coord_y = refined_py * TINYPOSE_SCALE_Y

                # Map back to full frame space
                frame_coord_x = crop_x1 + (crop_coord_x / float(TINYPOSE_INPUT_W)) * crop_w
                frame_coord_y = crop_y1 + (crop_coord_y / float(TINYPOSE_INPUT_H)) * crop_h

                clamped_x = float(max(0.0, min(frame_w, frame_coord_x)))
                clamped_y = float(max(0.0, min(frame_h, frame_coord_y)))
                kpt_conf = max(0.0, min(1.0, conf)) if conf >= kpt_conf_floor else 0.0

                person_kpts.append((clamped_x, clamped_y, kpt_conf))

            all_keypoints.append(person_kpts)

        return all_keypoints


class PaddlePoseBox:
    """Duck-typed bounding box matching Ultralytics box interface."""

    def __init__(self, x1: float, y1: float, x2: float, y2: float, conf: float):
        self.conf = np.array([conf], dtype=np.float32)
        self.xyxy = np.array([[x1, y1, x2, y2]], dtype=np.float32)


class PaddlePoseKeypoints:
    """Duck-typed keypoints container matching Ultralytics result.keypoints."""

    def __init__(self, keypoint_array: np.ndarray):
        # Shape: (N, 17, 3) where elements are [x, y, conf]
        self.data = keypoint_array


class PaddlePoseBoxesList(list):
    """List of boxes with .data attribute for Ultralytics compatibility."""

    def __init__(self, boxes: list[PaddlePoseBox]):
        super().__init__(boxes)
        if boxes:
            # (N, 6) matrix of [x1, y1, x2, y2, conf, cls=0]
            rows = []
            for b in boxes:
                x1, y1, x2, y2 = b.xyxy[0]
                rows.append([x1, y1, x2, y2, b.conf[0], 0.0])
            self.data = np.array(rows, dtype=np.float32)
        else:
            self.data = np.zeros((0, 6), dtype=np.float32)


class PaddlePoseResult:
    """Duck-typed inference result matching Ultralytics Result object."""

    def __init__(
        self,
        boxes: list[PaddlePoseBox],
        keypoints_data: np.ndarray,
        orig_shape: tuple[int, int],
    ):
        self.boxes = PaddlePoseBoxesList(boxes)
        self.keypoints = PaddlePoseKeypoints(keypoints_data)
        self.orig_shape = orig_shape


class PaddlePoseEngine:
    """Duck-typed drop-in replacement for ultralytics.YOLO pose model.

    Combines PicoDet pedestrian detector and PP-TinyPose pose estimator.
    """

    task = "pose"

    def __init__(
        self,
        models_dir: Path | str | None = None,
        providers: list[str] | None = None,
        runtime_profile: Any = None,
        auto_download: bool = True,
    ):
        if models_dir is None:
            models_dir = Path(__file__).resolve().parent / "models"
        self.models_dir = Path(models_dir)
        self.lock = threading.Lock()

        # Determine execution providers
        if providers is None and runtime_profile is not None:
            if getattr(runtime_profile, "name", "") in ("cuda", "tensorrt"):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif getattr(runtime_profile, "name", "") == "openvino":
                providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        elif providers is None:
            providers = ["CPUExecutionProvider"]

        pico_path, pose_path = ensure_tinypose_models(self.models_dir, download=auto_download)
        self.detector = PicoDetDetector(pico_path, providers=providers)
        self.pose_estimator = TinyPoseEstimator(pose_path, providers=providers)
        self.providers = providers
        print(
            f"[PaddlePoseEngine] Loaded PP-PicoDet ({pico_path.name}) & PP-TinyPose ({pose_path.name}) "
            f"with providers={providers}",
            flush=True,
        )

    def predict(
        self,
        source: np.ndarray,
        imgsz: int = 320,
        conf: float = 0.25,
        device: Any = None,
        verbose: bool = False,
        **kwargs,
    ) -> list[PaddlePoseResult]:
        """Predict boxes and keypoints on a frame, matching the Ultralytics API.

        Returns:
            List with a single `PaddlePoseResult` object for source image.
        """
        if source is None or not isinstance(source, np.ndarray) or source.size == 0:
            empty_kpts = np.zeros((0, 17, 3), dtype=np.float32)
            return [PaddlePoseResult([], empty_kpts, (0, 0))]

        h, w = source.shape[:2]
        with self.lock:
            # 1. Detect person boxes using PicoDet
            detected_boxes = self.detector.detect(source, conf_min=conf)
            if not detected_boxes:
                empty_kpts = np.zeros((0, 17, 3), dtype=np.float32)
                return [PaddlePoseResult([], empty_kpts, (h, w))]

            # 2. Run PP-TinyPose on detected boxes
            keypoints_list = self.pose_estimator.estimate(source, detected_boxes)

            # 3. Format into duck-typed result
            boxes: list[PaddlePoseBox] = []
            for b in detected_boxes:
                boxes.append(PaddlePoseBox(b[0], b[1], b[2], b[3], b[4]))

            keypoints_data = np.array(keypoints_list, dtype=np.float32)
            return [PaddlePoseResult(boxes, keypoints_data, (h, w))]
