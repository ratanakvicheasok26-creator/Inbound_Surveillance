"""Video file camera adapter: real-time 1.0x playback with seamless infinite looping."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from adapters.base import (
    BaseCameraAdapter,
    FramePacket,
    decode_file_uri,
    packet_from_bgr,
)


def resolve_video_path(raw_path: str | Path) -> Path:
    raw = decode_file_uri(str(raw_path or "").strip())

    p = Path(raw)
    if p.is_file():
        return p.resolve()

    from paths import data_dir, is_frozen, resource_dir
    import sys

    filename = p.name
    if "?" in filename:
        filename = filename.split("?")[0]

    resource = resource_dir()
    data_videos = data_dir() / "videos"

    candidates: list[Path] = []
    if not p.is_absolute():
        candidates.extend([
            p,
            Path.cwd() / p,
            resource / p,
            resource / "videos" / p,
            data_videos / p,
        ])

    if filename:
        candidates.extend([
            resource / "videos" / filename,
            resource / filename,
            data_videos / filename,
            data_dir() / filename,
            Path.cwd() / "edge" / "videos" / filename,
            Path.cwd() / "videos" / filename,
            Path.cwd() / filename,
            Path(__file__).resolve().parent.parent / "videos" / filename,
        ])
        exe_parent = Path(sys.executable).resolve().parent
        candidates.extend([
            exe_parent / "videos" / filename,
            exe_parent / "edge" / "videos" / filename,
        ])
        project_root = resource.parent if not is_frozen() else Path.cwd()
        candidates.extend([
            project_root / "edge" / "videos" / filename,
            project_root / "tools" / "virtual-camera" / "videos" / filename,
        ])

    for cand in candidates:
        try:
            if cand.is_file():
                return cand.resolve()
        except Exception:
            pass

    return p


class VideoFileAdapter(BaseCameraAdapter):
    """Adapter for native video file playback.

    Paces frame delivery at authentic camera FPS (1.0x speed) and seamlessly
    loops back to the start when reaching EOF.
    """

    def __init__(self, file_path: str | Path):
        self.raw_path = file_path
        self.file_path: Path | None = None
        self.error: Optional[str] = None
        self._cap: cv2.VideoCapture | None = None
        self._pending_first: np.ndarray | None = None
        self.fps: float = 25.0
        self.frame_interval: float = 1.0 / 25.0
        self._last_frame_time: float = 0.0

    def connect(self) -> bool:
        self.release()
        resolved = resolve_video_path(self.raw_path)
        if not resolved.is_file():
            self.error = f"Video file not found: {self.raw_path}"
            return False

        self.file_path = resolved
        cap = cv2.VideoCapture(str(resolved))
        if not cap.isOpened():
            self.error = f"Could not open video file: {resolved}"
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps) or fps > 240:
            fps = 25.0
        self.fps = float(fps)
        self.frame_interval = 1.0 / self.fps

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            self.error = f"Video file opened but contains no readable frames: {resolved}"
            return False

        self._cap = cap
        self._pending_first = frame
        self._last_frame_time = 0.0
        self.error = None
        return True

    def read_frame(self) -> Optional[FramePacket]:
        if self._pending_first is not None:
            frame = self._pending_first
            self._pending_first = None
            self._last_frame_time = time.time()
            return packet_from_bgr(frame)

        if self._cap is None or not self._cap.isOpened():
            return None

        # Real-time speed regulation (1.0x camera speed)
        if self._last_frame_time > 0.0:
            elapsed = time.time() - self._last_frame_time
            remaining = self.frame_interval - elapsed
            if remaining > 0.001:
                time.sleep(remaining)

        ok, frame = self._cap.read()
        if not ok or frame is None:
            # Reached end of video - seamlessly loop back to start
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok or frame is None:
                # Fallback in case seek is unsupported by backend/container
                if self.file_path:
                    self._cap.open(str(self.file_path))
                    ok, frame = self._cap.read()
            if not ok or frame is None:
                self.error = "End of video reached and could not loop back."
                return None

        self._last_frame_time = time.time()
        return packet_from_bgr(frame)

    def release(self) -> None:
        self._pending_first = None
        cap = self._cap
        self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def is_connected(self) -> bool:
        return self._cap is not None and bool(self._cap.isOpened())

    def __repr__(self) -> str:
        return f"VideoFileAdapter(file_path={self.file_path or self.raw_path})"
