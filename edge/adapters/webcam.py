"""Local webcam adapter: DirectShow (Windows) / V4L2 (Linux) / AVFoundation (macOS)."""

from __future__ import annotations

import time
from typing import Optional

import cv2

from adapters.base import (
    BaseCameraAdapter,
    FramePacket,
    V4L_RELEASE_PAUSE,
    open_capture,
    packet_from_bgr,
    read_first_frame,
    webcam_backends,
)


def open_webcam_index(idx: int) -> cv2.VideoCapture | None:
    seen: set[int] = set()
    for backend in webcam_backends():
        if backend in seen:
            continue
        seen.add(backend)
        cap = open_capture(idx, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
    return None


class WebcamAdapter(BaseCameraAdapter):
    """Single-index webcam. Never probes neighboring devices."""

    def __init__(self, index: int):
        self.index = int(index)
        self.error: Optional[str] = None
        self._cap: cv2.VideoCapture | None = None
        self._pending_first = None

    def connect(self) -> bool:
        self.release()
        cap = open_webcam_index(self.index)
        if cap is None:
            self.error = f"Could not open camera index {self.index}."
            return False
        frame = read_first_frame(cap, tries=4, pause=0.1)
        if frame is None:
            cap.release()
            time.sleep(V4L_RELEASE_PAUSE)
            self.error = (
                f"Camera index {self.index} opened but produced no frames "
                "(busy device or index out of range)."
            )
            return False
        self._cap = cap
        self._pending_first = frame
        self.error = None
        return True

    def read_frame(self) -> Optional[FramePacket]:
        if self._pending_first is not None:
            frame = self._pending_first
            self._pending_first = None
            return packet_from_bgr(frame)
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
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
            time.sleep(V4L_RELEASE_PAUSE)

    def is_connected(self) -> bool:
        return self._cap is not None and bool(self._cap.isOpened())

    def __repr__(self) -> str:
        return f"WebcamAdapter(index={self.index})"
