"""RTSP adapter with TCP then UDP and a hard 2s FFmpeg socket timeout."""

from __future__ import annotations

import os
import socket
import threading
import time
import urllib.parse
from typing import Optional

import cv2

from adapters.base import (
    BaseCameraAdapter,
    FramePacket,
    RTSP_STIMEOUT_US,
    V4L_RELEASE_PAUSE,
    open_capture,
    packet_from_bgr,
    read_first_frame,
    redact_source,
)

_ENV_LOCK = threading.Lock()

# stimeout is microseconds. 2000000 = 2s — FFmpeg's default is ~20s.
_RTSP_FFMPEG_OPTS = (
    f"rtsp_transport;tcp|stimeout;{RTSP_STIMEOUT_US}|max_delay;500000|fflags;nobuffer",
    f"rtsp_transport;udp|stimeout;{RTSP_STIMEOUT_US}|max_delay;500000|fflags;nobuffer",
)


def _probe_rtsp_host(url: str, timeout: float = 2.0) -> str | None:
    """Fail fast when the NVR/phone is offline — FFmpeg can ignore stimeout
    during TCP SYN to an unreachable address and hang for tens of seconds.
    """
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    port = parsed.port or 554
    if not host:
        return f"Invalid RTSP URL '{redact_source(url)}'."
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return None
    except OSError as exc:
        return f"Could not reach '{redact_source(url)}': {exc}"


class RTSPAdapter(BaseCameraAdapter):
    """Open an RTSP URL. Tries TCP (NVRs/firewalls) then UDP (some phone apps)."""

    def __init__(self, url: str):
        self.url = str(url)
        self.error: Optional[str] = None
        self._cap: cv2.VideoCapture | None = None
        self._pending_first = None

    def connect(self) -> bool:
        self.release()
        probe_err = _probe_rtsp_host(self.url)
        if probe_err:
            self.error = probe_err
            return False
        ffmpeg = getattr(cv2, "CAP_FFMPEG", None)
        attempts: list[tuple[str, int | None]] = [(opts, ffmpeg) for opts in _RTSP_FFMPEG_OPTS]
        attempts.append(("", None))

        last_err: str | None = None
        seen: set[tuple[str, int | None]] = set()
        for opts, backend in attempts:
            key = (opts, backend)
            if key in seen:
                continue
            seen.add(key)
            with _ENV_LOCK:
                prev_opts = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
                try:
                    if opts:
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
                    elif prev_opts is not None:
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = prev_opts
                    cap = open_capture(self.url, backend)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                finally:
                    if prev_opts is not None:
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = prev_opts
                    else:
                        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            if not cap.isOpened():
                cap.release()
                last_err = f"Could not connect to camera '{redact_source(self.url)}'."
                continue
            frame = read_first_frame(cap, tries=4, pause=0.15)
            if frame is not None:
                self._cap = cap
                self._pending_first = frame
                self.error = None
                return True
            cap.release()
            time.sleep(V4L_RELEASE_PAUSE)
            last_err = (
                f"Connected to '{redact_source(self.url)}', but received no video frame. "
                "The stream may be offline, already in use, or using a transport "
                "this PC cannot read."
            )

        hint = (
            " Generic RTSP on port 554 is for CCTV/NVRs, not phones. "
            "For IP Webcam, use rtsp://PHONE_IP:8080/h264_ulaw.sdp with empty "
            "username and password, or Protocol 'Phone HTTP' at "
            "http://PHONE_IP:8080/video. Both devices must be on the same Wi-Fi."
        )
        self.error = (last_err or f"Could not connect to camera '{redact_source(self.url)}'.") + hint
        return False

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
        return f"RTSPAdapter(url={redact_source(self.url)!r})"
