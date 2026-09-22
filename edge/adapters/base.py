"""Camera adapter contract and shared capture helpers."""

from __future__ import annotations

import re
import sys
import time
import urllib.parse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# Open/read timeouts stay short so a dead URL cannot stall the grabber
# (FFmpeg's default RTSP timeout is ~20s).
CAPTURE_TIMEOUT_MS = 2000
RTSP_STIMEOUT_US = 2_000_000
V4L_RELEASE_PAUSE = 0.05
_VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v")
_UNIX_ROOTS = {"home", "users", "tmp", "var", "opt", "media", "mnt", "data"}


class FramePacket:
    def __init__(
        self,
        frame: np.ndarray,
        timestamp: float,
        width: int,
        height: int,
        jpeg: bytes | None = None,
    ):
        self.frame = frame
        self.timestamp = timestamp
        self.width = width
        self.height = height
        self.jpeg = jpeg


class BaseCameraAdapter(ABC):
    """Open, read, and release a single video source.

    ``connect`` / ``read_frame`` / ``release`` run on the grabber thread only.
    Construction must be cheap and free of I/O so the HTTP thread can
    ``switch_source`` without blocking.
    """

    error: Optional[str] = None

    @abstractmethod
    def connect(self) -> bool:
        """Open camera connection with strict non-blocking timeouts."""
        pass

    @abstractmethod
    def read_frame(self) -> Optional[FramePacket]:
        """Read raw frame from source without blocking AI loop."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Cleanly close and free hardware/network handles."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass


def redact_source(value: Any) -> str:
    return re.sub(r"(://[^:/?#]+):([^@]+)@", r"\1:****@", str(value))


def parse_source(value: Any) -> int | str:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


def webcam_index(value: Any) -> int:
    """Coerce a webcam source to a device index.

    Saved configs have produced values like ``0vi`` that are not pure digits,
    so ``int(source)`` raises and the laptop camera never opens.
    """
    parsed = parse_source(value)
    if isinstance(parsed, int):
        return parsed
    text = str(parsed or "").strip()
    if text.isdigit():
        return int(text)
    match = re.search(r"(?:video\s*)?(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def source_has_video_ext(text: str) -> bool:
    path = str(text or "").split("?", 1)[0].rstrip("/").lower()
    return path.endswith(_VIDEO_EXTS)


def decode_file_uri(raw: str) -> str:
    """Turn ``file://`` URIs into native filesystem paths.

    Windows ``file:///C:/Users/clip.mp4`` must become ``C:\\Users\\clip.mp4``,
    not ``/C:/Users/clip.mp4``. A naive ``text[7:]`` strip does the latter and
    OpenCV cannot open the file.
    """
    text = str(raw or "").strip()
    if not text.lower().startswith("file:"):
        return text
    normalized = text.replace("\\", "/")
    # file://C:/Users/... is a common malformed form; force three slashes.
    if re.match(r"file://[A-Za-z]:", normalized, re.IGNORECASE):
        normalized = "file:///" + normalized[7:]
    try:
        parsed = urllib.parse.urlparse(normalized)
    except Exception:
        return text[5:].lstrip("/") if text.lower().startswith("file:") else text
    path = urllib.parse.unquote(parsed.path or "")
    netloc = parsed.netloc or ""
    if sys.platform == "win32":
        from urllib.request import url2pathname

        if netloc and re.fullmatch(r"[A-Za-z]", netloc):
            path = f"/{netloc}:{path}"
        elif netloc and netloc.lower() not in ("localhost", "127.0.0.1"):
            unc = "\\\\" + netloc + path.replace("/", "\\")
            return unc
        converted = url2pathname(path)
        return converted or text
    if netloc and netloc.lower() not in ("localhost", "127.0.0.1", ""):
        return "/" + netloc + path
    return path or text


def unwrap_local_video_source(source: Any) -> str | None:
    """Return a filesystem path when ``source`` is a local video file.

    The hub used to wrap paths like ``/home/user/clip.mp4`` into
    ``rtsp://user@home/user/clip.mp4``. That made go2rtc DNS-lookup ``home``
    and OpenCV/FFmpeg treat a file as RTSP, which hangs or crashes ingest.
    """
    text = str(source or "").strip()
    if not text:
        return None
    if text.lower().startswith("file:"):
        text = decode_file_uri(text)
    lower = text.lower()
    networked = lower.startswith(
        ("rtsp://", "http://", "https://", "tapo://", "onvif://", "whep://", "webrtc://", "whip://")
    )
    if source_has_video_ext(text) and not networked:
        return text
    if not lower.startswith(("rtsp://", "http://", "https://")) or not source_has_video_ext(text):
        return None
    try:
        parsed = urllib.parse.urlparse(text)
    except Exception:
        return None
    host = parsed.hostname or ""
    path = urllib.parse.unquote(parsed.path or "")
    if not host or not path:
        return None
    if len(host) == 1 and host.isalpha():
        win_path = path.replace("/", "\\") if sys.platform == "win32" else path
        drive = f"{host.upper()}:{win_path}"
        if source_has_video_ext(drive):
            return drive
    candidate = "/" + host + path
    if not source_has_video_ext(candidate):
        return None
    if host.lower() in _UNIX_ROOTS:
        return candidate
    try:
        if Path(candidate).is_file():
            return candidate
        if Path(path).is_file():
            return path
    except Exception:
        pass
    return None


def protocol_from_source(source: Any) -> str:
    if isinstance(source, int) or str(source or "").strip().isdigit():
        return "webcam"
    raw = str(source or "").strip()
    text = raw.lower()
    if unwrap_local_video_source(raw) is not None or (
        source_has_video_ext(text)
        and not text.startswith(("rtsp://", "http://", "https://", "tapo://", "onvif://"))
    ):
        return "video"
    if text.startswith("onvif://") or (
        (text.startswith("http://") or text.startswith("https://")) and "/onvif" in text
    ):
        return "onvif"
    if text.startswith("tapo://"):
        return "tapo"
    if text.startswith("webrtc://") or text.startswith("whep://") or text.startswith("whip://"):
        return "webrtc"
    if text.startswith("http://") or text.startswith("https://"):
        return "phone"
    return "rtsp"


def set_capture_timeouts(cap: cv2.VideoCapture) -> None:
    open_to = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
    read_to = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
    if open_to is not None:
        cap.set(open_to, CAPTURE_TIMEOUT_MS)
    if read_to is not None:
        cap.set(read_to, CAPTURE_TIMEOUT_MS)


def open_capture(source: Any, backend: int | None = None) -> cv2.VideoCapture:
    cap = cv2.VideoCapture()
    set_capture_timeouts(cap)
    if backend is None:
        cap.open(source)
    else:
        cap.open(source, backend)
    return cap


def read_first_frame(cap: cv2.VideoCapture, tries: int = 8, pause: float = 0.25):
    for _ in range(tries):
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        time.sleep(pause)
    return None


def packet_from_bgr(frame: np.ndarray) -> FramePacket:
    h, w = frame.shape[:2]
    return FramePacket(frame, time.time(), int(w), int(h))


def safe_release(adapter: BaseCameraAdapter | None) -> None:
    if adapter is None:
        return
    try:
        adapter.release()
    except Exception:
        pass


def webcam_backends() -> list[int]:
    """Native backend only for the requested index — no MSMF/index probing."""
    backends: list[int] = []
    if sys.platform.startswith("linux"):
        if hasattr(cv2, "CAP_V4L2"):
            backends.append(cv2.CAP_V4L2)
    elif sys.platform == "win32":
        dshow = getattr(cv2, "CAP_DSHOW", None)
        if dshow is not None:
            backends.append(dshow)
            return backends
    elif sys.platform == "darwin":
        avf = getattr(cv2, "CAP_AVFOUNDATION", None)
        if avf is not None:
            backends.append(avf)
    backends.append(cv2.CAP_ANY)
    return backends
