"""HTTP Digest/Basic MJPEG adapter for phone apps such as IP Webcam."""

from __future__ import annotations

import time
import urllib.parse
from typing import Any, Optional

import cv2
import numpy as np
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

from adapters.base import (
    BaseCameraAdapter,
    FramePacket,
    packet_from_bgr,
    redact_source,
)


def _split_http_auth(url: str) -> tuple[str, str, str]:
    parsed = urllib.parse.urlparse(url)
    user = urllib.parse.unquote(parsed.username or "")
    password = urllib.parse.unquote(parsed.password or "")
    host = parsed.hostname or ""
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    clean = urllib.parse.urlunparse(
        (parsed.scheme, netloc, parsed.path or "/", parsed.params, parsed.query, parsed.fragment)
    )
    return clean, user, password


def _http_stream_candidates(url: str) -> list[str]:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path or "/"
    if path not in ("", "/"):
        return [url]
    out: list[str] = []
    for extra in ("/video", "/mjpegfeed", "/videofeed", "/shot.jpg"):
        out.append(
            urllib.parse.urlunparse((parsed.scheme, parsed.netloc, extra, "", parsed.query, ""))
        )
    return out


def _decode_jpeg(data: bytes):
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


class HttpMjpegCapture:
    """MJPEG/JPEG HTTP capture with Basic and Digest login.

    IP Webcam (and similar phone apps) often require Digest auth. OpenCV/FFmpeg
    only speaks Basic, so a browser can open the same URL while VideoCapture fails.
    """

    def __init__(self, url: str, username: str = "", password: str = ""):
        self.url = url
        self.username = username
        self.password = password
        self.error: str | None = None
        self._session = requests.Session()
        self._session.trust_env = False
        self._session.headers.update(
            {
                "User-Agent": "InboundSurveillance/1.0",
                "Accept": "multipart/x-mixed-replace,image/jpeg,*/*",
            }
        )
        self._resp: requests.Response | None = None
        self._buf = bytearray()
        self._mode = "mjpeg"
        self._opened = False
        self._last_connect_try = time.time()
        err = self._connect()
        if err:
            self.error = err
            self._opened = False
        else:
            self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def set(self, *_args: Any, **_kwargs: Any) -> bool:
        return False

    def release(self) -> None:
        self._opened = False
        self._close_body()
        try:
            self._session.close()
        except Exception:
            pass

    def read(self) -> tuple[bool, Any]:
        now = time.time()
        if not self._opened:
            if (now - getattr(self, "_last_connect_try", 0.0)) >= 2.0:
                self._last_connect_try = now
                if self._connect() is None:
                    self._opened = True
                else:
                    return False, None
            else:
                return False, None
        try:
            frame = self._next_frame()
            if frame is not None:
                return True, frame
        except Exception:
            self._close_body()
            self._opened = False
            self._last_connect_try = time.time()
            if self._connect() is None:
                self._opened = True
                try:
                    frame = self._next_frame()
                    if frame is not None:
                        return True, frame
                except Exception:
                    return False, None
        return False, None

    def _auth_attempts(self) -> list[Any]:
        if not self.username:
            return [None]
        return [
            HTTPDigestAuth(self.username, self.password),
            HTTPBasicAuth(self.username, self.password),
        ]

    def _close_body(self) -> None:
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:
                pass
            self._resp = None
        self._buf = bytearray()

    def _connect(self) -> str | None:
        self._close_body()
        last = f"Could not connect to camera '{redact_source(self.url)}'."
        # (connect, read) seconds — keep well under FFmpeg's 20s default hang.
        timeout = (2.0, 2.0)
        for auth in self._auth_attempts():
            try:
                resp = self._session.get(
                    self.url,
                    auth=auth,
                    stream=True,
                    timeout=timeout,
                    allow_redirects=True,
                )
            except requests.RequestException as exc:
                last = f"Could not reach '{redact_source(self.url)}': {exc}"
                continue
            if resp.status_code == 401:
                resp.close()
                last = (
                    f"Login failed for '{redact_source(self.url)}'. "
                    "This phone stream uses HTTP Digest authentication — a browser can "
                    "open it, but a plain RTSP/OpenCV connection cannot. Recheck the "
                    "username and password from IP Webcam."
                )
                continue
            if resp.status_code >= 400:
                last = f"HTTP {resp.status_code} from '{redact_source(self.url)}'."
                resp.close()
                continue
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "text/html" in ctype:
                resp.close()
                last = (
                    f"'{redact_source(self.url)}' is the phone's web page, not a video "
                    "stream. Use http://PHONE_IP:8080/video"
                )
                continue
            self._resp = resp
            if "image/jpeg" in ctype and "multipart" not in ctype:
                self._mode = "snapshot"
            else:
                self._mode = "mjpeg"
            return None
        return last

    def _next_frame(self):
        if self._mode == "snapshot":
            return self._read_snapshot()
        return self._read_mjpeg()

    def _read_snapshot(self):
        if self._resp is not None:
            data = self._resp.content
            self._close_body()
            frame = _decode_jpeg(data)
            if frame is not None:
                return frame
        timeout = (2.0, 2.0)
        last_exc: Exception | None = None
        for auth in self._auth_attempts():
            try:
                resp = self._session.get(self.url, auth=auth, timeout=timeout)
            except requests.RequestException as exc:
                last_exc = exc
                continue
            if resp.status_code == 200:
                frame = _decode_jpeg(resp.content)
                if frame is not None:
                    return frame
            last_exc = RuntimeError(f"HTTP {resp.status_code}")
        if last_exc:
            raise last_exc
        return None

    def _read_mjpeg(self):
        if self._resp is None:
            err = self._connect()
            if err:
                raise RuntimeError(err)
        assert self._resp is not None
        while True:
            start = self._buf.find(b"\xff\xd8")
            end = self._buf.find(b"\xff\xd9", start + 2) if start >= 0 else -1
            if start >= 0 and end >= 0:
                jpeg = bytes(self._buf[start : end + 2])
                del self._buf[: end + 2]
                frame = _decode_jpeg(jpeg)
                if frame is not None:
                    return frame
                continue
            if start > 0:
                del self._buf[:start]
            if len(self._buf) > 4_000_000:
                self._buf = bytearray()
            chunk = next(self._resp.iter_content(chunk_size=16_384), b"")
            if not chunk:
                raise RuntimeError("HTTP MJPEG stream ended")
            self._buf.extend(chunk)


class PhoneHttpAdapter(BaseCameraAdapter):
    """Phone MJPEG/JPEG HTTP (IP Webcam). Connect is deferred out of the constructor."""

    def __init__(self, url: str):
        self.url = str(url)
        self.error: Optional[str] = None
        self._cap: HttpMjpegCapture | None = None
        self._pending_first = None

    def connect(self) -> bool:
        self.release()
        clean, user, password = _split_http_auth(self.url)
        last_err: str | None = None
        for candidate in _http_stream_candidates(clean):
            cap = HttpMjpegCapture(candidate, user, password)
            if not cap.isOpened():
                last_err = cap.error
                cap.release()
                continue
            ok, frame = cap.read()
            if ok and frame is not None:
                self._cap = cap
                self._pending_first = frame
                self.error = None
                return True
            last_err = (
                f"Logged in to '{redact_source(candidate)}' but received no JPEG frames."
            )
            cap.release()
        hint = (
            " For IP Webcam, Protocol must be Phone HTTP, URL "
            "http://PHONE_IP:8080/video, and username/password filled if the app requires them."
        )
        self.error = (last_err or f"Could not connect to camera '{redact_source(self.url)}'.") + hint
        return False

    def read_frame(self) -> Optional[FramePacket]:
        if self._pending_first is not None:
            frame = self._pending_first
            self._pending_first = None
            return packet_from_bgr(frame)
        if self._cap is None or not self._cap.isOpened():
            now = time.time()
            if (now - getattr(self, "_last_adapter_reconnect", 0.0)) >= 2.5:
                self._last_adapter_reconnect = now
                self.connect()
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

    def is_connected(self) -> bool:
        return self._cap is not None and bool(self._cap.isOpened())

    def __repr__(self) -> str:
        return f"PhoneHttpAdapter(url={redact_source(self.url)!r})"
