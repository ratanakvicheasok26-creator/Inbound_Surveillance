"""Asynchronous latest-frame grabber.

Camera I/O runs on an isolated thread. The AI / HTTP loops only ever see the
newest FramePacket — if the camera is faster than YOLO, older frames are
overwritten immediately instead of queuing up lag.
"""

from __future__ import annotations

import threading
import time
from queue import Full, Queue
from typing import Any, Optional

from adapters.base import BaseCameraAdapter, FramePacket, safe_release

import cv2


def _orient_bgr(frame, rotate_deg, flip):
    rot = int(rotate_deg or 0)
    if rot == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rot == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    flp = str(flip or "none").strip().lower()
    if flp in ("both", "hv", "vh", "all"):
        frame = cv2.flip(frame, -1)
    elif flp in ("h", "horizontal"):
        frame = cv2.flip(frame, 1)
    elif flp in ("v", "vertical"):
        frame = cv2.flip(frame, 0)
    return frame


class _AdapterReleaseQueue:
    """Serializes adapter.release() so camera switches do not spawn unbounded threads."""

    def __init__(self) -> None:
        self._q: Queue = Queue(maxsize=32)
        self._thread = threading.Thread(target=self._loop, name="AdapterRelease", daemon=True)
        self._thread.start()

    def submit(self, adapter: BaseCameraAdapter | None) -> None:
        if adapter is None:
            return
        try:
            self._q.put_nowait(adapter)
        except Full:
            threading.Thread(
                target=safe_release,
                args=(adapter,),
                name="AdapterReleaseOverflow",
                daemon=True,
            ).start()

    def _loop(self) -> None:
        while True:
            adapter = self._q.get()
            try:
                safe_release(adapter)
            except Exception:
                pass


_RELEASE_QUEUE = _AdapterReleaseQueue()


class AsyncFrameGrabber:
    """Background reader with a thread-safe single-slot latest-frame buffer."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._latest: FramePacket | None = None
        self._adapter: BaseCameraAdapter | None = None
        self._pending: BaseCameraAdapter | None = None
        self._thread: threading.Thread | None = None
        self.generation = 0
        self.connection_state = "STANDBY"
        self.error: str | None = None
        self.ingest_fps = 0.0
        self._ingest_count = 0
        self._ingest_t0 = time.time()
        self.output_rotate = 0
        self.output_flip = "none"
        self._last_jpeg: bytes | None = None
        self._last_jpeg_t = 0.0

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop,
                name="AsyncFrameGrabber",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._ready.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        self._thread = None
        with self._lock:
            adapter = self._adapter
            pending = self._pending
            self._adapter = None
            self._pending = None
            self._latest = None
            self.connection_state = "STANDBY"
            self.error = None
            self.ingest_fps = 0.0
            self._ingest_count = 0
            self._ingest_t0 = time.time()
        safe_release(adapter)
        safe_release(pending)

    def mark_connecting(self) -> None:
        """Flip telemetry to CONNECTING without opening a device (onboard worker)."""
        with self._lock:
            self.connection_state = "CONNECTING"
            self.error = None
            self._latest = None
            self._ready.clear()

    def switch_source(self, adapter: BaseCameraAdapter) -> None:
        """Queue a new adapter. Returns immediately — teardown and connect
        happen on the grabber thread so the HTTP server never waits on
        DirectShow / V4L2 / FFmpeg.
        """
        with self._lock:
            old_pending = self._pending
            self._pending = adapter
            self.generation += 1
            self._latest = None
            self.connection_state = "CONNECTING"
            self.error = None
            self._ready.clear()
            self._last_jpeg = None
            self._last_jpeg_t = 0.0
        _RELEASE_QUEUE.submit(old_pending)

    def clear_source(self) -> None:
        """Drop the active camera without opening another."""
        with self._lock:
            old_pending = self._pending
            self._pending = _IdleAdapter()
            self.generation += 1
            self._latest = None
            self.connection_state = "STANDBY"
            self.error = None
            self._ready.clear()
        _RELEASE_QUEUE.submit(old_pending)

    def peek_latest_frame(self) -> Optional[FramePacket]:
        """Return the newest packet without consuming it.

        Grid JPEG workers must not call ``get_latest_frame`` — that clears
        ``_ready`` and starves the AI loop (or vice versa), freezing tiles.
        """
        if self._stop.is_set():
            return None
        with self._lock:
            return self._latest

    def get_latest_frame(self, timeout: float = 0.1) -> Optional[FramePacket]:
        """Wait up to ``timeout`` seconds for a frame newer than the last consume.

        The slot holds at most one packet. Frames produced while the caller is
        busy (e.g. YOLO at 8 FPS vs camera at 30 FPS) overwrite the slot, so
        lag cannot accumulate.
        """
        if self._stop.is_set():
            return None
        if not self._ready.wait(timeout=timeout):
            return None
        if self._stop.is_set():
            return None
        with self._lock:
            packet = self._latest
            self._ready.clear()
            return packet

    def _set_state(self, state: str, error: str | None = None) -> None:
        with self._lock:
            self.connection_state = state
            self.error = error

    def _take_pending(self) -> tuple[BaseCameraAdapter | None, int]:
        with self._lock:
            pending = self._pending
            generation = self.generation
            if pending is None:
                return None, generation
            self._pending = None
            return pending, generation

    def _teardown_active(self) -> None:
        with self._lock:
            adapter = self._adapter
            self._adapter = None
            self._latest = None
            self._ready.clear()
        safe_release(adapter)

    def _loop(self) -> None:
        active_generation = 0
        consecutive_nulls = 0
        null_start_ts = 0.0
        while not self._stop.is_set():
            pending, generation = self._take_pending()
            if pending is not None:
                self._teardown_active()
                if isinstance(pending, _IdleAdapter):
                    self._set_state("STANDBY", None)
                    active_generation = generation
                    continue
                print(f"[AsyncFrameGrabber] Connecting {pending}")
                self._set_state("CONNECTING", None)
                ok = False
                err: str | None = None
                try:
                    ok = bool(pending.connect())
                    if not ok:
                        err = pending.error or "Could not open camera."
                except Exception as exc:
                    ok = False
                    err = str(exc)
                    pending.error = err
                if self.generation != generation:
                    safe_release(pending)
                    continue
                if not ok:
                    safe_release(pending)
                    self._set_state("FAILED", err)
                    print(f"[AsyncFrameGrabber] FAILED: {err}")
                    active_generation = generation
                    continue
                with self._lock:
                    self._adapter = pending
                active_generation = generation

            if self._stop.is_set():
                break

            with self._lock:
                adapter = self._adapter
                has_pending = self._pending is not None

            if has_pending:
                continue
            if adapter is None:
                self._stop.wait(0.05)
                continue

            try:
                packet = adapter.read_frame()
            except Exception:
                packet = None

            if self._stop.is_set() or self.generation != active_generation:
                continue

            if packet is None:
                now = time.time()
                if consecutive_nulls == 0:
                    null_start_ts = now
                consecutive_nulls += 1
                if self.connection_state != "FAILED":
                    self._set_state("RECONNECTING", getattr(adapter, "error", None))
                if (now - null_start_ts) >= 2.5 and consecutive_nulls >= 20:
                    null_start_ts = now
                    try:
                        if hasattr(adapter, "connect") and adapter.connect():
                            self._set_state("CONNECTING", None)
                        else:
                            self._set_state("FAILED", getattr(adapter, "error", "Stream disconnected"))
                    except Exception:
                        self._set_state("FAILED", getattr(adapter, "error", "Stream disconnected"))
                self._stop.wait(0.05)
                continue
            consecutive_nulls = 0
            null_start_ts = 0.0

            # Copy so YOLO can hold this array while the next grab overwrites the slot.
            try:
                frame = packet.frame.copy()
            except Exception:
                frame = packet.frame
            jpeg = None
            now_enc = time.time()
            if now_enc - self._last_jpeg_t >= 0.09:
                try:
                    encoded_frame = frame
                    rot = getattr(self, "output_rotate", 0) or 0
                    flp = getattr(self, "output_flip", "none") or "none"
                    if rot or str(flp) not in ("none", "", "0"):
                        encoded_frame = _orient_bgr(frame, rot, flp)
                    ok, buf = cv2.imencode(".jpg", encoded_frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
                    if ok:
                        jpeg = buf.tobytes()
                        self._last_jpeg = jpeg
                        self._last_jpeg_t = now_enc
                except Exception:
                    jpeg = self._last_jpeg
            else:
                jpeg = self._last_jpeg
            published = FramePacket(frame, packet.timestamp, packet.width, packet.height, jpeg=jpeg)
            with self._lock:
                if self.generation != active_generation:
                    continue
                self._latest = published
                if self.connection_state != "CONNECTED":
                    self.connection_state = "CONNECTED"
                    self.error = None
                self._ingest_count += 1
                now = time.time()
                elapsed = now - self._ingest_t0
                if elapsed >= 1.0:
                    self.ingest_fps = self._ingest_count / elapsed
                    self._ingest_count = 0
                    self._ingest_t0 = now
            self._ready.set()


class _IdleAdapter(BaseCameraAdapter):
    """Sentinel queued by ``clear_source`` so the grabber drops the live device."""

    error = None

    def connect(self) -> bool:
        return False

    def read_frame(self) -> Optional[FramePacket]:
        return None

    def release(self) -> None:
        return None

    def is_connected(self) -> bool:
        return False


def request_still(
    source: Any,
    *,
    gateway: Any = None,
    stream_id: str | None = None,
    timeout: float = 1.5,
    client: Any = None,
) -> Any:
    """On-demand evidence JPEG. Never a second continuous grabber.

    Tries go2rtc ``/api/frame.jpeg`` first, then a one-shot OpenCV grab on a
    worker thread so the caller can time out.
    """
    import cv2
    import numpy as np

    api = client
    if api is None and gateway is not None:
        api = getattr(gateway, "client", None)
    if api is not None and stream_id and hasattr(api, "snapshot_jpeg"):
        try:
            data = api.snapshot_jpeg(stream_id, timeout=timeout)
        except Exception:
            data = None
        if data:
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame

    result: list[Any] = [None]

    def _grab() -> None:
        from adapters import create_adapter, create_direct_adapter

        adapter = None
        try:
            if gateway is not None and stream_id:
                adapter = create_adapter(source, gateway=gateway, stream_id=stream_id)
            else:
                adapter = create_direct_adapter(source)
            if adapter.connect():
                packet = adapter.read_frame()
                if packet is not None:
                    result[0] = packet.frame
        except Exception:
            pass
        finally:
            safe_release(adapter)

    thread = threading.Thread(target=_grab, name="RequestStill", daemon=True)
    thread.start()
    thread.join(timeout=max(0.2, float(timeout) + 2.0))
    return result[0]
