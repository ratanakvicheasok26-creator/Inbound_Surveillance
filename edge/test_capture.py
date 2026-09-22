"""Phase 1 capture tests: latest-frame drop and non-blocking source switch."""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import create_adapter
from adapters.base import BaseCameraAdapter, FramePacket, protocol_from_source
from adapters.phone_http import PhoneHttpAdapter
from adapters.rtsp import RTSPAdapter
from adapters.webcam import WebcamAdapter
from capture import AsyncFrameGrabber, request_still


class FakeAdapter(BaseCameraAdapter):
    def __init__(self, fps: float = 30.0, name: str = "fake"):
        self.fps = fps
        self.name = name
        self.error = None
        self.connect_calls = 0
        self.release_calls = 0
        self.frames_produced = 0
        self._connected = False
        self._lock = threading.Lock()
        self.released = threading.Event()

    def connect(self) -> bool:
        self.connect_calls += 1
        self._connected = True
        return True

    def read_frame(self) -> FramePacket | None:
        if not self._connected:
            return None
        time.sleep(1.0 / self.fps)
        with self._lock:
            self.frames_produced += 1
            n = self.frames_produced
        frame = np.full((8, 8, 3), n % 255, dtype=np.uint8)
        return FramePacket(frame, time.time(), 8, 8)

    def release(self) -> None:
        self._connected = False
        self.release_calls += 1
        self.released.set()

    def is_connected(self) -> bool:
        return self._connected


class SlowConnectAdapter(FakeAdapter):
    def __init__(self, delay: float = 2.0, name: str = "slow"):
        super().__init__(name=name)
        self.delay = delay

    def connect(self) -> bool:
        time.sleep(self.delay)
        return super().connect()


def test_create_adapter_routing() -> None:
    assert isinstance(create_adapter(0), WebcamAdapter)
    assert isinstance(create_adapter("2"), WebcamAdapter)
    assert isinstance(create_adapter("rtsp://cam/stream"), RTSPAdapter)
    assert isinstance(create_adapter("http://192.168.1.8:8080/video"), PhoneHttpAdapter)
    assert protocol_from_source(0) == "webcam"
    assert protocol_from_source("http://x/video") == "phone"
    assert protocol_from_source("rtsp://x") == "rtsp"
    assert protocol_from_source("onvif://192.168.1.1") == "onvif"
    assert protocol_from_source("http://192.168.1.1/onvif/device_service") == "onvif"
    assert protocol_from_source("tapo://u:p@host") == "tapo"
    assert protocol_from_source("whep://host/path") == "webrtc"
    assert protocol_from_source("clip.mp4") == "video"
    assert protocol_from_source("file:///tmp/clip.mkv") == "video"


def test_latest_frame_does_not_accumulate_lag() -> None:
    grabber = AsyncFrameGrabber()
    adapter = FakeAdapter(fps=30.0)
    grabber.start()
    try:
        grabber.switch_source(adapter)
        deadline = time.time() + 2.0
        first = None
        while time.time() < deadline:
            first = grabber.get_latest_frame(0.1)
            if first is not None:
                break
        assert first is not None, "grabber never produced a frame"

        ages: list[float] = []
        for _ in range(16):
            time.sleep(0.125)  # 8 FPS consumer vs 30 FPS camera
            pkt = grabber.get_latest_frame(0.2)
            assert pkt is not None
            ages.append(time.time() - pkt.timestamp)

        tail = ages[-8:]
        assert max(tail) < 0.08, f"frame lag accumulated: {tail}"
        assert adapter.frames_produced > 16, "producer should keep running ahead of consumer"
        print(f"ok latest-frame lag max={max(tail)*1000:.1f}ms produced={adapter.frames_produced}")
    finally:
        grabber.stop()
        assert adapter.release_calls >= 1


def test_peek_does_not_consume_ready_slot() -> None:
    grabber = AsyncFrameGrabber()
    adapter = FakeAdapter(fps=8.0)
    grabber.start()
    try:
        grabber.switch_source(adapter)
        deadline = time.time() + 2.0
        peeked = None
        while time.time() < deadline:
            peeked = grabber.peek_latest_frame()
            if peeked is not None:
                break
            time.sleep(0.02)
        assert peeked is not None, "peek never saw a frame"
        consumed = grabber.get_latest_frame(0.05)
        assert consumed is not None, "peek must leave the ready slot for the AI consumer"
        print("ok peek_latest_frame does not consume ready slot")
    finally:
        grabber.stop()


def test_switch_source_returns_immediately() -> None:
    grabber = AsyncFrameGrabber()
    slow = SlowConnectAdapter(delay=2.0)
    fast = FakeAdapter(name="fast")
    grabber.start()
    try:
        t0 = time.time()
        grabber.switch_source(slow)
        assert time.time() - t0 < 0.05, "switch_source blocked on connect()"
        assert grabber.connection_state == "CONNECTING"

        t1 = time.time()
        grabber.switch_source(fast)
        elapsed = time.time() - t1
        assert elapsed < 0.05, f"second switch_source blocked ({elapsed:.3f}s)"

        deadline = time.time() + 3.0
        pkt = None
        while time.time() < deadline:
            pkt = grabber.get_latest_frame(0.1)
            if pkt is not None and grabber.connection_state == "CONNECTED":
                break
        assert pkt is not None
        assert grabber.connection_state == "CONNECTED"
        assert slow.released.wait(timeout=3.0), "slow adapter was not torn down"
        print(f"ok non-blocking switch ({elapsed*1000:.1f}ms)")
    finally:
        grabber.stop()


def test_http_telemetry_stays_responsive_during_connect() -> None:
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.request import urlopen

    grabber = AsyncFrameGrabber()
    grabber.start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"connection": grabber.connection_state}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            print("[test server log]", fmt % args)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        grabber.switch_source(SlowConnectAdapter(delay=1.5))
        port = server.server_port or server.socket.getsockname()[1]
        url = f"http://127.0.0.1:{port}/"
        samples: list[float] = []
        import requests
        session = requests.Session()
        session.trust_env = False
        for _ in range(10):
            t0 = time.time()
            resp = session.get(url, timeout=1.0)
            payload = resp.json()
            samples.append(time.time() - t0)
            assert payload["connection"] in ("CONNECTING", "CONNECTED")
        assert max(samples) < 0.2, f"telemetry stalled: {samples}"
        print(f"ok telemetry during connect max={max(samples)*1000:.1f}ms")
    finally:
        server.shutdown()
        grabber.stop()


def test_rtsp_unreachable_fails_fast() -> None:
    from adapters.rtsp import RTSPAdapter

    adapter = RTSPAdapter("rtsp://192.0.2.1:554/offline")
    t0 = time.time()
    ok = adapter.connect()
    elapsed = time.time() - t0
    assert not ok
    assert elapsed < 2.5, f"RTSP connect hung {elapsed:.1f}s"
    assert adapter.error and "Could not reach" in adapter.error
    print(f"ok RTSP fail-fast {elapsed:.2f}s")


def test_failed_connect_does_not_leave_adapter() -> None:
    class DeadAdapter(BaseCameraAdapter):
        error = "offline"

        def connect(self) -> bool:
            return False

        def read_frame(self) -> FramePacket | None:
            return None

        def release(self) -> None:
            self.released = True

        def is_connected(self) -> bool:
            return False

    grabber = AsyncFrameGrabber()
    dead = DeadAdapter()
    grabber.start()
    try:
        grabber.switch_source(dead)
        deadline = time.time() + 2.0
        while time.time() < deadline and grabber.connection_state == "CONNECTING":
            time.sleep(0.02)
        assert grabber.connection_state == "FAILED"
        assert grabber.error == "offline"
        assert grabber.get_latest_frame(0.05) is None
        print("ok failed connect")
    finally:
        grabber.stop()


def test_rapid_switch_ten_times_no_deadlock() -> None:
    grabber = AsyncFrameGrabber()
    adapters: list[FakeAdapter] = []
    for i in range(10):
        if i == 9:
            adapters.append(FakeAdapter(name="last"))
        elif i % 2 == 0:
            adapters.append(SlowConnectAdapter(delay=0.15, name=f"slow-{i}"))
        else:
            adapters.append(FakeAdapter(name=f"fake-{i}"))
    last = adapters[-1]
    grabber.start()
    try:
        t0 = time.time()
        for adapter in adapters:
            grabber.switch_source(adapter)
        elapsed = time.time() - t0
        assert elapsed < 1.0, f"rapid switch_source blocked ({elapsed:.3f}s)"

        deadline = time.time() + 5.0
        pkt = None
        while time.time() < deadline:
            pkt = grabber.get_latest_frame(0.1)
            if (
                pkt is not None
                and grabber.connection_state == "CONNECTED"
                and last.frames_produced > 0
            ):
                break
        assert pkt is not None, "grabber never produced a frame from last adapter"
        assert grabber.connection_state == "CONNECTED"
        assert last.frames_produced > 0, "last adapter never produced a frame"

        for adapter in adapters[:-1]:
            assert adapter.released.wait(timeout=3.0), (
                f"superseded adapter {adapter.name} was not torn down"
            )
        print(f"ok rapid switch x10 ({elapsed*1000:.1f}ms)")
    finally:
        grabber.stop()


class RapidSwitchTests(unittest.TestCase):
    def test_rapid_switch_ten_times_no_deadlock(self) -> None:
        test_rapid_switch_ten_times_no_deadlock()


def test_request_still_uses_gateway_jpeg() -> None:
    class FakeClient:
        def snapshot_jpeg(self, stream_id: str, timeout=None):
            assert stream_id == "cam-01-main"
            frame = np.zeros((16, 24, 3), dtype=np.uint8)
            import cv2

            ok, buf = cv2.imencode(".jpg", frame)
            assert ok
            return buf.tobytes()

    class FakeGateway:
        client = FakeClient()

    still = request_still(
        "rtsp://cam/main",
        gateway=FakeGateway(),
        stream_id="cam-01-main",
        timeout=0.5,
    )
    assert still is not None
    assert still.shape[0] == 16
    assert still.shape[1] == 24
    print("ok request_still jpeg")


def test_phone_http_adapter_connect_and_read() -> None:
    from unittest.mock import MagicMock, patch
    from adapters.phone_http import HttpMjpegCapture, PhoneHttpAdapter

    # Test HttpMjpegCapture creation and time reference
    with patch("requests.Session.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "image/jpeg"}
        mock_resp.iter_content = MagicMock(return_value=iter([]))
        mock_get.return_value = mock_resp

        cap = HttpMjpegCapture("http://192.168.1.50:8080/video")
        assert cap.isOpened() is True
        assert hasattr(cap, "_last_connect_try")
        assert isinstance(cap._last_connect_try, float)

        adapter = PhoneHttpAdapter("http://192.168.1.50:8080/video")
        with patch.object(adapter, "connect", return_value=True):
            adapter._cap = cap
            # Ensure read_frame runs without NameError: name 'time' is not defined
            _ = adapter.read_frame()
    print("ok phone_http adapter connect and read")


if __name__ == "__main__":
    test_create_adapter_routing()
    test_latest_frame_does_not_accumulate_lag()
    test_peek_does_not_consume_ready_slot()
    test_switch_source_returns_immediately()
    test_http_telemetry_stays_responsive_during_connect()
    test_rtsp_unreachable_fails_fast()
    test_failed_connect_does_not_leave_adapter()
    test_rapid_switch_ten_times_no_deadlock()
    test_request_still_uses_gateway_jpeg()
    test_phone_http_adapter_connect_and_read()
    print("all capture tests passed")
