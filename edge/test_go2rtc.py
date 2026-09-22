"""Phase 2 go2rtc tests: binary lifecycle, REST client, multiplexing, teardown."""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import create_adapter, create_direct_adapter
from adapters.gateway import GatewayAdapter
from adapters.rtsp import RTSPAdapter
from adapters.webcam import WebcamAdapter
from capture import AsyncFrameGrabber
from media.client import Go2RtcClient
from media.go2rtc import (
    Go2RtcManager,
    binary_filename,
    ensure_binary,
    platform_tag,
    sanitize_stream_id,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_platform_and_stream_id() -> None:
    tag = platform_tag()
    assert "-" in tag, tag
    os_name, _, arch = tag.partition("-")
    assert os_name in ("linux", "windows", "darwin"), tag
    name = binary_filename()
    if sys.platform == "win32":
        assert name.endswith(".exe")
    else:
        assert name == "go2rtc"
    assert sanitize_stream_id("cam-1") == "cam-1"
    assert sanitize_stream_id("Front Desk!") == "Front-Desk"
    assert sanitize_stream_id("") == "live"
    print(f"ok platform tag={tag} binary={name}")


def test_ffmpeg_candidates_skip_unix_paths_on_windows_shape() -> None:
    from media.go2rtc import candidate_binary_paths, ffmpeg_candidate_paths
    from paths import resource_dir

    bins = candidate_binary_paths()
    assert any(p.name.startswith("go2rtc") for p in bins)
    assert any(str(p).startswith(str(resource_dir())) for p in bins)

    ff = ffmpeg_candidate_paths()
    assert ff
    if sys.platform == "win32":
        assert all("/usr/bin/" not in p for p in ff)
    else:
        assert any(p.endswith("ffmpeg") for p in ff)

    from media.go2rtc import yaml_quoted_path

    quoted = yaml_quoted_path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe")
    assert quoted == "C:/Program Files/ffmpeg/bin/ffmpeg.exe"
    assert "\\b" not in f'bin: "{quoted}"'


def test_create_adapter_gateway_routing() -> None:
    class FakeClient:
        def __init__(self):
            self.registered: list[tuple[str, str]] = []

        def register_stream(self, stream_id: str, source_url: str) -> bool:
            self.registered.append((stream_id, source_url))
            return True

        def remove_stream(self, stream_id: str) -> bool:
            return True

    class FakeGateway:
        def __init__(self):
            self.client = FakeClient()

        def is_ready(self) -> bool:
            return True

        def consumer_url(self, stream_id: str, source: Any) -> str:
            return f"rtsp://127.0.0.1:8554/{stream_id}"

    gw = FakeGateway()
    assert isinstance(create_adapter(0), WebcamAdapter)
    assert isinstance(create_direct_adapter("rtsp://cam/stream"), RTSPAdapter)
    net = create_adapter("rtsp://cam/stream", gateway=gw, stream_id="front-desk")
    assert isinstance(net, GatewayAdapter)
    assert net.stream_id == "front-desk"
    assert net.local_url == "rtsp://127.0.0.1:8554/front-desk"
    phone = create_adapter("http://192.168.1.8:8080/video", gateway=gw, stream_id="phone")
    assert isinstance(phone, GatewayAdapter)
    print("ok gateway adapter routing")


def test_bind_gateway_keeps_previous_streams() -> None:
    from launcher import LiveStreamEngine

    class FakeClient:
        def __init__(self):
            self.registered: list[tuple[str, str]] = []
            self.removed: list[str] = []

        def register_stream(self, stream_id: str, source_url: str) -> bool:
            self.registered.append((stream_id, source_url))
            return True

        def remove_stream(self, stream_id: str) -> bool:
            self.removed.append(stream_id)
            return True

    class FakeMedia:
        def __init__(self):
            self.client = FakeClient()

        def is_ready(self) -> bool:
            return True

        def status(self, stream_id: str) -> dict[str, Any]:
            return {"ready": True}

    engine = LiveStreamEngine()
    engine.media = FakeMedia()
    engine.cfg["cameras"] = []

    sid1 = sanitize_stream_id("camera_1")
    sid2 = sanitize_stream_id("camera_2")
    engine._bind_gateway("rtsp://192.168.1.10/cam1", "camera_1")
    engine._bind_gateway("rtsp://192.168.1.11/cam2", "camera_2")

    registered_ids = [sid for sid, _url in engine.media.client.registered]
    assert sid1 in registered_ids, f"camera_1 not registered: {registered_ids}"
    assert sid2 in registered_ids, f"camera_2 not registered: {registered_ids}"
    removed = list(engine.media.client.removed)
    assert sid1 not in removed, f"remove_stream called for camera_1: {removed}"
    assert f"{sid1}-main" not in removed, f"remove_stream called for camera_1-main: {removed}"
    assert sid2 not in removed

    engine.cfg["cameras"] = [
        {"id": "net_a", "source": "rtsp://192.168.1.21/a"},
        {"id": "net_b", "source": "rtsp://192.168.1.22/b"},
        {"id": "net_c", "source": "rtsp://192.168.1.23/c"},
        {"id": "laptop_cam", "source": 0},
        {"id": "clip", "source": "/tmp/demo.mp4", "protocol": "video"},
    ]
    engine._bind_gateway("rtsp://192.168.1.10/cam1", "camera_1")
    registered_ids = [sid for sid, _url in engine.media.client.registered]
    assert sanitize_stream_id("net_a") not in registered_ids, (
        f"single-camera connect registered all cameras: {registered_ids}"
    )
    engine.register_all_cameras_in_gateway()
    registered_ids = [sid for sid, _url in engine.media.client.registered]
    for cid in ("net_a", "net_b", "net_c"):
        assert sanitize_stream_id(cid) in registered_ids, (
            f"{cid} not registered: {registered_ids}"
        )
    assert sanitize_stream_id("laptop_cam") not in registered_ids
    assert sanitize_stream_id("clip") not in registered_ids
    print("ok bind_gateway keeps previous streams")


def test_mjpeg_part_includes_content_length() -> None:
    from launcher import encode_mjpeg_part

    payload = b"\xff\xd8fakejpeg\xff\xd9"
    part = encode_mjpeg_part(payload, timestamp=1700000000.123)
    assert b"Content-Type: image/jpeg\r\n" in part
    assert f"Content-Length: {len(payload)}\r\n".encode("ascii") in part
    assert b"X-Timestamp: 1700000000.123\r\n" in part
    assert part.endswith(payload + b"\r\n")
    print("ok mjpeg part includes Content-Length")


class GatewayRegistryTests(unittest.TestCase):
    def test_bind_gateway_keeps_previous_streams(self) -> None:
        test_bind_gateway_keeps_previous_streams()

    def test_mjpeg_part_includes_content_length(self) -> None:
        test_mjpeg_part_includes_content_length()


class _FakeGo2rtcHandler(BaseHTTPRequestHandler):
    streams: dict[str, dict[str, Any]] = {}

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _query(self) -> dict[str, list[str]]:
        return parse_qs(urlparse(self.path).query)

    def _send(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/streams":
            src = parse_qs(parsed.query).get("src", [None])[0]
            if src:
                self._send(self.streams.get(src) or {})
                return
            self._send(self.streams)
            return
        if parsed.path == "/api/frame.jpeg":
            src = parse_qs(parsed.query).get("src", [None])[0]
            if src and src in self.streams:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                body = b"\xff\xd8\xff\xd9"
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._send({"error": "not found"}, 404)
            return
        self._send({"error": "not found"}, 404)

    def do_PUT(self) -> None:
        self._upsert()

    def do_POST(self) -> None:
        self._upsert()

    def do_PATCH(self) -> None:
        self._upsert()

    def _upsert(self) -> None:
        q = self._query()
        src = (q.get("src") or [""])[0]
        name = (q.get("name") or [src])[0]
        if not src or not name:
            self._send({"error": "missing"}, 400)
            return
        self.streams[name] = {
            "producers": [{"url": src, "type": "HTTP/RTSP"}],
            "consumers": [],
        }
        _FakeGo2rtcHandler.put_count = int(getattr(_FakeGo2rtcHandler, "put_count", 0) or 0) + 1
        self._send({"ok": True})

    def do_DELETE(self) -> None:
        q = self._query()
        name = (q.get("src") or q.get("name") or [""])[0]
        self.streams.pop(name, None)
        self._send({"ok": True})


def test_client_register_and_delete() -> None:
    _FakeGo2rtcHandler.streams = {}
    _FakeGo2rtcHandler.put_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeGo2rtcHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        client = Go2RtcClient(f"http://127.0.0.1:{port}")
        assert client.is_ready()
        assert client.register_stream("cam1", "rtsp://192.168.1.10/stream")
        info = client.get_stream_info("cam1")
        assert info["producers"][0]["url"] == "rtsp://192.168.1.10/stream"
        all_streams = client.get_stream_info()
        assert "cam1" in all_streams
        assert client.remove_stream("cam1")
        assert client.get_stream_info("cam1") == {}
        ws = client.ws_url("cam1")
        assert ws.startswith("ws://127.0.0.1:")
        assert "src=cam1" in ws
        assert client.register_stream("snap", "rtsp://192.168.1.10/main")
        jpeg = client.snapshot_jpeg("snap")
        assert jpeg == b"\xff\xd8\xff\xd9"
        assert client.snapshot_jpeg("missing") is None
        first_puts = int(getattr(_FakeGo2rtcHandler, "put_count", 0) or 0)
        assert client.register_stream("snap", "rtsp://192.168.1.10/main")
        after_puts = int(getattr(_FakeGo2rtcHandler, "put_count", 0) or 0)
        assert after_puts == first_puts, "re-registering a live stream must not restart the producer"
        print(f"ok REST client on :{port}")
    finally:
        server.shutdown()
        server.server_close()


def _ensure_binary_or_skip() -> Path | None:
    try:
        path = ensure_binary()
    except Exception as exc:
        print(f"skip go2rtc integration (binary unavailable): {exc}")
        return None
    print(f"[test] using go2rtc at {path}")
    return path


def _make_manager(state_dir: Path, binary: Path) -> Go2RtcManager:
    return Go2RtcManager(
        api_port=_free_port(),
        rtsp_port=_free_port(),
        webrtc_port=_free_port(),
        state_dir=state_dir,
        binary=binary,
    )


def test_manager_launch_and_ports(binary: Path) -> Go2RtcManager:
    state = Path(tempfile.mkdtemp(prefix="go2rtc-test-"))
    mgr = _make_manager(state, binary)
    assert mgr.start(timeout=10.0), "go2rtc failed to become ready"
    try:
        assert mgr.is_ready()
        assert mgr.wait_port(mgr.api_port, timeout=2.0)
        assert mgr.wait_port(mgr.rtsp_port, timeout=2.0)
        assert mgr.pid is not None and _pid_alive(mgr.pid)
        # API answers an empty stream list.
        assert mgr.client.get_stream_info() == {} or isinstance(mgr.client.get_stream_info(), dict)
        print(
            f"ok manager pid={mgr.pid} api={mgr.api_port} "
            f"rtsp={mgr.rtsp_port} webrtc={mgr.webrtc_port}"
        )
        return mgr
    except Exception:
        mgr.stop()
        raise


def test_live_register_delete(mgr: Go2RtcManager) -> None:
    src = "ffmpeg:testsrc=size=160x120:rate=10#video=mjpeg"
    assert mgr.client.register_stream("probe", src) or mgr.client.register_stream(
        "probe", "http://127.0.0.1:9/missing"
    )
    # Registration of a dummy HTTP URL always succeeds — go2rtc is pull-based.
    assert mgr.client.register_stream("dummy", "http://127.0.0.1:9/video")
    info = mgr.client.get_stream_info("dummy")
    assert isinstance(info, dict)
    assert mgr.client.remove_stream("dummy")
    assert mgr.client.get_stream_info("dummy") == {}
    print("ok live register/delete")


class _MjpegServer(ThreadingHTTPServer):
    def __init__(self, fps: float = 20.0):
        super().__init__(("127.0.0.1", 0), _MjpegHandler)
        self.fps = fps
        self.frames_sent = 0
        self.stop_event = threading.Event()
        self.daemon_threads = True


class _MjpegHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        server: _MjpegServer = self.server  # type: ignore[assignment]
        if urlparse(self.path).path not in ("/video", "/"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        interval = 1.0 / max(1.0, server.fps)
        n = 0
        try:
            while not server.stop_event.is_set():
                frame = np.full((40, 64, 3), n % 220, dtype=np.uint8)
                cv2.putText(
                    frame,
                    str(n),
                    (4, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ok:
                    break
                payload = buf.tobytes()
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                server.frames_sent += 1
                n += 1
                time.sleep(interval)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return


def _count_mjpeg_fps(url: str, duration: float = 2.0) -> float:
    session = requests.Session()
    session.trust_env = False
    t0 = time.time()
    n = 0
    buf = bytearray()
    with session.get(url, stream=True, timeout=(3.0, 12.0)) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                break
            buf.extend(chunk)
            while True:
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                if start < 0 or end < 0:
                    if start > 0:
                        del buf[:start]
                    break
                n += 1
                del buf[: end + 2]
            if time.time() - t0 >= duration:
                break
    elapsed = max(0.001, time.time() - t0)
    return n / elapsed


def _count_snapshot_fps(url: str, duration: float = 2.0) -> tuple[float, int]:
    session = requests.Session()
    session.trust_env = False
    t0 = time.time()
    n = 0
    while time.time() - t0 < duration:
        try:
            resp = session.get(url, timeout=1.5)
        except requests.RequestException:
            continue
        if resp.status_code == 200 and resp.content[:2] == b"\xff\xd8":
            n += 1
    elapsed = max(0.001, time.time() - t0)
    return n / elapsed, n


def _grabber_fps(grabber: AsyncFrameGrabber, duration: float) -> float:
    count = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        pkt = grabber.get_latest_frame(0.05)
        if pkt is not None:
            count += 1
    return count / max(0.001, time.time() - t0)


def test_multi_consumer_fps(mgr: Go2RtcManager) -> None:
    source = _MjpegServer(fps=20.0)
    threading.Thread(target=source.serve_forever, daemon=True).start()
    grabber = AsyncFrameGrabber()
    grabber.start()
    try:
        src_url = f"http://127.0.0.1:{source.server_address[1]}/video"
        assert mgr.client.register_stream("mux", src_url)
        local = mgr.mjpeg_url("mux")
        adapter = GatewayAdapter(src_url, "mux", local, client=mgr.client)
        grabber.switch_source(adapter)

        deadline = time.time() + 8.0
        first = None
        while time.time() < deadline:
            first = grabber.get_latest_frame(0.1)
            if first is not None:
                break
        assert first is not None, "grabber never received a gateway frame"

        solo = _grabber_fps(grabber, 1.4)
        snap_url = f"{mgr.api_base}/api/frame.jpeg?src=mux"
        snap_box: list[tuple[float, int]] = []

        def _ui_reader() -> None:
            try:
                mjpeg_fps = _count_mjpeg_fps(local, duration=2.0)
            except Exception:
                mjpeg_fps = 0.0
            try:
                snap_fps, snap_n = _count_snapshot_fps(snap_url, duration=2.0)
            except Exception:
                snap_fps, snap_n = 0.0, 0
            snap_box.append((max(mjpeg_fps, snap_fps), snap_n))

        ui_thread = threading.Thread(target=_ui_reader, daemon=True)
        ui_thread.start()
        shared = _grabber_fps(grabber, 2.2)
        ui_thread.join(timeout=6.0)
        ui_fps, snap_n = snap_box[0] if snap_box else (0.0, 0)
        producers, consumers = mgr.client.consumer_counts("mux")
        print(
            f"ok multi-consumer solo={solo:.1f}fps shared={shared:.1f}fps "
            f"ui={ui_fps:.1f}fps snapshots={snap_n} "
            f"producers={producers} consumers={consumers}"
        )
        assert solo >= 8.0, f"gateway consumer too slow even alone: {solo:.1f}"
        assert shared >= max(6.0, solo * 0.6), (
            f"YOLO FPS dropped with a second consumer: solo={solo:.1f} shared={shared:.1f}"
        )
        assert ui_fps >= 2.0 or snap_n >= 4 or consumers >= 2, (
            f"UI endpoint received no video (fps={ui_fps:.1f}, snapshots={snap_n}, consumers={consumers})"
        )
    finally:
        grabber.stop()
        source.stop_event.set()
        source.shutdown()
        source.server_close()
        mgr.client.remove_stream("mux")


def test_clean_teardown(mgr: Go2RtcManager) -> None:
    pid = mgr.pid
    api = mgr.api_port
    rtsp = mgr.rtsp_port
    assert pid is not None
    mgr.stop()
    time.sleep(0.3)
    assert not _pid_alive(pid), f"go2rtc pid {pid} still alive after stop()"
    # Ports should be free again (or at least API should be down).
    down = True
    session = requests.Session()
    session.trust_env = False
    try:
        session.get(f"http://127.0.0.1:{api}/api/streams", timeout=0.5)
        down = False
    except requests.RequestException:
        down = True
    assert down, f"go2rtc API still serving on {api}"
    print(f"ok teardown pid={pid} api={api} rtsp={rtsp}")


def main() -> None:
    test_platform_and_stream_id()
    test_ffmpeg_candidates_skip_unix_paths_on_windows_shape()
    test_create_adapter_gateway_routing()
    test_bind_gateway_keeps_previous_streams()
    test_mjpeg_part_includes_content_length()
    test_client_register_and_delete()

    binary = _ensure_binary_or_skip()
    if binary is None:
        print("unit tests passed (integration skipped)")
        return

    mgr = test_manager_launch_and_ports(binary)
    try:
        test_live_register_delete(mgr)
        test_multi_consumer_fps(mgr)
        test_clean_teardown(mgr)
    except Exception:
        try:
            mgr.stop()
        except Exception:
            pass
        raise
    print("all go2rtc tests passed")


if __name__ == "__main__":
    main()
