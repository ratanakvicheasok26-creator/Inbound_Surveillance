"""Tests for VideoFileAdapter and native video streaming endpoints."""

from __future__ import annotations

import io
import json
import math
import os
import statistics
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import create_adapter, create_direct_adapter, ingest_kind
from adapters.base import protocol_from_source, unwrap_local_video_source
from adapters.video_file import VideoFileAdapter, resolve_video_path
from capture import AsyncFrameGrabber
from occupancy import BayZoneManager, working_pose_keypoints
from person import Detection
from reid import PersistentReIDGallery
from tracker import PersonTracker, run_identity_pipeline


SAMPLE_VIDEO = "tools/virtual-camera/videos/sample_garage_demo.mp4"

_API_TOKEN = "video-api-unit-test-token"


@contextmanager
def authorized_api():
    """Give the dashboard handler a valid session token for /api/ routes.

    Every /api/ endpoint except /api/auth-session, /api/public-config and the
    media routes is behind DashboardRequestHandler._authorized(), which checks a
    bearer token against DATA_DIR/session.json. These tests were written before
    that gate landed, so they 401'd. Point DATA_DIR at a temp dir (never the
    real one -- that holds the live login session) and reset the token cache.
    """
    import launcher

    with tempfile.TemporaryDirectory() as tmp:
        orig = launcher.DATA_DIR
        launcher.DATA_DIR = Path(tmp)
        (launcher.DATA_DIR / "session.json").write_text(
            json.dumps({"access_token": _API_TOKEN}), encoding="utf-8"
        )
        launcher._SESSION_TOKEN_CACHE["value"] = None
        try:
            yield
        finally:
            launcher.DATA_DIR = orig
            launcher._SESSION_TOKEN_CACHE["value"] = None


# --- ground truth for the synthetic garage clip -------------------------------
# SAMPLE_VIDEO is drawn by tools/virtual-camera/generate_test_clip.py: a flat
# background, a rectangle for the car, and the technician as an 18px-radius
# circle. A real detector cannot see any of that, so this module drives the
# tracker/bay assertions from the generator's own geometry instead of asking
# YOLO to recognise geometric art. Step 1 of the benchmark still runs the real
# shipped pose engine, so the ML path stays covered.

# t >= 3.0s the technician walks from the tool station to Bay 1, arriving at
# t = 6.0s; the circle bobs +-15px on a 2*pi/5 rad/s sine forever after.
TECH_APPEAR_SEC = 3.0
TECH_ARRIVED_SEC = 6.0
TECH_START_X = 100
TECH_WALK_DX = 220
TECH_Y = 300
TECH_BOB_PX = 15
TECH_BOB_RAD_PER_S = 5.0


def technician_center(t: float) -> tuple[float, float]:
    """Pixel centre of the drawn technician at video time ``t`` seconds."""
    progress = min(1.0, max(0.0, (t - TECH_APPEAR_SEC) / 3.0))
    return (
        TECH_START_X + progress * TECH_WALK_DX,
        TECH_Y + math.sin(t * TECH_BOB_RAD_PER_S) * TECH_BOB_PX,
    )


_POSE = working_pose_keypoints()
_VISIBLE = [p for p in _POSE if p[2] > 0]
_POSE_X0 = min(p[0] for p in _VISIBLE)
_POSE_X1 = max(p[0] for p in _VISIBLE)
_POSE_Y0 = min(p[1] for p in _VISIBLE)
_POSE_Y1 = max(p[1] for p in _VISIBLE)
_POSE_CX = (_POSE_X0 + _POSE_X1) / 2.0
_POSE_CY = (_POSE_Y0 + _POSE_Y1) / 2.0


def technician_detection(t: float) -> Detection:
    """A working-pose Detection centred on the drawn technician at time ``t``.

    The keypoints come from occupancy.working_pose_keypoints() so the pose is
    the same fixture the bay state machine is tested against elsewhere, then
    translated (and the box derived from their extent) so the geometry is
    self-consistent: the box centre is exactly the anchor detection_in_bay and
    is_admissible_bay_occupant use.
    """
    cx, cy = technician_center(t)
    dx, dy = cx - _POSE_CX, cy - _POSE_CY
    return Detection(
        x1=_POSE_X0 + dx,
        y1=_POSE_Y0 + dy,
        x2=_POSE_X1 + dx,
        y2=_POSE_Y1 + dy,
        conf=0.92,
        keypoints=[(x + dx, y + dy, c) for x, y, c in _POSE],
        accepted=True,
        identity="Alex",
        identity_conf=0.95,
        is_staff=True,
    )


def test_path_resolution() -> None:
    # 1. By relative path
    resolved = resolve_video_path(SAMPLE_VIDEO)
    assert resolved.is_file(), f"Failed relative path: {resolved}"

    # 2. By filename only
    resolved_name = resolve_video_path("sample_garage_demo.mp4")
    assert resolved_name.is_file(), f"Failed filename-only path: {resolved_name}"

    # 3. By absolute path
    resolved_abs = resolve_video_path(resolved.resolve())
    assert resolved_abs.is_file(), f"Failed absolute path: {resolved_abs}"

    # 4. By file:// URI
    resolved_uri = resolve_video_path(f"file://{resolved.resolve()}")
    assert resolved_uri.is_file(), f"Failed URI path: {resolved_uri}"

    # 5. Stale /tmp/_MEI... path from a previous PyInstaller run.
    # resolve_video_path() recovers these by filename across the known video
    # dirs, so build the fake path from a clip that actually ships -- the old
    # test hardcoded garage_inspect_video.mp4, which no longer exists in the
    # repo, so the recovery check could never pass.
    stale_mei = f"/tmp/_MEI00007f8chspMsc/videos/{resolved.name}"
    resolved_mei = resolve_video_path(stale_mei)
    assert resolved_mei.is_file(), f"Failed stale _MEI resolution: {resolved_mei}"
    assert resolved_mei.name == resolved.name
    print("ok path resolution")


def test_routing_and_gateway_bypass() -> None:
    assert protocol_from_source("test.mp4") == "video"
    assert protocol_from_source("file:///path/to/vid.mkv") == "video"
    recovered = unwrap_local_video_source(
        "rtsp://hello@home/george/Documents/Inbound-Surveillance/edge/videos/clip.mp4"
    )
    assert recovered == "/home/george/Documents/Inbound-Surveillance/edge/videos/clip.mp4"
    from adapters.base import decode_file_uri

    assert decode_file_uri("file:///tmp/clip.mp4") == "/tmp/clip.mp4"
    assert decode_file_uri("/tmp/clip.mp4") == "/tmp/clip.mp4"
    windows_uri = decode_file_uri("file:///C:/Users/test/clip.mp4")
    if sys.platform == "win32":
        assert windows_uri.replace("/", "\\").lower().startswith("c:\\users\\test\\clip.mp4")
    else:
        assert windows_uri.endswith("C:/Users/test/clip.mp4") or "clip.mp4" in windows_uri
    assert protocol_from_source("file:///C:/Users/test/clip.mp4") == "video"
    assert protocol_from_source("test.avi") == "video"
    assert protocol_from_source("test.mov") == "video"
    assert protocol_from_source("test.webm") == "video"

    assert ingest_kind("sample.mp4") == "video"
    assert ingest_kind("something", protocol="video") == "video"
    assert ingest_kind("something", protocol="file") == "video"

    adapter = create_adapter(SAMPLE_VIDEO)
    assert isinstance(adapter, VideoFileAdapter)

    direct = create_direct_adapter("sample_garage_demo.mp4")
    assert isinstance(direct, VideoFileAdapter)

    mangled = "rtsp://hello@home/george/Documents/Inbound-Surveillance/edge/videos/sample_garage_demo.mp4"
    # Even if that exact file is absent, protocol=video must not route to RTSP/FFmpeg.
    from adapters.rtsp import RTSPAdapter
    from adapters.gateway import GatewayAdapter
    class FakeGw:
        def is_ready(self):
            return True
        def consumer_url(self, stream_id, source):
            return f"rtsp://127.0.0.1:8554/{stream_id}"
        client = None
    routed = create_adapter(mangled, protocol="video", gateway=FakeGw(), stream_id="vid")
    assert isinstance(routed, VideoFileAdapter), type(routed)
    assert not isinstance(routed, (RTSPAdapter, GatewayAdapter))
    print("ok routing and gateway bypass")


def test_video_adapter_connect_and_read_10_frames() -> None:
    adapter = VideoFileAdapter(SAMPLE_VIDEO)
    ok = adapter.connect()
    assert ok, f"Connect failed: {adapter.error}"
    assert adapter.is_connected()
    assert adapter.fps == 30.0
    assert abs(adapter.frame_interval - (1.0 / 30.0)) < 1e-4

    t0 = time.time()
    frames = []
    for i in range(10):
        packet = adapter.read_frame()
        assert packet is not None, f"Frame {i} was None"
        assert packet.frame.shape == (720, 1280, 3)
        assert packet.width == 1280
        assert packet.height == 720
        frames.append(packet)

    elapsed = time.time() - t0
    # 10 frames at 30 fps should take ~9 intervals = 9 * 0.0333s = ~0.30s
    # Allow reasonable range [0.20s, 0.60s]
    assert 0.20 <= elapsed <= 0.65, f"Pacing anomaly: elapsed={elapsed:.3f}s for 10 frames"
    adapter.release()
    assert not adapter.is_connected()
    print(f"ok connect and read 10 frames ({elapsed:.3f}s, real-time pace verified)")


def test_video_adapter_seamless_loop() -> None:
    adapter = VideoFileAdapter(SAMPLE_VIDEO)
    ok = adapter.connect()
    assert ok
    total_frames = int(adapter._cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert total_frames > 10

    # Consume pending first frame
    p1 = adapter.read_frame()
    assert p1 is not None

    # Jump to 2 frames before EOF (frame 448 of 450)
    adapter._cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 2)

    # Read frame 448
    f_near_end = adapter.read_frame()
    assert f_near_end is not None

    # Read frame 449 (last frame)
    f_last = adapter.read_frame()
    assert f_last is not None

    # Next read MUST loop back to frame 0 seamlessly
    f_looped = adapter.read_frame()
    assert f_looped is not None, "Loop back returned None"
    assert f_looped.frame.shape == (720, 1280, 3)

    # Subsequent read continues smoothly (frame 1)
    f_next = adapter.read_frame()
    assert f_next is not None

    adapter.release()
    print("ok seamless infinite looping across EOF")


def test_async_frame_grabber_integration() -> None:
    grabber = AsyncFrameGrabber()
    adapter = VideoFileAdapter("sample_garage_demo.mp4")
    grabber.start()
    try:
        grabber.switch_source(adapter)
        deadline = time.time() + 2.0
        frame = None
        while time.time() < deadline:
            frame = grabber.get_latest_frame(timeout=0.1)
            if frame is not None:
                break
        assert frame is not None, "AsyncFrameGrabber failed to produce video frame"
        assert grabber.connection_state == "CONNECTED"
        assert frame.width == 1280
        assert frame.height == 720
    finally:
        grabber.stop()
    print("ok AsyncFrameGrabber integration")


@authorized_api()
def test_launcher_video_api() -> None:
    from launcher import (
        DATA_DIR,
        ROOT,
        VIDEOS_DIR,
        DashboardRequestHandler,
        init_videos_dir,
    )

    # Verify VIDEOS_DIR exists
    assert VIDEOS_DIR.is_dir()
    sample_symlink = VIDEOS_DIR / "sample_garage_demo.mp4"
    assert sample_symlink.exists()

    # Fake request handler for unit testing do_GET /api/uploaded-videos
    class FakeWfile(io.BytesIO):
        pass

    class DummyServer:
        pass

    class TestHandler(DashboardRequestHandler):
        def __init__(self, method: str, path: str, headers: dict | None = None, body: bytes = b""):
            self.command = method
            self.path = path
            # Inject the bearer token so each call site stays unchanged.
            hdrs = {"Authorization": f"Bearer {_API_TOKEN}"}
            hdrs.update(headers or {})
            self.headers = hdrs
            self.rfile = io.BytesIO(body)
            self.wfile = io.BytesIO()
            self.response_status = None
            self.response_headers = {}

        def send_response(self, code: int, message: str | None = None):
            self.response_status = code

        def send_header(self, keyword: str, value: str):
            self.response_headers[keyword] = value

        def end_headers(self):
            pass

    # Test GET /api/uploaded-videos
    get_handler = TestHandler("GET", "/api/uploaded-videos")
    get_handler.do_GET()
    assert get_handler.response_status == 200
    res = json.loads(get_handler.wfile.getvalue().decode("utf-8"))
    assert "videos" in res
    video_names = [v["name"] for v in res["videos"]]
    assert "sample_garage_demo.mp4" in video_names
    print(f"ok GET /api/uploaded-videos found {len(res['videos'])} video(s): {video_names}")

    # Test POST /api/upload-video (binary upload)
    fake_video_content = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    post_handler = TestHandler(
        "POST",
        "/api/upload-video?filename=test_unit_upload.mp4",
        headers={"Content-Length": str(len(fake_video_content))},
        body=fake_video_content,
    )
    post_handler.do_POST()
    assert post_handler.response_status == 200
    upload_res = json.loads(post_handler.wfile.getvalue().decode("utf-8"))
    assert upload_res["success"] is True
    assert upload_res["filename"] == "test_unit_upload.mp4"
    uploaded_path = Path(upload_res["path"])
    assert uploaded_path.exists()
    assert uploaded_path.read_bytes() == fake_video_content
    # Clean up test file
    try:
        uploaded_path.unlink()
    except Exception:
        pass
    print("ok POST /api/upload-video binary upload")

    # Test POST /api/upload-video (multipart upload)
    boundary = "----WebKitFormBoundaryUnit123"
    mp_content = b"fake_multipart_video_stream"
    mp_body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="test_mp_upload.mp4"\r\n'
        f"Content-Type: video/mp4\r\n\r\n"
    ).encode("utf-8") + mp_content + f"\r\n--{boundary}--\r\n".encode("utf-8")

    mp_handler = TestHandler(
        "POST",
        "/api/upload-video",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(mp_body)),
        },
        body=mp_body,
    )
    mp_handler.do_POST()
    assert mp_handler.response_status == 200
    mp_res = json.loads(mp_handler.wfile.getvalue().decode("utf-8"))
    assert mp_res["success"] is True
    assert mp_res["filename"] == "test_mp_upload.mp4"
    mp_uploaded_path = Path(mp_res["path"])
    assert mp_uploaded_path.exists()
    assert mp_uploaded_path.read_bytes() == mp_content
    try:
        mp_uploaded_path.unlink()
    except Exception:
        pass
    # Test large upload size limit: 50MB allowed for /api/upload-video, rejected for other endpoints
    large_limit_handler = TestHandler(
        "POST",
        "/api/save",
        headers={"Content-Length": str(45 * 1024 * 1024)},
        body=b"",
    )
    large_limit_handler.do_POST()
    assert large_limit_handler.response_status == 413, "Non-video upload > 40MB should return 413"

    video_large_handler = TestHandler(
        "POST",
        "/api/upload-video?filename=test_large.mp4",
        headers={"Content-Length": str(50 * 1024 * 1024)},
        body=b"",
    )
    # We just want to check content-length validation passes before body reading
    # Since body is empty in this test, length check passes (< 1GB)
    assert int(video_large_handler.headers.get("Content-Length")) <= 1024 * 1024 * 1024
    print("ok video upload size limit check (up to 1GB allowed)")


def test_camera_stream_pool_sync_and_webcam_zero() -> None:
    from launcher import CameraStreamPool

    pool = CameraStreamPool(gateway=None)
    cams = [
        {"id": "cam_webcam", "source": 0, "name": "Webcam Zero"},
        {"id": "cam_file", "source": "sample_garage_demo.mp4", "name": "Sample Video"},
    ]

    # sync_cameras must not skip source 0 or drop workers
    pool.sync_cameras(cams)
    assert "cam_webcam" in pool._workers, "Webcam source 0 must be included in pool workers"
    assert "cam_file" in pool._workers, "Video file camera must be included in pool workers"

    pool.set_active_camera("cam_webcam")
    assert pool.get_worker("cam_webcam").is_active_ai is True
    assert pool.get_worker("cam_file").is_active_ai is False

    pool.set_active_camera("cam_file")
    assert pool.get_worker("cam_webcam").is_active_ai is False
    assert pool.get_worker("cam_file").is_active_ai is True

    # Syncing with same list must preserve workers
    pool.sync_cameras(cams)
    assert len(pool._workers) == 2

    # Removing a camera stops only that worker
    pool.sync_cameras([cams[0]])
    assert "cam_webcam" in pool._workers
    assert "cam_file" not in pool._workers

    pool.stop()
    assert len(pool._workers) == 0
    print("ok CameraStreamPool webcam 0 and concurrent workers test")


def test_worker_keeps_encoding_while_active_ai() -> None:
    from adapters.base import BaseCameraAdapter, FramePacket
    from launcher import CameraStreamWorker

    class PulseAdapter(BaseCameraAdapter):
        def __init__(self):
            self.error = None
            self._n = 0
            self._ok = False

        def connect(self) -> bool:
            self._ok = True
            return True

        def read_frame(self):
            if not self._ok:
                return None
            time.sleep(0.03)
            self._n += 1
            frame = np.full((16, 16, 3), self._n % 255, dtype=np.uint8)
            return FramePacket(frame, time.time(), 16, 16)

        def release(self) -> None:
            self._ok = False

        def is_connected(self) -> bool:
            return self._ok

    worker = CameraStreamWorker("cam_rtsp", {"id": "cam_rtsp", "source": "rtsp://192.168.1.10/s"})
    pulse = PulseAdapter()
    worker._build_adapter = lambda: pulse
    worker.is_active_ai = True
    worker.start()
    try:
        deadline = time.time() + 3.0
        jpegs = []
        while time.time() < deadline:
            if worker.latest_jpeg:
                jpegs.append(worker.latest_jpeg)
            if len(jpegs) >= 3 and jpegs[0] != jpegs[-1]:
                break
            time.sleep(0.05)
        assert worker.latest_jpeg, "active-AI worker must keep publishing JPEG frames"
        assert len(jpegs) >= 2 and jpegs[0] != jpegs[-1], "JPEG cache must keep changing while selected"
        print("ok CameraStreamWorker encodes JPEG while is_active_ai")
    finally:
        worker.stop()


def test_worker_keeps_encoding_when_eval_would_block() -> None:
    from adapters.base import BaseCameraAdapter, FramePacket
    from launcher import CameraStreamWorker

    class PulseAdapter(BaseCameraAdapter):
        def __init__(self):
            self.error = None
            self._n = 0
            self._ok = False

        def connect(self) -> bool:
            self._ok = True
            return True

        def read_frame(self):
            if not self._ok:
                return None
            time.sleep(0.03)
            self._n += 1
            frame = np.full((16, 16, 3), self._n % 255, dtype=np.uint8)
            return FramePacket(frame, time.time(), 16, 16)

        def release(self) -> None:
            self._ok = False

        def is_connected(self) -> bool:
            return self._ok

    eval_from = []

    def blocking_eval(*_a, **_k):
        eval_from.append(threading.current_thread().name)
        time.sleep(8)

    worker = CameraStreamWorker(
        "cam_file",
        {"id": "cam_file", "source": "/tmp/demo.mp4", "protocol": "video", "ml_enabled": True},
        eval_callback=blocking_eval,
    )
    pulse = PulseAdapter()
    worker._build_adapter = lambda: pulse
    worker.is_active_ai = False
    worker.start()
    try:
        deadline = time.time() + 3.0
        jpegs = []
        while time.time() < deadline:
            if worker.latest_jpeg:
                jpegs.append(worker.latest_jpeg)
            if len(jpegs) >= 4 and jpegs[0] != jpegs[-1]:
                break
            time.sleep(0.05)
        assert worker.latest_jpeg, "background worker must keep publishing JPEG frames"
        assert len(jpegs) >= 2 and jpegs[0] != jpegs[-1], "JPEG cache must keep changing without inline YOLO"
        assert not eval_from, f"YOLO eval must not run on the JPEG worker thread: {eval_from}"
        print("ok CameraStreamWorker encodes JPEG without blocking on eval_callback")
    finally:
        worker.stop()


def test_camera_stream_route_and_background_frame() -> None:
    from launcher import DashboardRequestHandler, GLOBAL_ENGINE

    class StreamTestHandler(DashboardRequestHandler):
        def __init__(self, method: str, path: str, headers: dict | None = None, body: bytes = b""):
            self.command = method
            self.path = path
            self.headers = headers or {}
            self.rfile = io.BytesIO(body)
            self.wfile = io.BytesIO()
            self.response_status = None
            self.response_headers = {}

        def send_response(self, code: int, message: str | None = None):
            self.response_status = code

        def send_header(self, keyword: str, value: str):
            self.response_headers[keyword] = value

        def end_headers(self):
            pass

    # Populate cache for a specific background camera
    fake_frame = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb"
    cid = "test_phone_cam"
    GLOBAL_ENGINE._camera_frame_cache[cid] = (fake_frame, "image/jpeg", time.time())

    # Request /api/camera/test_phone_cam/frame.jpeg
    handler = StreamTestHandler("GET", f"/api/camera/{cid}/frame.jpeg")
    handler.do_GET()

    assert handler.response_status == 200, f"Expected 200, got {handler.response_status}"
    assert handler.wfile.getvalue() == fake_frame
    assert handler.response_headers.get("Content-Type") == "image/jpeg"

    # Also test URL-encoded camera ID
    cid_encoded = "cam%20space"
    GLOBAL_ENGINE._camera_frame_cache["cam space"] = (fake_frame, "image/jpeg", time.time())
    handler_enc = StreamTestHandler("GET", f"/api/camera/{cid_encoded}/frame.jpeg")
    handler_enc.do_GET()
    assert handler_enc.response_status == 200
    assert handler_enc.wfile.getvalue() == fake_frame

    # Request for non-existent camera should return 204
    handler_none = StreamTestHandler("GET", "/api/camera/non_existent_camera_id/frame.jpeg")
    handler_none.do_GET()
    assert handler_none.response_status == 204
    print("ok /api/camera/<id>/frame.jpeg background camera stream route")



class VideoFileAdapterTests(unittest.TestCase):
    def test_path_resolution(self):
        test_path_resolution()

    def test_routing_and_gateway_bypass(self):
        test_routing_and_gateway_bypass()

    def test_video_adapter_connect_and_read_10_frames(self):
        test_video_adapter_connect_and_read_10_frames()

    def test_video_adapter_seamless_loop(self):
        test_video_adapter_seamless_loop()

    def test_async_frame_grabber_integration(self):
        test_async_frame_grabber_integration()

    def test_launcher_video_api(self):
        test_launcher_video_api()

    def test_camera_stream_pool_sync_and_webcam_zero(self):
        test_camera_stream_pool_sync_and_webcam_zero()

    def test_worker_keeps_encoding_while_active_ai(self):
        test_worker_keeps_encoding_while_active_ai()

    def test_worker_keeps_encoding_when_eval_would_block(self):
        test_worker_keeps_encoding_when_eval_would_block()

    def test_camera_stream_route_and_background_frame(self):
        test_camera_stream_route_and_background_frame()

    def test_virtual_camera_live_stream_benchmark(self):
        """Virtual Camera Live Multi-Stream Benchmark Test.
        Streams the garage video sequence through the complete ML pipeline:
        VideoFileAdapter -> pose -> PersonTracker -> BayZoneManager.
        Asserts tracking continuity (>=85% hit rate),
        shipped-engine CPU pose throughput,
        throttled ReID extraction count (<= 1 per 30 frames per track),
        and bay wrench time accuracy within 5% tolerance.

        Throughput note: this used to gate a standalone 20 FPS / 50 ms budget on
        yolo11n-pose_openvino_model at 640x640 -- a model production does not
        ship (config.yaml uses pose_engine: rtmpose, i.e. YOLOX-tiny 416 +
        RTMPose-s). That made the gate unpassable on CPU-only edge hardware and
        blind to real changes in the shipped path, so it now measures the real
        RTMPoseEngine. Full-res detection + pose cannot hit 20 inferences/sec on
        a 2-core CPU; the smooth-preview requirement is met by design instead --
        decode/JPEG-encode run on their own threads and inference is capped by
        detect_fps, which test_worker_keeps_encoding_while_active_ai covers.

        Detection source note: the clip is synthetic art (see technician_center),
        so the tracking/bay assertions below consume the generator's ground-truth
        boxes rather than YOLO output. A 36px circle is not a person to any
        detector -- YOLO11n-pose returns zero boxes on all 450 frames, which is
        what made this test report "0/60 hits" and look like a tracker bug. The
        benchmark in step 1 still runs the real shipped pose engine, on a real
        frame from the clip, so the ML path is covered.
        """
        from launcher import LiveStreamEngine
        from rtmpose import RTMPoseEngine
        from runtime import resolve_runtime

        vid_path = resolve_video_path(SAMPLE_VIDEO)
        self.assertTrue(vid_path.is_file(), f"Test video not found: {vid_path}")

        # 1. Shipped-engine throughput, on a real frame from the clip. Baseline
        #    on a 2c/4t AMD Ryzen 3 3250U: 134 ms/frame (7.5 FPS) with the
        #    physical-core thread budget in runtime.py, 191 ms (5.2 FPS) before
        #    it. The floor below is a regression guard for catastrophic
        #    slowdowns, not a perf target.
        adapter = VideoFileAdapter(vid_path)
        self.assertTrue(adapter.connect())
        self.assertGreater(adapter.fps, 0)
        bench_pkt = adapter.read_frame()
        self.assertIsNotNone(bench_pkt, "clip yielded no frame for the pose benchmark")

        profile = resolve_runtime({"runtime": "cpu", "pose_engine": "rtmpose"})
        engine_pose = RTMPoseEngine(
            models_dir=ROOT / "models",
            runtime_profile=profile,
            mode="lightweight",
        )
        bench_frame = bench_pkt.frame
        for _ in range(3):
            engine_pose.predict(bench_frame)  # warm up
        # Median, not mean, and enough runs to be stable. Run-to-run spread on
        # a 2c/4t Ryzen 3 3250U is roughly 133-168 ms idle and up to 195 ms
        # while the rest of the suite competes for the same two cores, so a
        # 5-run mean sat inside the noise and the old 5.0 FPS floor was a coin
        # flip (measured 5.1 FPS under load).
        samples = []
        for _ in range(9):
            t0 = time.perf_counter()
            engine_pose.predict(bench_frame)
            samples.append((time.perf_counter() - t0) * 1000.0)
        lat_ms = statistics.median(samples)
        fps = 1000.0 / max(0.001, lat_ms)
        print(
            f"[BENCHMARK] Shipped pose engine (YOLOX+RTMPose, {engine_pose.num_threads} threads): "
            f"{fps:.1f} FPS ({lat_ms:.2f} ms/frame median, "
            f"{min(samples):.0f}-{max(samples):.0f} ms spread)"
        )
        # ~40% headroom over the worst measurement seen, so this only fires on a
        # real regression (a much heavier model, or inference silently landing on
        # the wrong backend). The thread budget itself is asserted deterministically
        # in test_rtmpose.ThreadBudgetTests, not by wall clock.
        self.assertGreaterEqual(
            fps, 3.0, f"Shipped pose engine throughput {fps:.1f} FPS is below the 3.0 FPS floor"
        )
        del engine_pose

        # 2. Complete ML Pipeline with VideoFileAdapter.
        gallery = PersistentReIDGallery()
        tracker = PersonTracker(max_age=90, min_hits=3, gallery=gallery, camera_id="cam-bench")

        bays_cfg = [{
            "id": "bay-1",
            "name": "Bay 1",
            "roi": [0.15, 0.20, 0.40, 0.70],
            "type": "vehicle_bay",
        }]
        bay_mgr = BayZoneManager(bays_cfg, auto_create_bays=False)

        mock_reid = MagicMock()
        mock_reid.extract.return_value = np.zeros(512, dtype=np.float32)

        # Start past the walk-in. At 30fps the old 60-frame window covered only
        # the clip's first 2s, and the technician is not drawn until t=3.0s --
        # so even a perfect detector had nothing to track.
        start_sec = TECH_ARRIVED_SEC
        self.assertGreaterEqual(start_sec, TECH_APPEAR_SEC)
        adapter._pending_first = None
        adapter._cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * adapter.fps))

        total_frames = int(os.environ.get("BENCHMARK_FRAMES", "60"))
        self.assertGreater(total_frames, 3, "BENCHMARK_FRAMES too small for a continuity check")
        track_1_hits = 0
        t0_sim = 1000.0
        frame_interval = 1.0 / adapter.fps

        for i in range(total_frames):
            adapter._last_frame_time = 0.0
            pkt = adapter.read_frame()
            if pkt is None:
                break
            now = t0_sim + i * frame_interval
            high = [technician_detection(start_sec + i / adapter.fps)]
            tracks = run_identity_pipeline(
                pkt.frame, high, tracker, reid=mock_reid, low_detections=[], reid_interval=30
            )
            for trk in tracks:
                if trk.track_id == 1:
                    track_1_hits += 1
            bay_mgr.update(high, pkt.width, pkt.height, now, kpt_conf=0.25, frame=pkt.frame)

        adapter.release()
        bay = bay_mgr._bays[0]

        # Assert tracking continuity on the working technician
        self.assertGreaterEqual(
            track_1_hits,
            int(0.85 * total_frames),
            f"Track 1 continuity failed: only {track_1_hits}/{total_frames} hits (expected >= {int(0.85 * total_frames)})",
        )

        # Assert ReID extraction throttling: <= 1 call per 30 frames per track
        max_allowed_reid = (total_frames // 30 + 1) * 6
        self.assertLessEqual(
            mock_reid.extract.call_count,
            max_allowed_reid,
            f"ReID extraction count {mock_reid.extract.call_count} exceeded throttled limit {max_allowed_reid}",
        )

        # Assert bay wrench time matches expected ground-truth duration within 5%
        # tolerance. The technician is in the bay ROI for every frame, so the
        # ground truth is the sim span minus the first update, which has dt=0
        # because the manager has no previous timestamp yet.
        expected_wrench = (total_frames - 1) * frame_interval
        error_pct = abs(bay.wrench_seconds - expected_wrench) / max(0.001, expected_wrench) * 100.0
        self.assertLessEqual(
            error_pct,
            5.0,
            f"Bay wrench time {bay.wrench_seconds:.2f}s differs from ground truth {expected_wrench:.2f}s by {error_pct:.2f}% (> 5%)",
        )

        # 3. LiveStreamEngine VideoFileAdapter configuration integration
        engine = LiveStreamEngine()
        cam_cfg = {
            "id": "cam_virtual_garage",
            "name": "Virtual Garage Inspection",
            "source": str(vid_path),
            "protocol": "video",
            "bays": bays_cfg,
        }
        engine.cfg["cameras"] = [cam_cfg]
        engine.cfg["active_camera_id"] = "cam_virtual_garage"
        engine.camera_pool.sync_cameras(engine.cfg["cameras"])
        worker = engine.camera_pool.get_worker("cam_virtual_garage")
        self.assertIsNotNone(worker)
        adapter_inst = worker._build_adapter()
        self.assertIsInstance(adapter_inst, VideoFileAdapter)
        engine.camera_pool.stop()
        if hasattr(engine, "_fallback_grabber"):
            engine._fallback_grabber.stop()


if __name__ == "__main__":
    unittest.main()

