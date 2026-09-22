"""Phase 2 ONVIF tests: GetProfiles / GetStreamUri XML and sub vs main pick."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import create_adapter, create_direct_adapter, ingest_kind
from adapters.base import protocol_from_source
from adapters.onvif import (
    ONVIFAdapter,
    inject_url_auth,
    onvif_xaddr,
    parse_device_info_xml,
    parse_profiles_xml,
    parse_stream_uri_xml,
    pick_sub_and_main,
)
from adapters.tapo import TapoAdapter, tapo_source_url
from adapters.webrtc import WebRTCAdapter, webrtc_go2rtc_url
from proof import scale_roi_px

GET_PROFILES = """\
<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:trt="http://www.onvif.org/ver10/media/wsdl"
            xmlns:tt="http://www.onvif.org/ver10/schema">
  <s:Body>
    <trt:GetProfilesResponse>
      <trt:Profiles token="Profile_1">
        <tt:Name>MainStream</tt:Name>
        <tt:VideoEncoderConfiguration>
          <tt:Encoding>H264</tt:Encoding>
          <tt:Resolution>
            <tt:Width>1920</tt:Width>
            <tt:Height>1080</tt:Height>
          </tt:Resolution>
          <tt:RateControl>
            <tt:FrameRateLimit>25</tt:FrameRateLimit>
          </tt:RateControl>
        </tt:VideoEncoderConfiguration>
      </trt:Profiles>
      <trt:Profiles token="Profile_2">
        <tt:Name>SubStream</tt:Name>
        <tt:VideoEncoderConfiguration>
          <tt:Encoding>H264</tt:Encoding>
          <tt:Resolution>
            <tt:Width>640</tt:Width>
            <tt:Height>360</tt:Height>
          </tt:Resolution>
          <tt:RateControl>
            <tt:FrameRateLimit>15</tt:FrameRateLimit>
          </tt:RateControl>
        </tt:VideoEncoderConfiguration>
      </trt:Profiles>
    </trt:GetProfilesResponse>
  </s:Body>
</s:Envelope>
"""

GET_STREAM_URI = """\
<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:trt="http://www.onvif.org/ver10/media/wsdl"
            xmlns:tt="http://www.onvif.org/ver10/schema">
  <s:Body>
    <trt:GetStreamUriResponse>
      <trt:MediaUri>
        <tt:Uri>rtsp://192.168.1.64:554/Streaming/Channels/102</tt:Uri>
      </trt:MediaUri>
    </trt:GetStreamUriResponse>
  </s:Body>
</s:Envelope>
"""

GET_DEVICE_INFO = """\
<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
  <s:Body>
    <tds:GetDeviceInformationResponse>
      <tds:Manufacturer>Hikvision</tds:Manufacturer>
      <tds:Model>DS-2CD2142FWD-I</tds:Model>
    </tds:GetDeviceInformationResponse>
  </s:Body>
</s:Envelope>
"""


def test_parse_profiles_and_pick() -> None:
    profiles = parse_profiles_xml(GET_PROFILES)
    assert len(profiles) == 2
    by_token = {p.token: p for p in profiles}
    assert by_token["Profile_1"].width == 1920
    assert by_token["Profile_1"].height == 1080
    assert by_token["Profile_1"].fps == 25
    assert by_token["Profile_1"].encoding == "H264"
    assert by_token["Profile_2"].width == 640
    sub, main = pick_sub_and_main(profiles)
    assert sub is not None and main is not None
    assert sub.token == "Profile_2"
    assert main.token == "Profile_1"
    print("ok profile pick sub=640x360 main=1920x1080")


def test_parse_stream_uri_and_auth() -> None:
    uri = parse_stream_uri_xml(GET_STREAM_URI)
    assert uri == "rtsp://192.168.1.64:554/Streaming/Channels/102"
    injected = inject_url_auth(uri, "admin", "secret")
    assert injected.startswith("rtsp://admin:secret@192.168.1.64")
    already = inject_url_auth("rtsp://admin:x@host/stream", "u", "p")
    assert already == "rtsp://admin:x@host/stream"
    print("ok GetStreamUri + auth inject")


def test_parse_device_info() -> None:
    manufacturer, model = parse_device_info_xml(GET_DEVICE_INFO)
    assert manufacturer == "Hikvision"
    assert model == "DS-2CD2142FWD-I"
    print("ok device info")


def test_onvif_xaddr_and_construction_is_io_free() -> None:
    assert onvif_xaddr("onvif://192.168.1.64") == "http://192.168.1.64/onvif/device_service"
    assert onvif_xaddr("192.168.1.64") == "http://192.168.1.64/onvif/device_service"
    assert onvif_xaddr("", ["http://192.168.1.64/onvif/device_service"]).endswith("device_service")
    t0 = __import__("time").time()
    adapter = ONVIFAdapter("http://192.0.2.1/onvif/device_service", "admin", "pass")
    elapsed = __import__("time").time() - t0
    assert elapsed < 0.05, f"ONVIFAdapter __init__ did I/O ({elapsed:.3f}s)"
    assert adapter.xaddr.endswith("device_service")
    ptz = adapter.continuous_move(1, 0, 0)
    assert ptz == {"ok": False, "error": "PTZ not available"}
    assert adapter.stop()["ok"] is False
    assert adapter.preset("home")["error"] == "PTZ not available"
    print("ok ONVIF construction + PTZ stubs")


def test_factory_protocol_routing() -> None:
    assert protocol_from_source("onvif://192.168.1.1") == "onvif"
    assert protocol_from_source("http://192.168.1.1/onvif/device_service") == "onvif"
    assert protocol_from_source("tapo://u:p@192.168.1.20") == "tapo"
    assert protocol_from_source("whep://192.168.1.9/endpoint") == "webrtc"
    assert protocol_from_source("webrtc://192.168.1.9:1984/") == "webrtc"
    assert protocol_from_source("http://192.168.1.8:8080/video") == "phone"
    assert ingest_kind("rtsp://cam/stream", protocol="onvif") == "rtsp"
    assert ingest_kind("http://192.168.1.1/onvif/device_service") == "onvif"
    assert ingest_kind("192.168.1.20", protocol="tapo") == "tapo"
    assert ingest_kind("https://nvr/whep", protocol="webrtc") == "webrtc"
    onvif = create_adapter(
        "http://192.168.1.64/onvif/device_service",
        protocol="onvif",
        username="admin",
        password="x",
    )
    assert isinstance(onvif, ONVIFAdapter)
    tapo = create_direct_adapter("tapo://u:p@192.168.1.20", protocol="tapo")
    assert isinstance(tapo, TapoAdapter)
    rtc = create_adapter("whep://192.168.1.9/whep", protocol="webrtc")
    assert isinstance(rtc, WebRTCAdapter)
    print("ok factory onvif/tapo/webrtc routing")


def test_tapo_and_webrtc_helpers() -> None:
    assert tapo_source_url("192.168.1.20", "user", "pass").startswith("tapo://user:pass@")
    assert tapo_source_url("tapo://u:p@host") == "tapo://u:p@host"
    t0 = __import__("time").time()
    TapoAdapter("192.168.1.20", username="u", password="p")
    WebRTCAdapter("whep://192.168.1.9/whep")
    assert __import__("time").time() - t0 < 0.05
    assert webrtc_go2rtc_url("whep://192.168.1.9/whep") == "http://192.168.1.9/whep"
    assert webrtc_go2rtc_url("webrtc://nvr.local:1984/").startswith("webrtc:")
    print("ok tapo/webrtc helpers")


def test_scale_roi_px() -> None:
    roi = (64, 36, 192, 108)
    scaled = scale_roi_px(roi, (640, 360), (1920, 1080))
    assert scaled == (192, 108, 576, 324)
    print("ok ROI scale 640x360 -> 1920x1080")


def test_webrtc_fails_cleanly_without_gateway() -> None:
    adapter = WebRTCAdapter("whep://192.0.2.1/whep")
    assert adapter.connect() is False
    assert adapter.error and "go2rtc" in adapter.error
    print("ok webrtc fail without gateway")


def test_camera_schema_main_source() -> None:
    from launcher import _normalize_cameras, upsert_camera

    cams = _normalize_cameras(
        [
            {
                "id": "cam-01",
                "name": "Front Lift Bay",
                "source": "rtsp://x/sub",
                "main_source": "rtsp://x/main",
                "protocol": "rtsp",
                "vendor": "hikvision",
                "xaddrs": ["http://192.168.1.64/onvif/device_service"],
            }
        ]
    )
    assert cams[0]["main_source"] == "rtsp://x/main"
    assert cams[0]["xaddrs"][0].endswith("device_service")
    cfg: dict = {"cameras": cams}
    entry = upsert_camera(
        cfg,
        {"id": "cam-01", "name": "Front Lift Bay", "source": "rtsp://x/sub", "main_source": "rtsp://x/main2"},
    )
    assert entry["main_source"] == "rtsp://x/main2"
    print("ok camera schema main_source")


if __name__ == "__main__":
    test_parse_profiles_and_pick()
    test_parse_stream_uri_and_auth()
    test_parse_device_info()
    test_onvif_xaddr_and_construction_is_io_free()
    test_factory_protocol_routing()
    test_tapo_and_webrtc_helpers()
    test_scale_roi_px()
    test_webrtc_fails_cleanly_without_gateway()
    test_camera_schema_main_source()
    print("all onvif tests passed")
