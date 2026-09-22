"""Phase 1 discovery tests: ProbeMatches XML, fake mDNS, non-blocking engine."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discovery.scanner import (
    DiscoveredDevice,
    DiscoveryEngine,
    device_from_mdns,
    enumerate_local_webcams,
    parse_probe_matches,
)

PROBE_MATCHES = """\
<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
            xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <s:Body>
    <d:ProbeMatches>
      <d:ProbeMatch>
        <d:Types>dn:NetworkVideoTransmitter</d:Types>
        <d:Scopes>onvif://www.onvif.org/name/FrontDoor onvif://www.onvif.org/hardware/DS-2CD2142FWD-I onvif://www.onvif.org/manufacturer/Hikvision</d:Scopes>
        <d:XAddrs>http://192.168.1.64/onvif/device_service</d:XAddrs>
        <d:MetadataVersion>1</d:MetadataVersion>
      </d:ProbeMatch>
    </d:ProbeMatches>
  </s:Body>
</s:Envelope>
"""


def test_parse_probe_matches() -> None:
    devices = parse_probe_matches(PROBE_MATCHES, from_addr="192.168.1.64")
    assert len(devices) == 1
    dev = devices[0]
    assert dev.ip == "192.168.1.64"
    assert dev.port == 80
    assert dev.service_type == "onvif"
    assert dev.name == "FrontDoor"
    assert dev.model == "DS-2CD2142FWD-I"
    assert dev.manufacturer == "Hikvision"
    assert dev.xaddrs == ["http://192.168.1.64/onvif/device_service"]
    payload = dev.as_dict()
    assert payload["ip"] == "192.168.1.64"
    print("ok parse ProbeMatches")


def test_device_from_mdns() -> None:
    onvif = device_from_mdns(
        "YardCam._onvif._tcp.local.",
        "_onvif._tcp.local.",
        ["192.168.1.50"],
        80,
        {b"manufacturer": b"Tapo", "model": "C310"},
    )
    assert onvif is not None
    assert onvif.ip == "192.168.1.50"
    assert onvif.service_type == "onvif"
    assert onvif.port == 80
    assert "onvif" in onvif.xaddrs[0]
    assert onvif.manufacturer == "Tapo"
    assert onvif.model == "C310"

    rtsp = device_from_mdns("NVR._rtsp._tcp.local.", "_rtsp._tcp.local.", ["10.0.0.9"], 554, {})
    assert rtsp is not None
    assert rtsp.service_type == "rtsp"
    assert rtsp.xaddrs[0].startswith("rtsp://")

    http = device_from_mdns("Phone._http._tcp.local.", "_http._tcp.local.", ["10.0.0.8"], 8080, {})
    assert http is not None
    assert http.service_type == "http"
    print("ok fake mDNS records")


def test_local_webcams_do_not_raise() -> None:
    cams = enumerate_local_webcams()
    assert isinstance(cams, list)
    for cam in cams:
        assert cam.service_type == "webcam"
        assert cam.as_dict()["name"]
    print(f"ok local webcams n={len(cams)}")


def test_engine_scan_returns_immediately_and_coalesces() -> None:
    started: list[int] = []

    def scan() -> list[DiscoveredDevice]:
        started.append(1)
        time.sleep(0.25)
        return [
            DiscoveredDevice(
                ip="192.168.1.64",
                port=80,
                service_type="onvif",
                name="FrontDoor",
                xaddrs=["http://192.168.1.64/onvif/device_service"],
                manufacturer="Hikvision",
                model="DS-2CD",
            )
        ]

    engine = DiscoveryEngine(scan_fn=scan)
    idle = engine.results()
    assert idle["status"] == "idle"
    assert idle["devices"] == []

    t0 = time.time()
    first = engine.start_scan()
    second = engine.start_scan()
    elapsed = time.time() - t0
    assert elapsed < 0.05, f"start_scan blocked ({elapsed:.3f}s)"
    assert first == {"success": True, "status": "scanning"}
    assert second["status"] == "scanning"
    scanning = engine.results()
    assert scanning["status"] == "scanning"

    deadline = time.time() + 2.0
    results = scanning
    while time.time() < deadline:
        results = engine.results()
        if results["status"] == "done":
            break
        time.sleep(0.02)
    assert results["status"] == "done"
    assert results["error"] is None
    assert len(started) == 1, "concurrent scans should coalesce"
    assert results["devices"][0]["name"] == "FrontDoor"
    print("ok discovery engine coalesce")


def test_engine_error_status() -> None:
    def boom() -> list[DiscoveredDevice]:
        raise RuntimeError("udp blocked")

    engine = DiscoveryEngine(scan_fn=boom)
    engine.start_scan()
    deadline = time.time() + 2.0
    results = engine.results()
    while time.time() < deadline and results["status"] == "scanning":
        time.sleep(0.02)
        results = engine.results()
    assert results["status"] == "error"
    assert "udp blocked" in (results["error"] or "")
    print("ok discovery error status")


if __name__ == "__main__":
    test_parse_probe_matches()
    test_device_from_mdns()
    test_local_webcams_do_not_raise()
    test_engine_scan_returns_immediately_and_coalesces()
    test_engine_error_status()
    print("all discovery tests passed")
