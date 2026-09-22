"""WS-Discovery + mDNS + local webcam scanner.

Probes run on a worker thread. HTTP handlers only call ``start_scan`` /
``results`` — never the UDP or mDNS I/O.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


WS_DISCOVERY_ADDR = ("239.255.255.250", 3702)
WS_DISCOVERY_TIMEOUT = 2.0

_MDNS_TYPES = (
    "_onvif._tcp.local.",
    "_rtsp._tcp.local.",
    "_http._tcp.local.",
)


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1] if tag.startswith("{") else tag


def _text(el: ET.Element | None) -> str:
    if el is None or el.text is None:
        return ""
    return str(el.text).strip()


def _find_local(root: ET.Element, name: str) -> ET.Element | None:
    for child in root.iter():
        if _local(child.tag) == name:
            return child
    return None


def _findall_local(root: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in root.iter() if _local(child.tag) == name]


def _scope_value(scopes: str, key: str) -> str:
    needle = f"/{key}/"
    for part in (scopes or "").split():
        idx = part.lower().find(needle)
        if idx < 0:
            continue
        return part[idx + len(needle) :].replace("%20", " ").strip()
    return ""


def _host_port_from_url(url: str, default_port: int = 80) -> tuple[str, int]:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = int(parsed.port or default_port)
    return host, port


@dataclass
class DiscoveredDevice:
    ip: str
    port: int
    service_type: str
    name: str = ""
    xaddrs: list[str] = field(default_factory=list)
    manufacturer: str = ""
    model: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ip": self.ip,
            "port": self.port,
            "service_type": self.service_type,
            "name": self.name,
            "xaddrs": list(self.xaddrs),
            "manufacturer": self.manufacturer,
            "model": self.model,
        }

    def identity_key(self) -> tuple[str, int, str]:
        return (self.ip, int(self.port or 0), self.service_type)


def parse_probe_matches(xml_text: str | bytes, from_addr: str | None = None) -> list[DiscoveredDevice]:
    """Parse a WS-Discovery ProbeMatches SOAP body into devices."""
    if isinstance(xml_text, bytes):
        xml_text = xml_text.decode("utf-8", errors="replace")
    text = str(xml_text or "").strip()
    if not text:
        return []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []

    devices: list[DiscoveredDevice] = []
    matches = _findall_local(root, "ProbeMatch")
    if not matches:
        # Some stacks nest a single match without the wrapper name we expect.
        if _find_local(root, "XAddrs") is not None:
            matches = [root]
    for match in matches:
        xaddrs_el = _find_local(match, "XAddrs")
        xaddrs = [part for part in _text(xaddrs_el).split() if part]
        scopes = _text(_find_local(match, "Scopes"))
        types = _text(_find_local(match, "Types")).lower()
        name = _scope_value(scopes, "name") or _scope_value(scopes, "hardware")
        model = _scope_value(scopes, "hardware")
        manufacturer = _scope_value(scopes, "manufacturer") or _scope_value(scopes, "brand")
        ip = from_addr or ""
        port = 80
        if xaddrs:
            host, port = _host_port_from_url(xaddrs[0], 80)
            ip = ip or host
        if not ip:
            continue
        service_type = "onvif"
        if "networkvideotransmitter" in types or "onvif" in types:
            service_type = "onvif"
        devices.append(
            DiscoveredDevice(
                ip=ip,
                port=port,
                service_type=service_type,
                name=name or ip,
                xaddrs=xaddrs,
                manufacturer=manufacturer,
                model=model,
            )
        )
    return devices


def build_probe_xml() -> str:
    mid = f"uuid:{uuid.uuid4()}"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"'
        ' xmlns:a="http://schemas.xmlsoap.org/ws/2004/08/addressing"'
        ' xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"'
        ' xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
        "<s:Header>"
        "<a:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</a:Action>"
        f"<a:MessageID>{mid}</a:MessageID>"
        "<a:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</a:To>"
        "</s:Header>"
        "<s:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></s:Body>"
        "</s:Envelope>"
    )


def ws_discovery(timeout: float = WS_DISCOVERY_TIMEOUT) -> list[DiscoveredDevice]:
    """UDP multicast Probe. Short socket timeout — never called from HTTP."""
    probe = build_probe_xml().encode("utf-8")
    devices: list[DiscoveredDevice] = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        except OSError:
            pass
        sock.settimeout(max(0.2, float(timeout)))
        sock.bind(("", 0))
        try:
            sock.sendto(probe, WS_DISCOVERY_ADDR)
        except OSError:
            return []
        deadline = time.time() + max(0.2, float(timeout))
        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            sock.settimeout(max(0.05, remaining))
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                break
            except OSError:
                break
            from_ip = addr[0] if addr else None
            devices.extend(parse_probe_matches(data, from_ip))
    finally:
        try:
            sock.close()
        except OSError:
            pass
    return _dedupe(devices)


def device_from_mdns(
    name: str,
    type_: str,
    addresses: list[str] | tuple[str, ...],
    port: int,
    properties: dict[str, Any] | None = None,
) -> DiscoveredDevice | None:
    """Build a device from a zeroconf-style service record (no I/O)."""
    props = properties or {}
    ip = ""
    for addr in addresses or []:
        text = str(addr).strip()
        if not text or text.startswith(":"):
            continue
        ip = text
        break
    if not ip:
        return None
    type_l = str(type_ or "").lower()
    if "_onvif._tcp" in type_l:
        service_type = "onvif"
    elif "_rtsp._tcp" in type_l:
        service_type = "rtsp"
    else:
        service_type = "http"
    pretty = str(name or "").replace("." + str(type_ or "").rstrip("."), "")
    pretty = pretty.rstrip(".")
    def _prop(key: str) -> str:
        raw = props.get(key.encode("utf-8"), props.get(key, ""))
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw or "")

    manufacturer = _prop("manufacturer")
    model = _prop("model")
    xaddrs: list[str] = []
    if service_type == "onvif":
        xaddrs = [f"http://{ip}:{int(port or 80)}/onvif/device_service"]
    elif service_type == "rtsp":
        xaddrs = [f"rtsp://{ip}:{int(port or 554)}/"]
    else:
        xaddrs = [f"http://{ip}:{int(port or 80)}/"]
    return DiscoveredDevice(
        ip=ip,
        port=int(port or 0),
        service_type=service_type,
        name=pretty or ip,
        xaddrs=xaddrs,
        manufacturer=manufacturer,
        model=model,
    )


def mdns_scan(timeout: float = WS_DISCOVERY_TIMEOUT) -> list[DiscoveredDevice]:
    """Browse ONVIF/RTSP/HTTP via zeroconf. Skip entirely if the import fails."""
    try:
        from zeroconf import ServiceBrowser, Zeroconf
    except Exception:
        return []

    found: list[DiscoveredDevice] = []
    lock = threading.Lock()

    class _Listener:
        def add_service(self, zc: Any, type_: str, name: str) -> None:
            try:
                info = zc.get_service_info(type_, name, timeout=int(max(0.2, timeout) * 1000))
            except Exception:
                return
            if info is None:
                return
            addresses: list[str] = []
            try:
                addresses = list(info.parsed_addresses())
            except Exception:
                for raw in getattr(info, "addresses", None) or []:
                    try:
                        addresses.append(socket.inet_ntoa(raw))
                    except OSError:
                        continue
            props = {}
            try:
                props = dict(info.properties or {})
            except Exception:
                props = {}
            device = device_from_mdns(name, type_, addresses, int(getattr(info, "port", 0) or 0), props)
            if device is None:
                return
            with lock:
                found.append(device)

        def remove_service(self, zc: Any, type_: str, name: str) -> None:
            return None

        def update_service(self, zc: Any, type_: str, name: str) -> None:
            return None

    zc = None
    browsers: list[Any] = []
    try:
        zc = Zeroconf()
        listener = _Listener()
        for svc in _MDNS_TYPES:
            try:
                browsers.append(ServiceBrowser(zc, svc, listener))
            except Exception:
                continue
        time.sleep(max(0.2, float(timeout)))
    except Exception:
        return []
    finally:
        for browser in browsers:
            try:
                browser.cancel()
            except Exception:
                pass
        if zc is not None:
            try:
                zc.close()
            except Exception:
                pass
    return _dedupe(found)


def enumerate_local_webcams() -> list[DiscoveredDevice]:
    """List local capture indexes without opening the devices."""
    devices: list[DiscoveredDevice] = []
    indexes: list[int] = []
    if sys.platform.startswith("linux"):
        for path in sorted(Path("/dev").glob("video*")):
            name = path.name
            if not name.startswith("video"):
                continue
            suffix = name[5:]
            if suffix.isdigit():
                indexes.append(int(suffix))
    else:
        indexes = [0, 1, 2]
    seen: set[int] = set()
    for idx in indexes:
        if idx in seen:
            continue
        seen.add(idx)
        devices.append(
            DiscoveredDevice(
                ip="127.0.0.1",
                port=idx,
                service_type="webcam",
                name=f"Webcam {idx}",
                xaddrs=[str(idx)],
            )
        )
    return devices


def _dedupe(devices: list[DiscoveredDevice]) -> list[DiscoveredDevice]:
    out: list[DiscoveredDevice] = []
    seen: set[tuple[str, int, str]] = set()
    for device in devices:
        key = device.identity_key()
        if key in seen:
            continue
        seen.add(key)
        out.append(device)
    return out


def run_discovery(timeout: float = WS_DISCOVERY_TIMEOUT) -> list[DiscoveredDevice]:
    """Run WS-Discovery, mDNS (if available), and local webcam enumeration."""
    devices: list[DiscoveredDevice] = []
    try:
        devices.extend(ws_discovery(timeout=timeout))
    except Exception:
        pass
    try:
        devices.extend(mdns_scan(timeout=timeout))
    except Exception:
        pass
    try:
        devices.extend(enumerate_local_webcams())
    except Exception:
        pass
    return _dedupe(devices)


class DiscoveryEngine:
    """Singleton-style scanner owned by ``LiveStreamEngine``.

    ``start_scan`` returns immediately. Concurrent calls coalesce into the
    in-flight worker. Results are stored under a lock.
    """

    def __init__(self, scan_fn: Callable[[], list[DiscoveredDevice]] | None = None):
        self._scan_fn = scan_fn or run_discovery
        self._lock = threading.Lock()
        self._status = "idle"
        self._devices: list[DiscoveredDevice] = []
        self._error: str | None = None
        self._started_at: float | None = None
        self._thread: threading.Thread | None = None

    def start_scan(self) -> dict[str, Any]:
        with self._lock:
            if self._status == "scanning":
                return {"success": True, "status": "scanning"}
            self._status = "scanning"
            self._error = None
            self._started_at = time.time()
            thread = threading.Thread(target=self._run, name="DiscoveryScan", daemon=True)
            self._thread = thread
        thread.start()
        return {"success": True, "status": "scanning"}

    def results(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "devices": [d.as_dict() for d in self._devices],
                "error": self._error,
                "started_at": self._started_at,
            }

    def _run(self) -> None:
        try:
            devices = list(self._scan_fn() or [])
            with self._lock:
                self._devices = devices
                self._status = "done"
                self._error = None
        except Exception as exc:
            with self._lock:
                self._status = "error"
                self._error = str(exc)
