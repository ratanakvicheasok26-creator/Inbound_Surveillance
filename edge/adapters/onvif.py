"""Lightweight ONVIF SOAP adapter (requests + xml.etree, not onvif-zeep).

Construction is I/O-free. SOAP, Digest auth, and stream URI resolution run
on the grabber thread or an onboard worker — never on the HTTP handler.
"""

from __future__ import annotations

import base64
import hashlib
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import quote, urlparse, urlunparse
from xml.sax.saxutils import escape as xml_escape

import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

from adapters.base import BaseCameraAdapter, FramePacket, redact_source, safe_release


SOAP_TIMEOUT = 3.0

_DEVICE_NS = "http://www.onvif.org/ver10/device/wsdl"
_MEDIA_NS = "http://www.onvif.org/ver10/media/wsdl"
_SCHEMA_NS = "http://www.onvif.org/ver10/schema"


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


def _int_text(el: ET.Element | None, default: int = 0) -> int:
    try:
        return int(float(_text(el)))
    except (TypeError, ValueError):
        return default


def _float_text(el: ET.Element | None, default: float = 0.0) -> float:
    try:
        return float(_text(el))
    except (TypeError, ValueError):
        return default


def onvif_xaddr(source: Any, xaddrs: Any = None) -> str:
    if isinstance(xaddrs, (list, tuple)):
        for item in xaddrs:
            text = str(item or "").strip()
            if text:
                return text
    elif xaddrs:
        text = str(xaddrs).strip()
        if text:
            return text
    text = str(source or "").strip()
    lower = text.lower()
    if lower.startswith("onvif://"):
        rest = text[8:]
        if "/" not in rest.split("@")[-1]:
            return f"http://{rest}/onvif/device_service"
        return f"http://{rest}"
    if text and not lower.startswith("http://") and not lower.startswith("https://"):
        return f"http://{text}/onvif/device_service"
    return text


def inject_url_auth(url: str, username: str, password: str) -> str:
    if not username:
        return url
    parsed = urlparse(url)
    if parsed.username:
        return url
    host = parsed.hostname or ""
    if not host:
        return url
    port = f":{parsed.port}" if parsed.port else ""
    user = quote(username, safe="")
    pw = quote(password, safe="")
    netloc = f"{user}:{pw}@{host}{port}"
    return urlunparse(
        (parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )


@dataclass
class MediaProfile:
    token: str
    name: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    encoding: str = ""

    @property
    def pixels(self) -> int:
        return max(0, int(self.width)) * max(0, int(self.height))

    def as_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "encoding": self.encoding,
        }


def parse_profiles_xml(xml_text: str | bytes) -> list[MediaProfile]:
    if isinstance(xml_text, bytes):
        xml_text = xml_text.decode("utf-8", errors="replace")
    text = str(xml_text or "").strip()
    if not text:
        return []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    out: list[MediaProfile] = []
    for node in _findall_local(root, "Profiles"):
        token = str(node.attrib.get("token") or node.attrib.get("Token") or "").strip()
        if not token:
            continue
        encoder = _find_local(node, "VideoEncoderConfiguration")
        width = 0
        height = 0
        fps = 0.0
        encoding = ""
        if encoder is not None:
            encoding = _text(_find_local(encoder, "Encoding"))
            width = _int_text(_find_local(encoder, "Width"))
            height = _int_text(_find_local(encoder, "Height"))
            fps = _float_text(_find_local(encoder, "FrameRateLimit"))
        out.append(
            MediaProfile(
                token=token,
                name=_text(_find_local(node, "Name")) or token,
                width=width,
                height=height,
                fps=fps,
                encoding=encoding,
            )
        )
    return out


def parse_stream_uri_xml(xml_text: str | bytes) -> str:
    if isinstance(xml_text, bytes):
        xml_text = xml_text.decode("utf-8", errors="replace")
    text = str(xml_text or "").strip()
    if not text:
        return ""
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return ""
    uri = _find_local(root, "Uri")
    if uri is not None:
        return _text(uri)
    media = _find_local(root, "MediaUri")
    if media is not None:
        return _text(_find_local(media, "Uri")) or _text(media)
    return ""


def parse_capabilities_media_xaddr(xml_text: str | bytes) -> str:
    if isinstance(xml_text, bytes):
        xml_text = xml_text.decode("utf-8", errors="replace")
    try:
        root = ET.fromstring(str(xml_text or "").strip())
    except ET.ParseError:
        return ""
    media = _find_local(root, "Media")
    if media is None:
        return ""
    xaddr = media.attrib.get("XAddr") or media.attrib.get("xaddr") or ""
    if xaddr:
        return str(xaddr).strip()
    return _text(_find_local(media, "XAddr"))


def parse_device_info_xml(xml_text: str | bytes) -> tuple[str, str]:
    if isinstance(xml_text, bytes):
        xml_text = xml_text.decode("utf-8", errors="replace")
    try:
        root = ET.fromstring(str(xml_text or "").strip())
    except ET.ParseError:
        return "", ""
    manufacturer = _text(_find_local(root, "Manufacturer"))
    model = _text(_find_local(root, "Model"))
    return manufacturer, model


def pick_sub_and_main(profiles: list[MediaProfile]) -> tuple[MediaProfile | None, MediaProfile | None]:
    """Lowest-res profile for AI, highest-res for evidence."""
    usable = [p for p in profiles if p.token]
    if not usable:
        return None, None
    ranked = sorted(usable, key=lambda p: (p.pixels or 0, p.width, p.height))
    return ranked[0], ranked[-1]


def _media_fallback_xaddr(device_xaddr: str) -> str:
    parsed = urlparse(device_xaddr)
    path = parsed.path or "/"
    lower = path.lower()
    if "device" in lower:
        path = path.replace("device_service", "media_service").replace("Device", "Media")
        if path == parsed.path:
            path = "/onvif/media_service"
    else:
        path = "/onvif/media_service"
    return urlunparse(
        (parsed.scheme or "http", parsed.netloc, path, parsed.params, parsed.query, parsed.fragment)
    )


def _wsse_header(username: str, password: str, *, digest: bool = True) -> str:
    if not username:
        return ""
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    nonce_raw = os.urandom(16)
    nonce_b64 = base64.b64encode(nonce_raw).decode("ascii")
    if digest:
        token = hashlib.sha1(nonce_raw + created.encode("utf-8") + password.encode("utf-8")).digest()
        password_el = (
            '<wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/'
            f'oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{base64.b64encode(token).decode("ascii")}</wsse:Password>'
        )
    else:
        password_el = (
            '<wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/'
            f'oasis-200401-wss-username-token-profile-1.0#PasswordText">{xml_escape(password)}</wsse:Password>'
        )
    return (
        "<wsse:Security s:mustUnderstand=\"1\">"
        "<wsse:UsernameToken>"
        f"<wsse:Username>{xml_escape(username)}</wsse:Username>"
        f"{password_el}"
        f'<wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/'
        f'oasis-200401-wss-soap-message-security-1.0#Base64Binary">{nonce_b64}</wsse:Nonce>'
        f"<wsu:Created>{created}</wsu:Created>"
        "</wsse:UsernameToken>"
        "</wsse:Security>"
    )


def _soap_envelope(body: str, security: str = "") -> str:
    header = f"<s:Header>{security}</s:Header>" if security else "<s:Header/>"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"'
        f' xmlns:tds="{_DEVICE_NS}"'
        f' xmlns:trt="{_MEDIA_NS}"'
        f' xmlns:tt="{_SCHEMA_NS}"'
        ' xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"'
        ' xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">'
        f"{header}<s:Body>{body}</s:Body></s:Envelope>"
    )


def _ptz_unavailable() -> dict[str, Any]:
    return {"ok": False, "error": "PTZ not available"}


class ONVIFAdapter(BaseCameraAdapter):
    """Resolve ONVIF profiles to RTSP, then open the AI URL via gateway/RTSP."""

    def __init__(
        self,
        xaddr: str = "",
        username: str = "",
        password: str = "",
        *,
        gateway: Any = None,
        client: Any = None,
        stream_id: str | None = None,
        source: Any = None,
        main_source: Any = None,
        xaddrs: Any = None,
    ):
        self.xaddr = onvif_xaddr(xaddr or source, xaddrs)
        self.username = str(username or "")
        self.password = str(password or "")
        self._gateway = gateway
        self._client = client if client is not None else getattr(gateway, "client", None)
        self._stream_id = stream_id
        self.error: Optional[str] = None
        self.profiles: list[MediaProfile] = []
        self.source: str = str(source or "")
        self.main_source: str = str(main_source or "")
        self.media_xaddr: str = ""
        self.manufacturer: str = ""
        self.model: str = ""
        self._inner: BaseCameraAdapter | None = None
        self._session: requests.Session | None = None
        self._http_auth: Any = None
        self._use_digest_token = True

    def _session_obj(self) -> requests.Session:
        if self._session is None:
            session = requests.Session()
            session.trust_env = False
            session.headers.update({"User-Agent": "InboundSurveillance/1.0"})
            self._session = session
        return self._session

    def _post_soap(self, url: str, action: str, body: str) -> str:
        session = self._session_obj()
        security = _wsse_header(self.username, self.password, digest=self._use_digest_token)
        envelope = _soap_envelope(body, security)
        headers = {
            "Content-Type": f'application/soap+xml; charset=utf-8; action="{action}"',
            "SOAPAction": f'"{action}"',
        }
        auths: list[Any] = [self._http_auth]
        if self.username:
            if not isinstance(self._http_auth, HTTPDigestAuth):
                auths.append(HTTPDigestAuth(self.username, self.password))
            if not isinstance(self._http_auth, HTTPBasicAuth):
                auths.append(HTTPBasicAuth(self.username, self.password))
        last_err = f"ONVIF request failed for '{redact_source(url)}'."
        seen: set[int] = set()
        for auth in auths:
            key = id(auth) if auth is not None else 0
            if key in seen:
                continue
            seen.add(key)
            try:
                resp = session.post(
                    url,
                    data=envelope.encode("utf-8"),
                    headers=headers,
                    auth=auth,
                    timeout=SOAP_TIMEOUT,
                )
            except requests.RequestException as exc:
                last_err = f"Could not reach ONVIF '{redact_source(url)}': {exc}"
                continue
            if resp.status_code == 401:
                last_err = f"ONVIF login failed for '{redact_source(url)}'."
                continue
            if resp.status_code >= 400:
                last_err = f"ONVIF HTTP {resp.status_code} from '{redact_source(url)}'."
                # Retry with PasswordText token once.
                if self._use_digest_token and self.username:
                    self._use_digest_token = False
                    try:
                        return self._post_soap(url, action, body)
                    except Exception:
                        continue
                continue
            self._http_auth = auth
            return resp.text or ""
        raise RuntimeError(last_err)

    def authenticate(self) -> bool:
        """WS-UsernameToken GetDeviceInformation. HTTP Digest if required."""
        if not self.xaddr:
            self.error = "ONVIF device address is missing."
            return False
        try:
            xml_text = self._post_soap(
                self.xaddr,
                f"{_DEVICE_NS}/GetDeviceInformation",
                "<tds:GetDeviceInformation/>",
            )
        except Exception as exc:
            self.error = str(exc)
            return False
        manufacturer, model = parse_device_info_xml(xml_text)
        self.manufacturer = manufacturer
        self.model = model
        self.error = None
        return True

    def _resolve_media_xaddr(self) -> str:
        if self.media_xaddr:
            return self.media_xaddr
        try:
            xml_text = self._post_soap(
                self.xaddr,
                f"{_DEVICE_NS}/GetCapabilities",
                "<tds:GetCapabilities><tds:Category>Media</tds:Category></tds:GetCapabilities>",
            )
            found = parse_capabilities_media_xaddr(xml_text)
        except Exception:
            found = ""
        self.media_xaddr = found or _media_fallback_xaddr(self.xaddr)
        return self.media_xaddr

    def get_profiles(self) -> list[MediaProfile]:
        media = self._resolve_media_xaddr()
        xml_text = self._post_soap(
            media,
            f"{_MEDIA_NS}/GetProfiles",
            "<trt:GetProfiles/>",
        )
        self.profiles = parse_profiles_xml(xml_text)
        return self.profiles

    def get_stream_uri(self, token: str) -> str:
        media = self._resolve_media_xaddr()
        body = (
            "<trt:GetStreamUri>"
            "<trt:StreamSetup>"
            "<tt:Stream>RTP-Unicast</tt:Stream>"
            "<tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport>"
            "</trt:StreamSetup>"
            f"<trt:ProfileToken>{xml_escape(token)}</trt:ProfileToken>"
            "</trt:GetStreamUri>"
        )
        xml_text = self._post_soap(media, f"{_MEDIA_NS}/GetStreamUri", body)
        uri = parse_stream_uri_xml(xml_text)
        return inject_url_auth(uri, self.username, self.password)

    def resolve(self) -> bool:
        """Authenticate, list profiles, pick sub/main RTSP URIs. No grabber open."""
        try:
            if str(self.source or "").lower().startswith("rtsp://"):
                if not self.main_source:
                    self.main_source = self.source
                return True
            if not self.authenticate():
                return False
            profiles = self.get_profiles()
            sub, main = pick_sub_and_main(profiles)
            if sub is None:
                self.error = f"No ONVIF media profiles at '{redact_source(self.xaddr)}'."
                return False
            try:
                self.source = self.get_stream_uri(sub.token)
            except Exception as exc:
                self.error = str(exc)
                return False
            if not self.source:
                self.error = f"ONVIF GetStreamUri returned empty for '{sub.token}'."
                return False
            if main is not None and main.token != sub.token:
                try:
                    self.main_source = self.get_stream_uri(main.token) or self.source
                except Exception:
                    self.main_source = self.source
            else:
                self.main_source = self.source
            self.error = None
            return True
        except Exception as exc:
            self.error = str(exc)
            return False

    def _open_inner(self) -> bool:
        from adapters import create_adapter, create_direct_adapter

        url = str(self.source or "")
        if not url:
            self.error = self.error or "ONVIF AI stream URL is missing."
            return False
        gateway = self._gateway
        ready = False
        if gateway is not None:
            try:
                ready = bool(gateway.is_ready())
            except Exception:
                ready = False
        try:
            if ready:
                inner = create_adapter(
                    url,
                    gateway=gateway,
                    client=self._client,
                    stream_id=self._stream_id,
                    protocol="rtsp",
                )
            else:
                inner = create_direct_adapter(url, protocol="rtsp")
            ok = bool(inner.connect())
        except Exception as exc:
            self.error = str(exc)
            return False
        if not ok:
            self.error = getattr(inner, "error", None) or self.error or "ONVIF RTSP connect failed."
            safe_release(inner)
            return False
        self._inner = inner
        self.error = None
        return True

    def connect(self) -> bool:
        self.release()
        if not self.resolve():
            return False
        return self._open_inner()

    def read_frame(self) -> Optional[FramePacket]:
        if self._inner is None:
            return None
        return self._inner.read_frame()

    def release(self) -> None:
        inner = self._inner
        self._inner = None
        safe_release(inner)
        session = self._session
        self._session = None
        if session is not None:
            try:
                session.close()
            except Exception:
                pass

    def is_connected(self) -> bool:
        return self._inner is not None and bool(self._inner.is_connected())

    def continuous_move(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return _ptz_unavailable()

    def stop(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return _ptz_unavailable()

    def preset(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return _ptz_unavailable()

    def __repr__(self) -> str:
        return f"ONVIFAdapter(xaddr={redact_source(self.xaddr)!r})"
