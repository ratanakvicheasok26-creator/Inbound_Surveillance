"""WebRTC / WHEP ingest via go2rtc. Python does not terminate ICE/DTLS."""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlparse

from adapters.base import BaseCameraAdapter, FramePacket, redact_source, safe_release
from adapters.rtsp import RTSPAdapter
from media.go2rtc import sanitize_stream_id


def webrtc_go2rtc_url(source: Any) -> str:
    """Map webrtc:// / whep:// / whip:// to a URL go2rtc can pull."""
    text = str(source or "").strip()
    lower = text.lower()
    if lower.startswith("whep://"):
        rest = text[7:]
        return f"http://{rest}"
    if lower.startswith("whip://"):
        rest = text[7:]
        return f"http://{rest}"
    if lower.startswith("webrtc://"):
        rest = text[9:]
        parsed = urlparse(f"http://{rest}")
        scheme = "https" if parsed.port == 443 else "http"
        return f"webrtc:{scheme}://{rest}"
    return text


class WebRTCAdapter(BaseCameraAdapter):
    """Register a WHEP/WHIP/WebRTC URL with go2rtc and read local RTSP."""

    def __init__(
        self,
        source: Any,
        *,
        gateway: Any = None,
        client: Any = None,
        stream_id: str | None = None,
    ):
        self.source_url = str(source or "")
        self.go2rtc_url = webrtc_go2rtc_url(source)
        self._gateway = gateway
        self._client = client if client is not None else getattr(gateway, "client", None)
        self.stream_id = sanitize_stream_id(stream_id or "live")
        self.error: Optional[str] = None
        self._inner: BaseCameraAdapter | None = None
        self.via = "webrtc"

    def _local_rtsp(self) -> str:
        gateway = self._gateway
        if gateway is not None and hasattr(gateway, "rtsp_url"):
            try:
                return str(gateway.rtsp_url(self.stream_id))
            except Exception:
                pass
        return f"rtsp://127.0.0.1:8554/{self.stream_id}"

    def connect(self) -> bool:
        self.release()
        try:
            if self._client is None:
                self.error = "go2rtc is not running; WebRTC ingest requires the media gateway."
                return False
            try:
                ok = bool(self._client.register_stream(self.stream_id, self.go2rtc_url))
            except Exception as exc:
                self.error = f"go2rtc register failed: {exc}"
                return False
            if not ok:
                self.error = f"go2rtc rejected WebRTC source '{redact_source(self.go2rtc_url)}'."
                return False
            inner = RTSPAdapter(self._local_rtsp())
            try:
                opened = bool(inner.connect())
            except Exception as exc:
                opened = False
                inner.error = str(exc)
            if not opened:
                self.error = inner.error or "WebRTC restream produced no frames."
                safe_release(inner)
                return False
            self._inner = inner
            self.via = "gateway"
            self.error = None
            return True
        except Exception as exc:
            self.error = str(exc)
            return False

    def read_frame(self) -> Optional[FramePacket]:
        if self._inner is None:
            return None
        return self._inner.read_frame()

    def release(self) -> None:
        inner = self._inner
        self._inner = None
        safe_release(inner)

    def is_connected(self) -> bool:
        return self._inner is not None and bool(self._inner.is_connected())

    def __repr__(self) -> str:
        return f"WebRTCAdapter(src={redact_source(self.source_url)!r})"
