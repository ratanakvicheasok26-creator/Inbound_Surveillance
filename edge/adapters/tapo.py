"""TP-Link Tapo cameras via go2rtc ``tapo://`` (no KLAP crypto in Python)."""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import quote, urlparse

from adapters.base import BaseCameraAdapter, FramePacket, redact_source, safe_release
from adapters.rtsp import RTSPAdapter
from media.go2rtc import sanitize_stream_id

_TAPO_RTSP_PATHS = ("/stream2", "/stream1")


def _split_auth(url: str) -> tuple[str, str, str]:
    parsed = urlparse(url)
    user = parsed.username or ""
    password = parsed.password or ""
    host = parsed.hostname or ""
    port = parsed.port
    netloc = f"{host}:{port}" if port else host
    path = parsed.path or ""
    clean = f"{parsed.scheme}://{netloc}{path}"
    if parsed.query:
        clean += f"?{parsed.query}"
    return clean, user, password


def tapo_source_url(source: Any, username: str = "", password: str = "") -> str:
    text = str(source or "").strip()
    lower = text.lower()
    if lower.startswith("tapo://"):
        return text
    host = text
    user = username
    password = password
    if "://" in text:
        parsed = urlparse(text)
        host = parsed.hostname or text
        user = user or (parsed.username or "")
        password = password or (parsed.password or "")
    if not host:
        return text
    if user:
        return f"tapo://{quote(user, safe='')}:{quote(password, safe='')}@{host}"
    return f"tapo://{host}"


def tapo_host(source: Any) -> str:
    text = str(source or "").strip()
    if "://" in text:
        return urlparse(text).hostname or ""
    return text.split("/")[0].split(":")[0]


class TapoAdapter(BaseCameraAdapter):
    """Register ``tapo://`` with go2rtc, then read the local RTSP restream.

    If that handshake fails, probe common Tapo RTSP paths. Never crashes —
    failures set ``error`` and return False.
    """

    def __init__(
        self,
        source: Any,
        username: str = "",
        password: str = "",
        *,
        gateway: Any = None,
        client: Any = None,
        stream_id: str | None = None,
        main_source: Any = None,
    ):
        self.raw_source = source
        self.username = str(username or "")
        self.password = str(password or "")
        if not self.username:
            _, user, password = _split_auth(str(source or ""))
            self.username = user
            self.password = password or self.password
        self.tapo_url = tapo_source_url(source, self.username, self.password)
        self.source = str(source or "")
        self.main_source = str(main_source or "")
        self._gateway = gateway
        self._client = client if client is not None else getattr(gateway, "client", None)
        self.stream_id = sanitize_stream_id(stream_id or "live")
        self.error: Optional[str] = None
        self._inner: BaseCameraAdapter | None = None
        self.via = "tapo"

    def _local_rtsp(self) -> str:
        gateway = self._gateway
        if gateway is not None and hasattr(gateway, "rtsp_url"):
            try:
                return str(gateway.rtsp_url(self.stream_id))
            except Exception:
                pass
        return f"rtsp://127.0.0.1:8554/{self.stream_id}"

    def _register(self, url: str) -> bool:
        if self._client is None:
            return False
        try:
            return bool(self._client.register_stream(self.stream_id, url))
        except Exception as exc:
            self.error = f"go2rtc register failed: {exc}"
            return False

    def _open_local(self) -> bool:
        inner = RTSPAdapter(self._local_rtsp())
        try:
            ok = bool(inner.connect())
        except Exception as exc:
            ok = False
            inner.error = str(exc)
        if ok:
            self._inner = inner
            self.error = None
            return True
        self.error = inner.error or self.error
        safe_release(inner)
        return False

    def _connect_tapo_gateway(self) -> bool:
        if self._client is None:
            return False
        if not self._register(self.tapo_url):
            return False
        if self._open_local():
            self.via = "tapo"
            self.source = self.tapo_url
            return True
        return False

    def _rtsp_candidates(self) -> list[str]:
        host = tapo_host(self.raw_source) or tapo_host(self.tapo_url)
        if not host:
            return []
        user = quote(self.username, safe="") if self.username else ""
        pw = quote(self.password, safe="") if self.username else ""
        auth = f"{user}:{pw}@" if user else ""
        out: list[str] = []
        text = str(self.raw_source or "")
        if text.lower().startswith("rtsp://"):
            out.append(text)
        for path in _TAPO_RTSP_PATHS:
            out.append(f"rtsp://{auth}{host}:554{path}")
        return out

    def _connect_rtsp_fallback(self) -> bool:
        last_err = self.error
        candidates = self._rtsp_candidates()
        if not candidates:
            self.error = last_err or f"Could not resolve Tapo host from '{redact_source(self.raw_source)}'."
            return False
        main = candidates[-1]
        for url in candidates:
            adapter = RTSPAdapter(url)
            try:
                ok = bool(adapter.connect())
            except Exception as exc:
                ok = False
                adapter.error = str(exc)
            if not ok:
                last_err = adapter.error or last_err
                safe_release(adapter)
                continue
            self._inner = adapter
            self.source = url
            self.main_source = main if url != main else url
            if url.endswith("/stream2"):
                self.main_source = url[:-8] + "/stream1"
            self.via = "rtsp"
            self.error = None
            if self._client is not None:
                try:
                    self._client.register_stream(self.stream_id, url)
                except Exception:
                    pass
            return True
        self.error = last_err or f"Tapo RTSP fallback failed for '{redact_source(self.raw_source)}'."
        return False

    def connect(self) -> bool:
        self.release()
        try:
            if self._connect_tapo_gateway():
                return True
            return self._connect_rtsp_fallback()
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
        return f"TapoAdapter(src={redact_source(self.tapo_url)!r}, via={self.via})"
