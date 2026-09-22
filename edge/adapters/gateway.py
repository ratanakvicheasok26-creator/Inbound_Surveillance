"""Local go2rtc restream consumer for RTSP and phone HTTP cameras."""

from __future__ import annotations

import time
from typing import Any, Optional

from adapters.base import (
    BaseCameraAdapter,
    FramePacket,
    protocol_from_source,
    redact_source,
)
from adapters.phone_http import PhoneHttpAdapter
from adapters.rtsp import RTSPAdapter
from media.go2rtc import sanitize_stream_id


class GatewayAdapter(BaseCameraAdapter):
    """Register an upstream URL with go2rtc, then read the local restream.

    Python YOLO and the desktop UI both attach to go2rtc, so the camera is
    fetched once. If the gateway is down or the local restream never produces
    frames, the original RTSP / HTTP adapter is used instead.
    """

    def __init__(
        self,
        source_url: str,
        stream_id: str,
        local_url: str,
        client: Any | None = None,
        *,
        fallback: BaseCameraAdapter | None = None,
    ):
        self.source_url = str(source_url)
        self.stream_id = sanitize_stream_id(stream_id)
        self.local_url = str(local_url)
        self._client = client
        self._fallback_adapter = fallback
        self.error: Optional[str] = None
        self._inner: BaseCameraAdapter | None = None
        self.via = "gateway"

    def _make_local(self) -> BaseCameraAdapter:
        if protocol_from_source(self.local_url) == "phone":
            return PhoneHttpAdapter(self.local_url)
        return RTSPAdapter(self.local_url)

    def _register(self) -> bool:
        if self._client is None:
            return True
        try:
            return bool(self._client.register_stream(self.stream_id, self.source_url))
        except Exception as exc:
            self.error = f"go2rtc register failed: {exc}"
            return False

    def connect(self) -> bool:
        self.release(unregister=False)
        if not self._register():
            return self._connect_fallback()
        local = self._make_local()
        try:
            ok = bool(local.connect())
        except Exception as exc:
            ok = False
            local.error = str(exc)
        if ok:
            self._inner = local
            self.via = "gateway"
            self.error = None
            return True
        self.error = local.error
        try:
            local.release()
        except Exception:
            pass
        return self._connect_fallback()

    def _connect_fallback(self) -> bool:
        adapter = self._fallback_adapter
        self._fallback_adapter = None
        if adapter is None:
            kind = protocol_from_source(self.source_url)
            if kind == "phone":
                adapter = PhoneHttpAdapter(self.source_url)
            else:
                adapter = RTSPAdapter(self.source_url)
        try:
            ok = bool(adapter.connect())
        except Exception as exc:
            ok = False
            adapter.error = str(exc)
        if not ok:
            self.error = adapter.error or self.error or "Gateway and direct connect failed."
            try:
                adapter.release()
            except Exception:
                pass
            return False
        self._inner = adapter
        self.via = "direct"
        self.error = None
        return True

    def read_frame(self) -> Optional[FramePacket]:
        now = time.time()
        if self._inner is None or not self._inner.is_connected():
            if (now - getattr(self, "_last_gw_reconnect", 0.0)) >= 2.5:
                self._last_gw_reconnect = now
                self.connect()
            if self._inner is None:
                return None
        packet = self._inner.read_frame()
        if packet is None and self.via == "gateway":
            if (now - getattr(self, "_last_gw_fallback", 0.0)) >= 3.0:
                self._last_gw_fallback = now
                self._connect_fallback()
                if self._inner is not None:
                    packet = self._inner.read_frame()
        return packet

    def release(self, unregister: bool = False) -> None:
        """Close the local consumer. Stream registration is owned by the engine."""
        inner = self._inner
        self._inner = None
        if inner is not None:
            try:
                inner.release()
            except Exception:
                pass
        if unregister and self._client is not None:
            try:
                self._client.remove_stream(self.stream_id)
            except Exception:
                pass

    def is_connected(self) -> bool:
        return self._inner is not None and bool(self._inner.is_connected())

    def __repr__(self) -> str:
        return (
            f"GatewayAdapter(id={self.stream_id!r}, "
            f"src={redact_source(self.source_url)!r}, via={self.via})"
        )
