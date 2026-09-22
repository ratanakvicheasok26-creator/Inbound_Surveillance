"""Modular camera adapters for webcam, RTSP, phone, ONVIF, Tapo, and WebRTC."""

from __future__ import annotations

from typing import Any

from adapters.base import (
    BaseCameraAdapter,
    FramePacket,
    parse_source,
    protocol_from_source,
    redact_source,
    unwrap_local_video_source,
    webcam_index,
)
from adapters.gateway import GatewayAdapter
from adapters.onvif import ONVIFAdapter
from adapters.phone_http import PhoneHttpAdapter
from adapters.rtsp import RTSPAdapter
from adapters.tapo import TapoAdapter
from adapters.webcam import WebcamAdapter, open_webcam_index
from adapters.webrtc import WebRTCAdapter
from media.go2rtc import sanitize_stream_id

__all__ = [
    "BaseCameraAdapter",
    "FramePacket",
    "GatewayAdapter",
    "ONVIFAdapter",
    "PhoneHttpAdapter",
    "RTSPAdapter",
    "TapoAdapter",
    "WebRTCAdapter",
    "WebcamAdapter",
    "create_adapter",
    "create_direct_adapter",
    "ingest_kind",
    "open_webcam_index",
    "parse_source",
    "protocol_from_source",
    "redact_source",
    "unwrap_local_video_source",
    "webcam_index",
]


def ingest_kind(source: Any, protocol: str | None = None) -> str:
    """Adapter family used to open ``source`` (URL scheme wins over metadata)."""
    parsed = parse_source(source)
    if isinstance(parsed, int) or str(parsed).strip().isdigit():
        return "webcam"
    text = str(parsed).strip().lower()
    kind = str(protocol or "").strip().lower()
    if kind == "webcam":
        return "webcam"
    if kind in ("video", "file") or unwrap_local_video_source(parsed) is not None:
        return "video"
    if text.startswith("rtsp://"):
        return "rtsp"
    if text.startswith("tapo://") or kind == "tapo":
        return "tapo"
    if text.startswith(("webrtc://", "whep://", "whip://")) or kind == "webrtc":
        return "webrtc"
    if text.startswith("onvif://") or kind == "onvif" or "/onvif" in text:
        return "onvif"
    if text.startswith("http://") or text.startswith("https://"):
        return "phone"
    if kind:
        return kind
    return protocol_from_source(parsed)


def create_direct_adapter(
    source: Any,
    *,
    protocol: str | None = None,
    username: str = "",
    password: str = "",
    xaddrs: Any = None,
    main_source: Any = None,
    gateway: Any = None,
    client: Any = None,
    stream_id: str | None = None,
) -> BaseCameraAdapter:
    """Build a webcam / RTSP / phone / vendor adapter that talks to the camera itself."""
    parsed = parse_source(source)
    kind = ingest_kind(parsed, protocol)
    if kind == "webcam":
        return WebcamAdapter(webcam_index(parsed))
    if kind == "video":
        from adapters.video_file import VideoFileAdapter

        return VideoFileAdapter(unwrap_local_video_source(parsed) or str(parsed))
    if kind == "onvif":
        return ONVIFAdapter(
            str(parsed),
            username=username,
            password=password,
            xaddrs=xaddrs,
            source=parsed,
            main_source=main_source,
            gateway=gateway,
            client=client,
            stream_id=stream_id,
        )
    if kind == "tapo":
        return TapoAdapter(
            parsed,
            username=username,
            password=password,
            gateway=gateway,
            client=client,
            stream_id=stream_id,
            main_source=main_source,
        )
    if kind == "webrtc":
        return WebRTCAdapter(
            parsed,
            gateway=gateway,
            client=client,
            stream_id=stream_id,
        )
    if kind == "phone":
        return PhoneHttpAdapter(str(parsed))
    return RTSPAdapter(str(parsed))


def create_adapter(
    source: Any,
    *,
    gateway: Any = None,
    client: Any = None,
    stream_id: str | None = None,
    protocol: str | None = None,
    username: str = "",
    password: str = "",
    xaddrs: Any = None,
    main_source: Any = None,
) -> BaseCameraAdapter:
    """Build the adapter for a webcam index, RTSP URL, or vendor protocol.

    Network sources go through go2rtc when a running ``gateway`` is supplied
    so YOLO and the UI share one upstream pull. Webcams stay on
    ``WebcamAdapter``. Construction is I/O-free; ``connect()`` runs on the
    grabber thread. Pass ``protocol`` so saved ONVIF/Tapo/WebRTC cameras are
    not inferred from a bare host string.
    """
    parsed = parse_source(source)
    kind = ingest_kind(parsed, protocol)
    if kind == "webcam":
        return WebcamAdapter(webcam_index(parsed))
    if kind == "video":
        return create_direct_adapter(parsed, protocol="video")
    if kind in ("onvif", "tapo", "webrtc"):
        return create_direct_adapter(
            parsed,
            protocol=kind,
            username=username,
            password=password,
            xaddrs=xaddrs,
            main_source=main_source,
            gateway=gateway,
            client=client,
            stream_id=stream_id,
        )
    ready = False
    if gateway is not None:
        try:
            ready = bool(gateway.is_ready())
        except Exception:
            ready = False
    if not ready:
        return create_direct_adapter(parsed, protocol=kind)
    sid = sanitize_stream_id(stream_id or "live")
    local_url = gateway.consumer_url(sid, parsed)
    api = client if client is not None else getattr(gateway, "client", None)
    return GatewayAdapter(
        str(parsed),
        sid,
        local_url,
        client=api,
        fallback=create_direct_adapter(parsed, protocol=kind),
    )
