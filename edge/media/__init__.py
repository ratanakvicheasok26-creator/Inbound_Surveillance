"""Embedded go2rtc media gateway: process manager, REST client, and URLs."""

from __future__ import annotations

from media.client import Go2RtcClient
from media.go2rtc import (
    DEFAULT_API_PORT,
    DEFAULT_RTSP_PORT,
    DEFAULT_WEBRTC_PORT,
    GO2RTC_VERSION,
    Go2RtcManager,
    ensure_binary,
    platform_tag,
    sanitize_stream_id,
)

__all__ = [
    "DEFAULT_API_PORT",
    "DEFAULT_RTSP_PORT",
    "DEFAULT_WEBRTC_PORT",
    "GO2RTC_VERSION",
    "Go2RtcClient",
    "Go2RtcManager",
    "ensure_binary",
    "platform_tag",
    "sanitize_stream_id",
]
