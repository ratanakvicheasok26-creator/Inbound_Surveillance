"""REST client for the go2rtc API on localhost."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import requests


class Go2RtcClient:
    """Talk to ``http://127.0.0.1:1984`` (or a custom base URL).

    go2rtc's documented create verb is ``PUT /api/streams``. Older notes and
    some wrappers use ``POST``; this client tries PUT, then POST, then PATCH
    so registration works across 1.9.4+.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:1984", timeout: float = 3.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.trust_env = False
        self._session.headers.update({"User-Agent": "InboundSurveillance/1.0"})

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass

    def api_url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def ws_url(self, stream_id: str) -> str:
        http = self.base_url
        if http.startswith("https://"):
            ws = "wss://" + http[len("https://") :]
        elif http.startswith("http://"):
            ws = "ws://" + http[len("http://") :]
        else:
            ws = "ws://" + http
        return f"{ws}/api/ws?src={quote(stream_id, safe='')}"

    def mjpeg_url(self, stream_id: str) -> str:
        return f"{self.base_url}/api/stream.mjpeg?src={quote(stream_id, safe='')}"

    def is_ready(self) -> bool:
        try:
            resp = self._session.get(self.api_url("/api/streams"), timeout=self.timeout)
            return resp.status_code < 500
        except requests.RequestException:
            return False

    def register_stream(self, stream_id: str, source_url: str, *, replace: bool = False) -> bool:
        """Bind ``source_url`` as a named stream. Returns True on success.

        Re-PUT/PATCH of a live stream restarts the go2rtc producer and drops
        every OpenCV consumer — that freezes every grid tile except the
        laptop webcam (which never uses the gateway). Leave an existing
        producer in place unless ``replace`` is set or the URL clearly changed.
        """
        if not stream_id or not source_url:
            return False
        params = {"src": source_url, "name": stream_id}
        existing = self.get_stream_info(stream_id)
        if existing and not replace:
            if source_url in str(existing):
                return True
            producers = existing.get("producers") or []
            if isinstance(producers, list) and producers:
                return True
            return True
        methods = ("PATCH", "PUT", "POST") if existing else ("PUT", "POST", "PATCH")
        for method in methods:
            try:
                resp = self._session.request(
                    method,
                    self.api_url("/api/streams"),
                    params=params,
                    timeout=self.timeout,
                )
            except requests.RequestException:
                continue
            if resp.status_code < 400:
                return True
        if existing:
            # Stream is already on the gateway; leave it in place.
            return True
        return False

    def remove_stream(self, stream_id: str) -> bool:
        """Unregister a stream by name. True if gone (including already absent)."""
        if not stream_id:
            return False
        last_status = 599
        for params in ({"src": stream_id}, {"name": stream_id}):
            try:
                resp = self._session.delete(
                    self.api_url("/api/streams"),
                    params=params,
                    timeout=self.timeout,
                )
            except requests.RequestException:
                continue
            last_status = resp.status_code
            if resp.status_code < 400 or resp.status_code == 404:
                return True
        return last_status == 404

    def get_stream_info(self, stream_id: str | None = None) -> dict[str, Any]:
        """Return one stream dict, or all streams when ``stream_id`` is None."""
        try:
            resp = self._session.get(self.api_url("/api/streams"), timeout=self.timeout)
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError):
            return {}
        if not isinstance(payload, dict):
            return {}
        if stream_id is None:
            return payload
        info = payload.get(stream_id)
        return info if isinstance(info, dict) else {}

    def snapshot_jpeg(self, stream_id: str, timeout: float | None = None) -> bytes | None:
        """``GET /api/frame.jpeg?src=`` — one still, no extra continuous pull."""
        if not stream_id:
            return None
        try:
            resp = self._session.get(
                self.api_url("/api/frame.jpeg"),
                params={"src": stream_id},
                timeout=timeout if timeout is not None else self.timeout,
            )
        except requests.RequestException:
            return None
        if resp.status_code == 200 and resp.content[:2] == b"\xff\xd8":
            return resp.content
        return None

    def consumer_counts(self, stream_id: str) -> tuple[int, int]:
        info = self.get_stream_info(stream_id)
        producers = info.get("producers") or []
        consumers = info.get("consumers") or []
        return (
            len(producers) if isinstance(producers, list) else 0,
            len(consumers) if isinstance(consumers, list) else 0,
        )
