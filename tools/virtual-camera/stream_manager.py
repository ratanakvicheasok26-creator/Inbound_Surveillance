"""Process manager for MediaMTX and FFmpeg looping RTSP streams."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("stream_manager")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@dataclass
class StreamChannel:
    channel: str
    video_file: str
    video_path: str
    process: Optional[subprocess.Popen] = None
    started_at: float = field(default_factory=time.time)
    status: str = "stopped"
    error: Optional[str] = None
    pid: Optional[int] = None

    def to_dict(self, external_host: str = "localhost", external_rtsp_port: int = 8556, external_webrtc_port: int = 8889) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "video_file": self.video_file,
            "status": self.status,
            "error": self.error,
            "pid": self.pid,
            "uptime_seconds": int(time.time() - self.started_at) if self.status == "streaming" else 0,
            "rtsp_url": f"rtsp://{external_host}:{external_rtsp_port}/{self.channel}",
            "webrtc_url": f"http://{external_host}:{external_webrtc_port}/{self.channel}",
            "hls_url": f"http://{external_host}:8888/{self.channel}/index.m3u8",
        }


class StreamManager:
    """Supervises MediaMTX daemon and FFmpeg streaming pipelines."""

    def __init__(
        self,
        mediamtx_path: str = "mediamtx",
        mediamtx_config: str = "mediamtx.yml",
        rtsp_internal_port: int = 8554,
        external_rtsp_port: int = 8556,
        external_webrtc_port: int = 8889,
        external_host: str = "localhost",
    ):
        self.mediamtx_path = mediamtx_path
        self.mediamtx_config = mediamtx_config
        self.rtsp_internal_port = rtsp_internal_port
        self.external_rtsp_port = external_rtsp_port
        self.external_webrtc_port = external_webrtc_port
        self.external_host = external_host

        self._mediamtx_proc: Optional[subprocess.Popen] = None
        self._streams: Dict[str, StreamChannel] = {}
        self._lock = threading.Lock()
        self._stop_monitor = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

    def start_mediamtx(self) -> bool:
        """Launch MediaMTX if available and not already running."""
        with self._lock:
            if self._mediamtx_proc is not None and self._mediamtx_proc.poll() is None:
                return True

            executable = shutil.which(self.mediamtx_path) or self.mediamtx_path
            if not Path(executable).exists() and not shutil.which(executable):
                logger.warning(f"MediaMTX binary not found at '{executable}'. Assuming external server or container entrypoint.")
                return False

            cmd = [executable]
            if Path(self.mediamtx_config).exists():
                cmd.append(self.mediamtx_config)

            try:
                logger.info(f"Starting MediaMTX: {' '.join(cmd)}")
                self._mediamtx_proc = subprocess.Popen(
                    cmd,
                    stdout=None,
                    stderr=None,
                )
                time.sleep(1.0)
                if self._mediamtx_proc.poll() is not None:
                    logger.error(f"MediaMTX exited prematurely with code {self._mediamtx_proc.returncode}")
                    return False
                logger.info(f"MediaMTX started (PID {self._mediamtx_proc.pid})")
                return True
            except Exception as ex:
                logger.error(f"Failed to start MediaMTX: {ex}")
                return False

    def stop_mediamtx(self) -> None:
        with self._lock:
            if self._mediamtx_proc is not None:
                logger.info("Stopping MediaMTX...")
                try:
                    self._mediamtx_proc.terminate()
                    self._mediamtx_proc.wait(timeout=2.0)
                except Exception:
                    self._mediamtx_proc.kill()
                self._mediamtx_proc = None

    def start_stream(
        self,
        channel: str,
        video_path: str,
        force_transcode: bool = True,
    ) -> StreamChannel:
        """Start streaming a video file to an RTSP channel in an infinite loop."""
        video_p = Path(video_path).resolve()
        if not video_p.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        channel = channel.strip().replace(" ", "_").lower() or "garage"

        with self._lock:
            # Stop existing stream for this channel if active
            if channel in self._streams:
                self._stop_channel_locked(channel)

            target_rtsp = f"rtsp://127.0.0.1:{self.rtsp_internal_port}/{channel}"

            # Real-time infinite loop FFmpeg command
            # -re reads in real-time speed (normal camera rate)
            # -stream_loop -1 loops indefinitely
            cmd = [
                "ffmpeg",
                "-re",
                "-stream_loop", "-1",
                "-i", str(video_p),
            ]

            if force_transcode:
                # Transcode to guaranteed baseline/main H.264 profile with ultra-fast preset and 1-second keyframe interval
                cmd.extend([
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-tune", "zerolatency",
                    "-pix_fmt", "yuv420p",
                    "-b:v", "2500k",
                    "-maxrate", "3000k",
                    "-bufsize", "5000k",
                    "-g", "30",
                    "-an",  # strip audio for video-only surveillance stability
                ])
            else:
                # Stream copy if format is known clean H.264
                cmd.extend(["-c", "copy"])

            cmd.extend([
                "-f", "rtsp",
                "-rtsp_transport", "tcp",
                target_rtsp,
            ])

            logger.info(f"Launching FFmpeg for channel '{channel}': {' '.join(cmd)}")
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                )
            except Exception as ex:
                logger.error(f"Failed to launch FFmpeg: {ex}")
                stream = StreamChannel(
                    channel=channel,
                    video_file=video_p.name,
                    video_path=str(video_p),
                    status="error",
                    error=str(ex),
                )
                self._streams[channel] = stream
                return stream

            # Give FFmpeg 0.5s to verify it didn't immediately exit
            time.sleep(0.5)
            exit_code = proc.poll()
            if exit_code is not None:
                stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
                err_msg = f"FFmpeg failed with exit code {exit_code}: {stderr[:200]}"
                logger.error(err_msg)
                stream = StreamChannel(
                    channel=channel,
                    video_file=video_p.name,
                    video_path=str(video_p),
                    status="error",
                    error=err_msg,
                )
                self._streams[channel] = stream
                return stream

            stream = StreamChannel(
                channel=channel,
                video_file=video_p.name,
                video_path=str(video_p),
                process=proc,
                started_at=time.time(),
                status="streaming",
                pid=proc.pid,
            )
            self._streams[channel] = stream
            self._ensure_monitor_running()
            logger.info(f"Channel '{channel}' is streaming PID {proc.pid} -> {target_rtsp}")
            return stream

    def _stop_channel_locked(self, channel: str) -> bool:
        stream = self._streams.get(channel)
        if not stream or not stream.process:
            return False

        logger.info(f"Stopping stream for channel '{channel}' (PID {stream.pid})")
        try:
            stream.process.terminate()
            stream.process.wait(timeout=2.0)
        except Exception:
            try:
                stream.process.kill()
            except Exception:
                pass

        stream.status = "stopped"
        stream.process = None
        stream.pid = None
        return True

    def stop_stream(self, channel: str) -> bool:
        """Stop an active streaming channel."""
        with self._lock:
            return self._stop_channel_locked(channel)

    def stop_all_streams(self) -> None:
        """Terminate all active FFmpeg streams."""
        with self._lock:
            for channel in list(self._streams.keys()):
                self._stop_channel_locked(channel)

    def list_streams(self) -> list[dict[str, Any]]:
        """Return list of all configured and active channels."""
        with self._lock:
            out = []
            for stream in self._streams.values():
                # Check liveness
                if stream.process and stream.process.poll() is not None:
                    stream.status = "error" if stream.process.returncode != 0 else "stopped"
                    stream.error = f"Exited with code {stream.process.returncode}"
                    stream.process = None
                    stream.pid = None
                out.append(stream.to_dict(self.external_host, self.external_rtsp_port, self.external_webrtc_port))
            return out

    def get_stream(self, channel: str) -> Optional[dict[str, Any]]:
        with self._lock:
            stream = self._streams.get(channel)
            if not stream:
                return None
            return stream.to_dict(self.external_host, self.external_rtsp_port, self.external_webrtc_port)

    def _ensure_monitor_running(self) -> None:
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_monitor.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="StreamMonitor",
                daemon=True,
            )
            self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        """Periodically check stream health and clean up dead processes."""
        while not self._stop_monitor.is_set():
            time.sleep(2.0)
            with self._lock:
                for stream in list(self._streams.values()):
                    if stream.status == "streaming" and stream.process:
                        ret = stream.process.poll()
                        if ret is not None:
                            logger.warning(f"Stream for channel '{stream.channel}' stopped unexpectedly (code {ret})")
                            stream.status = "stopped" if ret == 0 else "error"
                            stream.error = f"Process terminated with code {ret}"
                            stream.process = None
                            stream.pid = None

    def shutdown(self) -> None:
        self._stop_monitor.set()
        self.stop_all_streams()
        self.stop_mediamtx()
