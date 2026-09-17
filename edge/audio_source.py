"""Audio Source Abstraction for Inbound Surveillance.

Provides a unified interface for audio capture from:
1. Laptop Microphone (via python-sounddevice)
2. USB Microphone (pluggable)
3. Camera / RTSP Microphone (pluggable)

All sources output standardized 16kHz, 16-bit mono PCM float32/int16 chunks.
"""

from __future__ import annotations

import abc
import logging
import queue
import subprocess
import threading
import time
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("audio_source")


class AudioSource(abc.ABC):
    """Abstract base class for all audio capture sources."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_running = False

    @abc.abstractmethod
    def start(self, callback: Callable[[np.ndarray, float], None]) -> bool:
        """Start capturing audio.
        
        Args:
            callback: Function called with (audio_chunk_float32, timestamp)
        """
        pass

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop capturing audio."""
        pass


class LaptopMicrophoneSource(AudioSource):
    """Captures audio from the laptop built-in microphone using sounddevice with arecord fallback."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 512,  # 32ms at 16kHz (optimal for Silero VAD)
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(sample_rate=sample_rate, channels=channels)
        self.chunk_size = chunk_size
        self.device_index = device_index
        self._stream = None
        self._proc: Optional[subprocess.Popen] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[np.ndarray, float], None]] = None
        self._lock = threading.Lock()

    def start(self, callback: Callable[[np.ndarray, float], None]) -> bool:
        with self._lock:
            if self.is_running:
                return True

            self._callback = callback

            # 1. Try python-sounddevice (if PortAudio is installed)
            started_sd = self._start_sounddevice()
            if started_sd:
                return True

            # 2. Fallback to native Linux arecord / ALSA stream
            started_arecord = self._start_arecord()
            if started_arecord:
                return True

            logger.error("[AudioSource] Could not start audio capture with sounddevice or arecord.")
            return False

    def _start_sounddevice(self) -> bool:
        try:
            import sounddevice as sd

            def _audio_callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"[AudioSource] sounddevice status: {status}")
                if self._callback is not None:
                    if indata.ndim > 1:
                        mono = indata[:, 0].copy()
                    else:
                        mono = indata.copy()
                    now = time.time()
                    try:
                        self._callback(mono, now)
                    except Exception as e:
                        logger.error(f"[AudioSource] Callback error: {e}")

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.chunk_size,
                device=self.device_index,
                callback=_audio_callback,
            )
            self._stream.start()
            self.is_running = True
            logger.info(f"[AudioSource] Laptop mic stream started via sounddevice at {self.sample_rate}Hz (chunk: {self.chunk_size})")
            return True
        except Exception as exc:
            logger.info(f"[AudioSource] sounddevice unavailable ({exc}), attempting arecord fallback...")
            self._stream = None
            return False

    def _start_arecord(self) -> bool:
        import shutil
        if not shutil.which("arecord"):
            logger.error("[AudioSource] 'arecord' binary not found on system.")
            return False

        try:
            device_arg = "default"
            if self.device_index is not None:
                # plug converts sample rate/channels (hardware may only
                # support stereo at 44.1kHz; VAD needs 16kHz mono).
                device_arg = f"plughw:{self.device_index},0"

            cmd = [
                "arecord",
                "-D", device_arg,
                "-f", "S16_LE",
                "-r", str(self.sample_rate),
                "-c", str(self.channels),
                "-t", "raw",
                "-q",
            ]
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.is_running = True
            self._worker_thread = threading.Thread(target=self._arecord_reader_loop, daemon=True)
            self._worker_thread.start()
            logger.info(f"[AudioSource] Laptop mic stream started via arecord ({device_arg}) at {self.sample_rate}Hz")
            return True
        except Exception as exc:
            logger.error(f"[AudioSource] Failed to start arecord stream: {exc}")
            self.is_running = False
            self._proc = None
            return False

    def _arecord_reader_loop(self) -> None:
        bytes_per_sample = 2  # 16-bit
        bytes_to_read = self.chunk_size * self.channels * bytes_per_sample

        while self.is_running and self._proc is not None and self._proc.stdout is not None:
            try:
                raw_bytes = self._proc.stdout.read(bytes_to_read)
                if not raw_bytes or len(raw_bytes) < bytes_to_read:
                    if not self.is_running:
                        break
                    time.sleep(0.01)
                    continue

                mono_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
                mono_float32 = mono_int16.astype(np.float32) / 32768.0

                now = time.time()
                if self._callback is not None:
                    try:
                        self._callback(mono_float32, now)
                    except Exception as cb_err:
                        logger.error(f"[AudioSource] arecord callback error: {cb_err}")

            except Exception as read_err:
                if self.is_running:
                    logger.error(f"[AudioSource] arecord read error: {read_err}")
                break

    def stop(self) -> None:
        with self._lock:
            if not self.is_running:
                return
            self.is_running = False

            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.warning(f"[AudioSource] Error closing sounddevice stream: {e}")
                self._stream = None

            if self._proc is not None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=1.0)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
                self._proc = None

            if self._worker_thread is not None and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=1.0)
                self._worker_thread = None

            self._callback = None
            logger.info("[AudioSource] Laptop microphone stream stopped.")


class USBMicrophoneSource(LaptopMicrophoneSource):
    """Captures audio from an external USB microphone."""
    pass


class CameraMicrophoneSource(AudioSource):
    """Captures audio from an IP camera RTSP/media audio stream."""

    def __init__(self, rtsp_url: str, sample_rate: int = 16000) -> None:
        super().__init__(sample_rate=sample_rate, channels=1)
        self.rtsp_url = rtsp_url

    def start(self, callback: Callable[[np.ndarray, float], None]) -> bool:
        # Pluggable camera audio extraction (e.g. ffmpeg pipe / RTSP backchannel)
        logger.info(f"[CameraAudioSource] Camera microphone configured for {self.rtsp_url}")
        return False

    def stop(self) -> None:
        self.is_running = False


class VideoFileAudioSource(AudioSource):
    """Extracts and streams audio from a video file in real-time pace with seamless looping."""

    def __init__(
        self,
        video_path: str,
        sample_rate: int = 16000,
        chunk_size: int = 512,  # 32ms at 16kHz
        loop: bool = False,
    ) -> None:
        super().__init__(sample_rate=sample_rate, channels=1)
        self.video_path = str(video_path)
        self.chunk_size = chunk_size
        self.loop = loop
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._callback: Optional[Callable[[np.ndarray, float], None]] = None
        self._lock = threading.Lock()

    def start(self, callback: Callable[[np.ndarray, float], None]) -> bool:
        with self._lock:
            if self.is_running:
                return True
            self._callback = callback
            self._stop_event.clear()
            self.is_running = True
            self._worker_thread = threading.Thread(target=self._stream_loop, daemon=True)
            self._worker_thread.start()
            logger.info(f"[VideoFileAudioSource] Started streaming audio from '{self.video_path}'")
            return True

    def _stream_loop(self) -> None:
        from pathlib import Path
        try:
            import av
        except ImportError:
            logger.error("[VideoFileAudioSource] 'av' (PyAV) is required for video audio decoding.")
            return

        while self.is_running and not self._stop_event.is_set():
            try:
                p = Path(self.video_path)
                if not p.is_file():
                    try:
                        from adapters.video_file import resolve_video_path
                        p = resolve_video_path(self.video_path)
                    except Exception:
                        pass

                if not p.is_file():
                    logger.warning(f"[VideoFileAudioSource] Video file not found: {self.video_path}")
                    time.sleep(2.0)
                    if not self.loop:
                        break
                    continue

                container = av.open(str(p))
                if len(container.streams.audio) == 0:
                    logger.warning(f"[VideoFileAudioSource] No audio stream found in {p}")
                    time.sleep(2.0)
                    if not self.loop:
                        break
                    continue

                resampler = av.AudioResampler(format="fltp", layout="mono", rate=self.sample_rate)
                pcm_buf = []
                for frame in container.decode(audio=0):
                    if self._stop_event.is_set():
                        break
                    for rframe in resampler.resample(frame):
                        arr = rframe.to_ndarray()[0]
                        pcm_buf.append(arr)
                for rframe in resampler.resample(None):
                    arr = rframe.to_ndarray()[0]
                    pcm_buf.append(arr)

                container.close()

                if not pcm_buf or self._stop_event.is_set():
                    time.sleep(1.0)
                    if not self.loop:
                        break
                    continue

                full_pcm = np.concatenate(pcm_buf)
                total_samples = len(full_pcm)
                chunk_duration = self.chunk_size / self.sample_rate

                idx = 0
                next_time = time.time()
                while idx < total_samples and not self._stop_event.is_set() and self.is_running:
                    chunk = full_pcm[idx : idx + self.chunk_size]
                    if len(chunk) < self.chunk_size:
                        chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))

                    now = time.time()
                    if self._callback is not None:
                        try:
                            self._callback(chunk, now)
                        except Exception as cb_err:
                            logger.error(f"[VideoFileAudioSource] Callback error: {cb_err}")

                    idx += self.chunk_size
                    next_time += chunk_duration
                    sleep_time = next_time - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        next_time = time.time()

                if not self.loop:
                    break

            except Exception as e:
                logger.error(f"[VideoFileAudioSource] Error decoding audio from {self.video_path}: {e}")
                time.sleep(2.0)

    def stop(self) -> None:
        with self._lock:
            if not self.is_running:
                return
            self.is_running = False
            self._stop_event.set()
            if self._worker_thread is not None and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=1.0)
                self._worker_thread = None
            self._callback = None
            logger.info("[VideoFileAudioSource] Video audio stream stopped.")

