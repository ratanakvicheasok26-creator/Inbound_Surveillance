"""Customer Complaint Detection & Evidence Service.

Orchestrates:
1. Audio capture from Laptop / USB / Camera microphone.
2. Silero VAD speech detection (filters noise and silence).
3. In-memory audio segment reuse (single capture for both WAV persistence & STT).
4. Khmer faster-whisper STT.
5. Netra-NMT Khmer-to-English translation.
6. Ollama structured complaint evaluation.
7. Camera frame retrieval & annotation at complaint moment.
8. SQLite complaint record persistence.
9. 3-Part Telegram alert dispatch (Voice + Text + Screenshot).
10. Event dispatching to Dashboard.
"""

from __future__ import annotations

import collections
import dataclasses
from datetime import datetime
import hashlib
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional
import uuid

import cv2
import numpy as np

from audio_source import (
    AudioSource,
    CameraMicrophoneSource,
    LaptopMicrophoneSource,
    USBMicrophoneSource,
    VideoFileAudioSource,
)
from complaint_auditor import (
    ComplaintAnalysis,
    OllamaComplaintAuditor,
    _khmer_coverage,
    _latin_coverage,
)
from db import insert_customer_complaint, update_complaint_telegram_status
from paths import data_dir
from speech_pipeline import KhmerSTTService, KhmerTranslationService
from telegram_out import TelegramOut
from vad import SileroVAD, SpeechSegment

logger = logging.getLogger("complaint_service")


def complaints_proof_dirs(root: Path | None = None) -> tuple[Path, Path]:
    """Writable audio/still folders (AppData when frozen, edge/ in source)."""
    base = Path(root or data_dir()) / "proofs" / "complaints"
    audio = base / "audio"
    stills = base / "stills"
    audio.mkdir(parents=True, exist_ok=True)
    stills.mkdir(parents=True, exist_ok=True)
    return audio, stills

TG_SEVERITY_ORDER = ["none", "low", "medium", "high", "critical"]


@dataclasses.dataclass
class ComplaintRecord:
    complaint_id: str
    customer_id: str
    camera_id: str
    audio_source: str
    timestamp: str
    audio_path: str
    screenshot_path: Optional[str]
    khmer_transcript: str
    english_transcript: str
    is_complaint: bool
    category: str
    severity: str
    summary: str
    telegram_sent: bool = False
    telegram_error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "complaint_id": self.complaint_id,
            "customer_id": self.customer_id,
            "camera_id": self.camera_id,
            "audio_source": self.audio_source,
            "timestamp": self.timestamp,
            "audio_path": self.audio_path,
            "audio_url": f"/api/complaints/audio/{Path(self.audio_path).name}" if self.audio_path else None,
            "screenshot_path": self.screenshot_path,
            "screenshot_url": f"/api/complaints/stills/{Path(self.screenshot_path).name}" if self.screenshot_path else None,
            "khmer_transcript": self.khmer_transcript,
            "english_transcript": self.english_transcript,
            "is_complaint": self.is_complaint,
            "category": self.category,
            "severity": self.severity,
            "summary": self.summary,
            "telegram_sent": self.telegram_sent,
            "telegram_error": self.telegram_error,
        }


class ComplaintMonitoringService:
    """Manages audio listening, real-time VAD, STT, translation, complaint classification, and dispatching."""

    def __init__(
        self,
        db_conn=None,
        telegram_out: Optional[TelegramOut] = None,
        get_camera_frame_fn: Optional[Callable[[], Optional[np.ndarray]]] = None,
        audio_source: Optional[AudioSource] = None,
        audio_source_type: str = "laptop",
        video_path: Optional[str] = None,
        video_loop: bool = False,
        device_index: Optional[int] = None,
        whisper_model: str = "sengtha/whisper-base-khmer",
        ollama_model: str = "qwen2.5:3b",
        ollama_host: str = "http://localhost:11434",
        vad_threshold: float = 0.35,
        vad_min_speech_duration_ms: int = 400,
        vad_max_speech_duration_s: float = 30.0,
        dedup_window_seconds: float = 60.0,
        telegram_min_severity: str = "low",
        telegram_alert_enabled: bool = True,
        enabled: bool = True,
        data_root: Optional[Path] = None,
    ) -> None:
        self.db_conn = db_conn
        self.telegram_out = telegram_out
        self.get_camera_frame_fn = get_camera_frame_fn
        self.telegram_alert_enabled = telegram_alert_enabled
        self.enabled = enabled
        self.audio_source_type = audio_source_type
        self.video_path = video_path
        self.video_loop = video_loop
        self.dedup_window_seconds = dedup_window_seconds
        self.telegram_min_severity = (telegram_min_severity or "low").lower()
        self._audio_source_label = str(audio_source_type or "laptop").lower()
        self._audio_dir, self._stills_dir = complaints_proof_dirs(data_root)

        # Select audio source
        if audio_source is not None:
            self.audio_source = audio_source
            self._audio_source_label = type(audio_source).__name__.lower().replace("source", "")
        elif audio_source_type.lower() == "video" and video_path:
            self.audio_source = VideoFileAudioSource(video_path=video_path, sample_rate=16000, loop=video_loop)
        elif audio_source_type.lower() == "usb":
            self.audio_source = USBMicrophoneSource(sample_rate=16000, device_index=device_index)
        elif audio_source_type.lower() == "camera":
            self.audio_source = CameraMicrophoneSource(sample_rate=16000)
        else:
            self.audio_source = LaptopMicrophoneSource(sample_rate=16000, device_index=device_index)

        # Core ML / VAD / STT components
        self.vad = SileroVAD(
            sample_rate=16000,
            threshold=max(0.01, min(vad_threshold, 0.99)),
            min_speech_duration_seconds=max(0.2, vad_min_speech_duration_ms / 1000.0),
            max_speech_duration_seconds=max(1.0, vad_max_speech_duration_s),
        )
        self.stt = KhmerSTTService(model_name_or_path=whisper_model)
        self.translator = KhmerTranslationService(ollama_model=ollama_model, ollama_host=ollama_host)
        self.auditor = OllamaComplaintAuditor(model=ollama_model, host=ollama_host)

        self._lock = threading.Lock()
        self._is_running = False
        self._event_listeners: List[Callable[[Dict[str, Any]], None]] = []

        # Worker queue for asynchronous processing without blocking audio capture
        self._process_queue: collections.deque = collections.deque(maxlen=16)
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Deduplication state keyed by (camera_id, fingerprint, transcript)
        self._recent_keys: Dict[tuple, float] = {}

    def add_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for real-time complaint events."""
        with self._lock:
            self._event_listeners.append(listener)

    def _notify_listeners(self, complaint_dict: Dict[str, Any]) -> None:
        with self._lock:
            listeners = list(self._event_listeners)
        for l in listeners:
            try:
                l(complaint_dict)
            except Exception as e:
                logger.error(f"[ComplaintService] Listener error: {e}")

    def start(self) -> bool:
        """Start background audio capture and processing workers."""
        with self._lock:
            if self._is_running:
                return True
            if not self.enabled:
                logger.info("[ComplaintService] Disabled by configuration.")
                return False

            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="ComplaintProcessingWorker",
                daemon=True,
            )
            self._worker_thread.start()

            # Preload the Whisper model in the background so the first utterance
            # isn't delayed by a single-threaded 20-30s model load at detection time.
            threading.Thread(
                target=self._preload_models,
                name="ComplaintModelPreload",
                daemon=True,
            ).start()

            ok = self.audio_source.start(self._on_audio_chunk)
            if ok:
                self._is_running = True
                logger.info("[ComplaintService] Audio monitoring started.")
            else:
                logger.warning("[ComplaintService] Audio source could not be started.")
            return ok

    def stop(self) -> None:
        """Stop audio monitoring."""
        with self._lock:
            if not self._is_running:
                return
            self._is_running = False
            self._stop_event.set()

        self.audio_source.stop()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        logger.info("[ComplaintService] Audio monitoring stopped.")

    def _preload_models(self) -> None:
        try:
            if self.stt is not None:
                self.stt.load_model()
        except Exception as exc:
            logger.warning(f"[ComplaintService] Background STT preload failed: {exc}")

    def set_audio_source(self, audio_source: AudioSource) -> bool:
        """Dynamically swap audio capture source while service is running."""
        with self._lock:
            old_source = self.audio_source
            was_running = self._is_running

            if old_source is not None:
                try:
                    old_source.stop()
                except Exception as e:
                    logger.warning(f"[ComplaintService] Error stopping old audio source: {e}")

            self.audio_source = audio_source
            self._audio_source_label = type(audio_source).__name__.lower().replace("source", "")

            if was_running and self.audio_source is not None:
                ok = self.audio_source.start(self._on_audio_chunk)
                if ok:
                    logger.info(f"[ComplaintService] Successfully switched to new audio source: {type(audio_source).__name__}")
                else:
                    logger.warning(f"[ComplaintService] Failed to start switched audio source: {type(audio_source).__name__}")
                return ok
            return True

    def _on_audio_chunk(self, chunk: np.ndarray, timestamp: float) -> None:
        """Called for every 32ms chunk of audio from the microphone."""
        segment: Optional[SpeechSegment] = self.vad.process_chunk(chunk, timestamp)
        if segment is not None:
            if len(self._process_queue) >= self._process_queue.maxlen:
                logger.warning(
                    f"[ComplaintService] Segment queue full ({self._process_queue.maxlen}); "
                    "dropping newest segment to protect pipeline latency."
                )
            else:
                self._process_queue.append(segment)

    def _worker_loop(self) -> None:
        """Asynchronously processes speech utterances so audio capture is never blocked."""
        while not self._stop_event.is_set():
            if not self._process_queue:
                time.sleep(0.05)
                continue

            segment: SpeechSegment = self._process_queue.popleft()
            try:
                self.process_speech_segment(segment)
            except Exception as exc:
                logger.error(f"[ComplaintService] Error processing speech segment: {exc}", exc_info=True)

    _AUDIO_FP_STRIDE: int = 8  # ~2k samples/s -> stable, fast fingerprints

    def _audio_fingerprint(self, audio: np.ndarray) -> str:
        """Coarse perceptual hash of a speech segment for duplicate detection."""
        try:
            samples = audio[:: self._AUDIO_FP_STRIDE]
            quant = np.clip(np.rint(samples * 16.0), -128, 127).astype(np.int8)
            return hashlib.sha256(quant.tobytes()).hexdigest()
        except Exception:
            return ""

    def _is_duplicate(self, camera_id: str, fingerprint: str, transcript: str) -> bool:
        """True if the same utterance was already handled within the dedup window.

        Matches either an identical audio fingerprint (looped test clips) or an
        overlapping/identical transcript (same utterance split into VAD segments).
        """
        if not self.dedup_window_seconds or self.dedup_window_seconds <= 0:
            return False
        now = time.time()
        transcript = (transcript or "").strip()
        cutoff = now - self.dedup_window_seconds
        with self._lock:
            stale = [k for k, ts in self._recent_keys.items() if ts < cutoff]
            for k in stale:
                self._recent_keys.pop(k, None)
            for (cam, fp, txt), ts in self._recent_keys.items():
                if cam != camera_id:
                    continue
                if fp and fingerprint and fp == fingerprint:
                    return True
                if txt and transcript and (transcript in txt or txt in transcript):
                    return True
        return False

    def _remember(self, camera_id: str, fingerprint: str, transcript: str) -> None:
        key = (camera_id, fingerprint, transcript or "")
        with self._lock:
            self._recent_keys[key] = time.time()

    def process_speech_segment(
        self,
        segment: SpeechSegment,
        customer_id: str = "Customer",
        camera_id: str = "cam-1",
    ) -> Optional[ComplaintRecord]:
        """Runs STT -> Translation -> Ollama Audit -> Evidence Packaging -> Telegram."""
        now = datetime.now()
        stamp_str = now.strftime("%Y-%m-%d_%H%M%S")
        uid = f"CMP-{int(time.time()*1000)%1000000:06d}"

        # 1. Save Original Audio (Only once, reusing the same buffer)
        audio_filename = f"complaint_{stamp_str}_{uid}.wav"
        audio_path = self._audio_dir / audio_filename
        try:
            import soundfile as sf

            sf.write(str(audio_path), segment.audio, segment.sample_rate, subtype="PCM_16")
            logger.info(f"[ComplaintService] Saved speech audio to {audio_path}")
        except Exception as exc:
            logger.error(f"[ComplaintService] Failed to save WAV file: {exc}")

        # 2. Khmer STT (faster-whisper)
        khmer_text = self.stt.transcribe(segment.audio, sample_rate=segment.sample_rate)
        if not khmer_text or len(khmer_text.strip()) < 2:
            logger.info("[ComplaintService] No intelligible Khmer speech transcribed. Dropping.")
            return None

        # 2b. Script validation gate — accept complaints in EITHER Khmer script
        # (native Khmer customers) OR Latin/English script (foreign customers).
        # Only pure-noise transcripts (low coverage in BOTH scripts) are dropped,
        # so bilingual complaints always reach the translator + auditor + Telegram.
        km_cov = _khmer_coverage(khmer_text)
        lat_cov = _latin_coverage(khmer_text)
        if km_cov < 0.5 and lat_cov < 0.5:
            logger.info(
                f"[ComplaintService] Low script coverage in both Khmer ({km_cov:.0%}) "
                f"and Latin ({lat_cov:.0%}); dropping transcript '{khmer_text}'."
            )
            return None
        logger.info(
            f"[ComplaintService] Script gate passed (khmer={km_cov:.0%}, latin={lat_cov:.0%}) "
            f"-> '{khmer_text}'"
        )

        # 3. Khmer -> English Translation (Netra-NMT / Ollama).
        # Bilingual: if the surviving transcript is ENGLISH-dominant (foreign
        # customer spoke English), skip the wrong-direction Khmer NMT and use
        # it verbatim — the auditor reads English directly. Only run NMT for
        # Khmer-dominant transcripts.
        if lat_cov >= 0.5 and km_cov < 0.5:
            logger.info(
                f"[ComplaintService] English-dominant transcript ({lat_cov:.0%}); "
                f"skipping Khmer NMT, using transcript as English."
            )
            english_text = khmer_text
        else:
            english_text = self.translator.translate_km_to_en(khmer_text)

        # 4. Ollama Complaint Evaluation
        analysis: ComplaintAnalysis = self.auditor.analyze(khmer_text, english_text)

        is_complaint = bool(analysis.is_complaint)
        category = analysis.category if is_complaint else (analysis.category or "inquiry")
        severity = analysis.severity if is_complaint else "none"
        summary = analysis.summary

        if is_complaint:
            logger.info(f"[ComplaintService] 🚨 CONFIRMED COMPLAINT [{uid}]: {category.upper()} ({severity}) - '{summary}'")
        else:
            logger.info(f"[ComplaintService] 💬 Dialogue [{uid}]: (Category: {category}) - '{summary}'")

        if is_complaint:
            # Deduplicate repeated/loop-replayed utterances and VAD splits.
            audio_fp = self._audio_fingerprint(segment.audio)
            if self._is_duplicate(camera_id, audio_fp, khmer_text):
                logger.info(
                    f"[ComplaintService] Duplicate complaint suppressed [{uid}] "
                    f"within {self.dedup_window_seconds}s window: '{khmer_text}'"
                )
                return None
            self._remember(camera_id, audio_fp, khmer_text)

        # 5. Capture / Retrieve Camera Frame
        screenshot_path = None
        screenshot_filename = f"complaint_{stamp_str}_{uid}.jpg"
        screenshot_dest = self._stills_dir / screenshot_filename

        frame = None
        if self.get_camera_frame_fn is not None:
            try:
                frame = self.get_camera_frame_fn()
            except Exception as e:
                logger.warning(f"[ComplaintService] Error fetching camera frame: {e}")

        if frame is not None:
            try:
                annotated = frame.copy()
                h, w = annotated.shape[:2]
                if is_complaint:
                    banner_text = f"CUSTOMER COMPLAINT [{uid}] - {category.upper()} ({severity.upper()})"
                    banner_color = (0, 0, 180) # Red
                else:
                    banner_text = f"CUSTOMER DIALOGUE [{uid}] - {category.upper()}"
                    banner_color = (50, 100, 50) # Dark Green
                time_text = now.strftime("%Y-%m-%d %H:%M:%S")
                cv2.rectangle(annotated, (0, 0), (w, 54), banner_color, -1)
                cv2.putText(annotated, banner_text, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated, time_text, (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.imwrite(str(screenshot_dest), annotated)
                screenshot_path = str(screenshot_dest)
                logger.info(f"[ComplaintService] Saved screenshot to {screenshot_dest}")
            except Exception as exc:
                logger.error(f"[ComplaintService] Error writing annotated frame: {exc}")

        # 6. Save Record in SQLite
        record = ComplaintRecord(
            complaint_id=uid,
            customer_id=customer_id,
            camera_id=camera_id,
            audio_source=self._audio_source_label,
            timestamp=now.isoformat(),
            audio_path=str(audio_path),
            screenshot_path=screenshot_path,
            khmer_transcript=khmer_text,
            english_transcript=english_text,
            is_complaint=is_complaint,
            category=category,
            severity=severity,
            summary=summary,
            telegram_sent=False,
        )

        if self.db_conn is not None:
            insert_customer_complaint(
                conn=self.db_conn,
                complaint_id=record.complaint_id,
                audio_path=record.audio_path,
                screenshot_path=record.screenshot_path,
                khmer_transcript=record.khmer_transcript,
                english_transcript=record.english_transcript,
                category=record.category,
                severity=record.severity,
                summary=record.summary,
                customer_id=record.customer_id,
                camera_id=record.camera_id,
                audio_source=record.audio_source,
                timestamp=record.timestamp,
                is_complaint=record.is_complaint,
                telegram_sent=False,
            )

        if severity not in TG_SEVERITY_ORDER:
            severity = "none"
        if self.telegram_min_severity not in TG_SEVERITY_ORDER:
            self.telegram_min_severity = "low"

        # 7. Dispatch to Telegram only if it is a confirmed complaint above the
        # configured minimum severity (avoids noise while keeping the DB record).
        tg_should_send = (
            is_complaint
            and severity != "none"
            and TG_SEVERITY_ORDER.index(severity) >= TG_SEVERITY_ORDER.index(self.telegram_min_severity)
        )
        if tg_should_send and self.telegram_alert_enabled and self.telegram_out is not None and self.telegram_out.enabled:
            try:
                tg_ok = self.telegram_out.send_complaint_alert(
                    complaint=record.as_dict(),
                    audio_path=audio_path,
                    photo_path=Path(screenshot_path) if screenshot_path else None,
                )
                record.telegram_sent = tg_ok
                if self.db_conn is not None:
                    update_complaint_telegram_status(self.db_conn, record.complaint_id, sent=tg_ok)
            except Exception as tg_err:
                logger.error(f"[ComplaintService] Telegram dispatch failed: {tg_err}")
                record.telegram_error = str(tg_err)
                if self.db_conn is not None:
                    update_complaint_telegram_status(self.db_conn, record.complaint_id, sent=False, error=str(tg_err))

        # 8. Notify live dashboard listeners
        self._notify_listeners(record.as_dict())
        return record

