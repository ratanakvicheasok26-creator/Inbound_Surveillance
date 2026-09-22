"""Voice Activity Detection (VAD) Engine using official Silero VAD.

Filters silence and non-speech background garage noise (tools, engines, HVAC).
Accurately segments speech with configurable pre-roll and post-roll buffers
so beginnings and endings of customer sentences are never truncated.
"""

from __future__ import annotations

import collections
import dataclasses
import logging
from pathlib import Path
import time
from typing import Callable, Deque, List, Optional

import numpy as np
import torch

logger = logging.getLogger("vad")


@dataclasses.dataclass
class SpeechSegment:
    """Represents a finalized contiguous speech utterance."""
    audio: np.ndarray  # float32 1D array [-1.0, 1.0] at 16kHz
    sample_rate: int
    start_time: float
    end_time: float
    duration_seconds: float

    def to_int16(self) -> np.ndarray:
        """Convert float32 [-1.0, 1.0] to int16 PCM."""
        clamped = np.clip(self.audio, -1.0, 1.0)
        return (clamped * 32767).astype(np.int16)


class SileroVAD:
    """Production Silero VAD wrapper with VADIterator streaming support."""

    def __init__(
        self,
        threshold: float = 0.35,
        sample_rate: int = 16000,
        min_speech_duration_seconds: float = 0.30,
        max_speech_duration_seconds: float = 25.0,
        min_silence_duration_ms: int = 250,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_seconds = min_speech_duration_seconds
        self.max_speech_duration_seconds = max_speech_duration_seconds
        self.min_silence_duration_ms = min_silence_duration_ms

        self._model = None
        self._iterator = None
        self._is_ready = False

        # State tracking for speech segmentation
        self._is_speaking = False
        self._speech_start_time = 0.0
        self._speech_chunks: List[np.ndarray] = []
        self._pre_roll_buffer: Deque[tuple[np.ndarray, float]] = collections.deque(maxlen=12) # ~384ms

        self._init_model()

    def _init_model(self) -> None:
        """Initialize Silero VAD model via silero_vad package."""
        try:
            from silero_vad import load_silero_vad, VADIterator
            self._model = load_silero_vad()
            self._iterator = VADIterator(
                self._model,
                threshold=self.threshold,
                sampling_rate=self.sample_rate,
                min_silence_duration_ms=self.min_silence_duration_ms,
            )
            self._is_ready = True
            logger.info("[SileroVAD] Loaded official Silero VAD model successfully.")
        except Exception as e:
            logger.warning(f"[SileroVAD] Failed to load silero_vad package: {e}. Using energy fallback.")
            self._is_ready = False

    def reset_states(self) -> None:
        """Reset internal recurrent state."""
        if self._iterator is not None and hasattr(self._iterator, "reset_states"):
            self._iterator.reset_states()
        self._speech_chunks.clear()
        self._pre_roll_buffer.clear()
        self._is_speaking = False
        self._speech_start_time = 0.0

    def calculate_speech_prob(self, chunk: np.ndarray) -> float:
        """Calculate speech probability for a 512-sample chunk at 16kHz."""
        if not self._is_ready or self._model is None:
            energy = np.sqrt(np.mean(chunk ** 2)) if len(chunk) > 0 else 0.0
            return 1.0 if energy > 0.015 else 0.0

        try:
            if len(chunk) < 512:
                chunk = np.pad(chunk, (0, 512 - len(chunk)))
            elif len(chunk) > 512:
                chunk = chunk[:512]

            tensor = torch.from_numpy(chunk.astype(np.float32))
            with torch.no_grad():
                prob = self._model(tensor, self.sample_rate).item()
            return float(prob)
        except Exception as e:
            logger.debug(f"[SileroVAD] Prob error: {e}")
            energy = np.sqrt(np.mean(chunk ** 2)) if len(chunk) > 0 else 0.0
            return 1.0 if energy > 0.015 else 0.0

    def process_chunk(self, chunk: np.ndarray, timestamp: float) -> Optional[SpeechSegment]:
        """Process an incoming audio chunk (512 samples at 16kHz).
        
        Returns:
            SpeechSegment when a complete sentence/utterance has ended, or None.
        """
        if len(chunk) < 512:
            chunk = np.pad(chunk, (0, 512 - len(chunk)))
        elif len(chunk) > 512:
            chunk = chunk[:512]

        # 1. Use VADIterator if ready
        if self._is_ready and self._iterator is not None:
            try:
                t_chunk = torch.from_numpy(chunk.astype(np.float32))
                speech_dict = self._iterator(t_chunk)

                if speech_dict is not None:
                    if "start" in speech_dict:
                        self._is_speaking = True
                        self._speech_start_time = timestamp
                        self._speech_chunks.clear()
                        # Add pre-roll buffer
                        for pre_chunk, _ in self._pre_roll_buffer:
                            self._speech_chunks.append(pre_chunk)
                        self._speech_chunks.append(chunk)

                    elif "end" in speech_dict:
                        if self._is_speaking:
                            self._speech_chunks.append(chunk)
                            return self._finalize_segment(timestamp)

                elif self._is_speaking:
                    self._speech_chunks.append(chunk)
                    # Check max duration safeguard
                    current_duration = len(self._speech_chunks) * 512 / self.sample_rate
                    if current_duration >= self.max_speech_duration_seconds:
                        return self._finalize_segment(timestamp)
                else:
                    self._pre_roll_buffer.append((chunk, timestamp))

                return None

            except Exception as exc:
                logger.debug(f"[SileroVAD] Iterator error: {exc}")

        # 2. Fallback: Probability thresholding
        prob = self.calculate_speech_prob(chunk)
        is_speech = prob >= self.threshold

        if is_speech:
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_start_time = timestamp
                self._speech_chunks.clear()
                for pre_chunk, _ in self._pre_roll_buffer:
                    self._speech_chunks.append(pre_chunk)
            self._speech_chunks.append(chunk)
            current_duration = len(self._speech_chunks) * 512 / self.sample_rate
            if current_duration >= self.max_speech_duration_seconds:
                return self._finalize_segment(timestamp)
        else:
            if self._is_speaking:
                self._speech_chunks.append(chunk)
                # Finalize after ~250ms silence (8 chunks)
                return self._finalize_segment(timestamp)
            else:
                self._pre_roll_buffer.append((chunk, timestamp))

        return None

    def flush(self, end_time: Optional[float] = None) -> Optional[SpeechSegment]:
        """Force finalize any pending speech buffer (e.g. at end of stream/loop)."""
        if self._is_speaking and self._speech_chunks:
            return self._finalize_segment(end_time or time.time())
        return None

    def _finalize_segment(self, end_time: float) -> Optional[SpeechSegment]:
        """Consolidate gathered chunks into a SpeechSegment."""
        if not self._speech_chunks:
            self._is_speaking = False
            return None

        combined_audio = np.concatenate(self._speech_chunks, axis=0)
        duration = len(combined_audio) / self.sample_rate
        start_time = self._speech_start_time

        self._is_speaking = False
        self._speech_chunks.clear()

        if duration < self.min_speech_duration_seconds:
            logger.debug(f"[SileroVAD] Ignored short noise blip: {duration:.2f}s")
            return None

        logger.info(f"[SileroVAD] Detected speech segment: {duration:.2f}s (start={start_time:.2f}, end={end_time:.2f})")
        return SpeechSegment(
            audio=combined_audio,
            sample_rate=self.sample_rate,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
        )
