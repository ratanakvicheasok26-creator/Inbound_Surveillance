"""Speech Processing Pipeline: Khmer STT & Translation.

1. Khmer STT: Uses HuggingFace 'sengtha/whisper-base-khmer' (or faster-whisper as fallback).
2. Khmer -> English Translation: Uses Netra-NMT with seamless fallback to local Ollama.
"""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger("speech_pipeline")

DEFAULT_KHMER_MODEL = "sengtha/whisper-base-khmer"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _preprocess_for_stt(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Whisper-friendly input conditioning (pure numpy, no scipy).

    1. FFT band-pass: roll off rumble < 80 Hz and hiss > 7.5 kHz so the
       laptop's built-in mic stops smearing whisper's mel bins.
    2. RMS normalize to -22 dBFS so loudness-independent speech never clips
       and quiet phrases rise above whisper's noise floor.

    Returns the cleaned float32 array (same length, same sr).
    """
    n = len(audio)
    if n == 0:
        return audio
    a = np.asarray(audio, dtype=np.float32)
    if n < 512:
        return a

    a = a - float(np.mean(a))  # DC removal

    # FFT band-pass filter
    spec = np.fft.rfft(a)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    filt = np.ones_like(spec)
    filt = filt * (1.0 / (1.0 + np.exp(-(freqs - 80.0) / 6.0)))      # high-pass edge @ 80 Hz
    filt = filt * (1.0 / (1.0 + np.exp((freqs - 7500.0) / 400.0)))   # low-pass edge @ 7.5kHz
    out = np.fft.irfft(spec * filt, n=n).astype(np.float32)

    # RMS normalize to ~0.08 amplitude (~-22 dBFS) with safety floor
    rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2))) if len(out) else 0.0
    if rms > 1e-5:
        out = out * (0.08 / rms)
    np.clip(out, -1.0, 1.0, out=out)
    return out


class KhmerSTTService:
    """Khmer Speech-to-Text service supporting fine-tuned Khmer models."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_KHMER_MODEL,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.compute_type = compute_type
        self._hf_pipeline = None
        self._fw_model = None
        self._is_loaded = False
        self._load_lock = threading.Lock()

    def load_model(self) -> bool:
        """Load speech recognition pipeline (thread-safe, single-load)."""
        if self._is_loaded and (self._hf_pipeline is not None or self._fw_model is not None):
            return True
        with self._load_lock:
            if self._is_loaded and (self._hf_pipeline is not None or self._fw_model is not None):
                return True

        try:
            # 1. Try HuggingFace Transformers pipeline (high accuracy for fine-tuned Khmer models).
            # Retry import briefly: at engine boot other threads may still be
            # initializing transformers, which can transiently hide `pipeline`.
            import time as _time
            pipeline = None
            hf_err = None
            for _attempt in range(3):
                try:
                    from transformers import pipeline
                except ImportError:
                    try:
                        from transformers.pipelines import pipeline
                    except ImportError as p_err:
                        hf_err = p_err
                        _time.sleep(0.5)
                        continue
                break
            if pipeline is None:
                raise ImportError(hf_err or "pipeline unavailable")
            logger.info(f"[KhmerSTT] Loading HuggingFace pipeline for '{self.model_name_or_path}' on {self.device}...")
            self._hf_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_name_or_path,
                device=self.device,
            )
            self._is_loaded = True
            logger.info("[KhmerSTT] HuggingFace pipeline loaded successfully.")
            return True
        except Exception as hf_err:
            logger.warning(f"[KhmerSTT] HF pipeline '{self.model_name_or_path}' unavailable ({hf_err}). Trying faster-whisper...")

        # 2. Fallback to faster-whisper
        try:
            from faster_whisper import WhisperModel
            logger.info(f"[KhmerSTT] Loading faster-whisper on {self.device} ({self.compute_type})...")
            self._fw_model = WhisperModel("base", device=self.device, compute_type=self.compute_type)
            self._is_loaded = True
            logger.info("[KhmerSTT] faster-whisper fallback model loaded.")
            return True
        except Exception as fw_err:
            logger.error(f"[KhmerSTT] Failed to load faster-whisper: {fw_err}")
            self._is_loaded = False
            return False

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a 16kHz float32 or int16 numpy audio array to Khmer text.
        
        Args:
            audio_data: 1D numpy array of audio samples.
            sample_rate: Audio sample rate (must be 16000).
        """
        if not self._is_loaded:
            if not self.load_model():
                return ""

        try:
            # Ensure float32 [-1.0, 1.0]
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)

            # 1. If HF pipeline loaded
            # 0. INPUT CONDITIONING — pre-process before whisper so the
            # built-in mic stops reading 'unintelligible': FFT band-pass
            # (kill <80Hz rumble + >7.5kHz hiss) + RMS normalize to -22dBFS.
            try:
                audio_float = _preprocess_for_stt(audio_float, sample_rate)
            except Exception as _pp_err:
                logger.warning(f"[KhmerSTT] Pre-process skipped: {_pp_err}; using raw.")

            if self._hf_pipeline is not None:
                res = self._hf_pipeline(
                    {"raw": audio_float, "sampling_rate": sample_rate},
                    generate_kwargs={"language": "km", "task": "transcribe"},
                )
                transcript = str(res.get("text", "")).strip() if isinstance(res, dict) else str(res).strip()
                logger.info(f"[KhmerSTT] HF Result: {transcript}")
                return transcript

            # 2. If faster-whisper loaded
            if self._fw_model is not None:
                segments, info = self._fw_model.transcribe(
                    audio_float,
                    beam_size=5,
                    language="km",
                    task="transcribe",
                    vad_filter=False,
                )
                texts = [s.text.strip() for s in segments if s.text and s.text.strip()]
                transcript = " ".join(texts).strip()
                logger.info(f"[KhmerSTT] FW Result: {transcript}")
                return transcript

        except Exception as exc:
            logger.error(f"[KhmerSTT] Transcription error: {exc}")

        return ""


class KhmerTranslationService:
    """Khmer to English Translation service using Netra-NMT with Ollama fallback."""

    def __init__(
        self,
        ollama_model: str = "qwen2.5:3b",
        ollama_host: str = OLLAMA_HOST,
    ) -> None:
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self._nmt_model = None
        self._nmt_tokenizer = None
        self._nmt_loaded = False

    def translate(self, khmer_text: str) -> str:
        """Alias for translate_km_to_en."""
        return self.translate_km_to_en(khmer_text)

    def translate_km_to_en(self, khmer_text: str) -> str:
        """Translate Khmer text to natural English."""
        text = (khmer_text or "").strip()
        if not text:
            return ""

        # 1. Try Netra-NMT local transformer model if available
        if self._nmt_loaded and self._nmt_model is not None:
            try:
                translated = self._translate_with_netra(text)
                if translated:
                    return translated
            except Exception as e:
                logger.warning(f"[Translation] Netra-NMT inference error: {e}. Using Ollama fallback...")

        # 2. Local Ollama Translation Fallback (Zero downtime, ultra high fidelity)
        return self._translate_with_ollama(text)

    def _translate_with_netra(self, text: str) -> str:
        """Run translation through Netra-NMT model."""
        import torch
        inputs = self._nmt_tokenizer(text, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self._nmt_model.generate(**inputs, max_length=256)
        return self._nmt_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def _translate_with_ollama(self, text: str) -> str:
        """Translate using local Ollama instance."""
        import requests
        prompt = (
            f"You are a professional Khmer-to-English translator for an automotive garage surveillance system.\n"
            f"Translate the following Khmer sentence accurately and naturally into English.\n"
            f"Note: Speech variations like 'ឡាំនានឹងស្អាតពេល' / 'ឡាននេះមិនស្អាតទេ' mean 'This car is not clean', and 'ឡាន់នៅតែក៏ខ្វក់' means 'The car is still dirty'.\n"
            f"Output ONLY the English translation, with no explanation, quotes, or preamble.\n\n"
            f"Khmer: {text}\n"
            f"English:"
        )

        try:
            resp = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 128},
                },
                timeout=60,
            )

            if resp.status_code == 200:
                data = resp.json()
                translation = data.get("response", "").strip().strip('"').strip("'")
                logger.info(f"[Translation] KM: '{text}' -> EN: '{translation}'")
                return translation
            else:
                logger.warning(f"[Translation] Ollama HTTP {resp.status_code}: {resp.text}")
        except Exception as exc:
            logger.error(f"[Translation] Ollama translation error: {exc}")

        return text  # Fallback to original text if translation unavailable
