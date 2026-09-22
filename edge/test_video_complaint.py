"""Video Customer Complaint Pipeline Tester.

Processes a real surveillance/customer video (e.g. IMG_1053.MOV):
1. Decodes audio & extracts video snapshot frames.
2. Runs Voice Activity Detection (VAD).
3. Transcribes Khmer customer dialogue (sengtha/whisper-base-khmer).
4. Translates Khmer to English.
5. Performs LLM Complaint Auditing (Ollama).
6. Persists incident to SQLite database & saves audio/snapshot proofs.
7. Triggers Telegram dispatch formatted payload.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
import random
import sys
import wave

import av
import cv2
import numpy as np

# Ensure edge dir in sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from complaint_auditor import OllamaComplaintAuditor
from db import connect, insert_customer_complaint
from speech_pipeline import KhmerSTTService, KhmerTranslationService
from telegram_out import TelegramOut
from vad import SileroVAD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("video_complaint_test")


def process_video_file(video_path: str, bay_id: str = "bay_1") -> dict:
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info(f"Opening video: {video_path}")
    container = av.open(str(video_file))

    # 1. Extract audio (resample to 16kHz mono float32)
    logger.info("Extracting and resampling audio track...")
    audio_pcm = np.array([], dtype=np.float32)
    if len(container.streams.audio) > 0:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        pcm_chunks = []
        for frame in container.decode(audio=0):
            for rframe in resampler.resample(frame):
                pcm_chunks.append(rframe.to_ndarray()[0])
        if pcm_chunks:
            audio_pcm = np.concatenate(pcm_chunks)
    
    audio_duration_sec = len(audio_pcm) / 16000.0 if len(audio_pcm) > 0 else 0.0
    logger.info(f"Decoded {len(audio_pcm)} audio samples ({audio_duration_sec:.2f}s).")

    # 2. Extract snapshot image from video
    logger.info("Extracting representative video snapshot frame...")
    snapshot_bgr = None
    container.seek(0)
    for frame in container.decode(video=0):
        # Convert pyav video frame to BGR image
        img = frame.to_ndarray(format="bgr24")
        if img is not None:
            snapshot_bgr = img
            # Pick a frame around 1-2 seconds in
            if frame.time is not None and frame.time >= 1.0:
                break

    if snapshot_bgr is None:
        # Create fallback test frame if video stream empty
        snapshot_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(snapshot_bgr, "Camera Feed", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    # 3. Voice Activity Detection (VAD)
    logger.info("Running Silero VAD analysis...")
    vad = SileroVAD(threshold=0.3)
    has_speech = False
    chunk_size = 512
    if len(audio_pcm) > chunk_size:
        for i in range(0, len(audio_pcm) - chunk_size, chunk_size):
            chunk = audio_pcm[i : i + chunk_size]
            prob = vad.calculate_speech_prob(chunk)
            if prob >= 0.3:
                has_speech = True
                break
    logger.info(f"VAD Speech Detected: {has_speech}")

    # 4. Speech to Text (Khmer)
    logger.info("Transcribing speech with fine-tuned Khmer model...")
    stt = KhmerSTTService()
    khmer_transcript = stt.transcribe(audio_pcm) if len(audio_pcm) > 0 else ""
    logger.info(f"Khmer Transcript: '{khmer_transcript}'")

    # 5. Translation (Khmer -> English)
    logger.info("Translating transcript to English...")
    translator = KhmerTranslationService()
    english_transcript = translator.translate(khmer_transcript) if khmer_transcript else ""
    logger.info(f"English Translation: '{english_transcript}'")

    # 6. Ollama Quality & Complaint Audit
    logger.info("Running Ollama complaint classification...")
    auditor = OllamaComplaintAuditor()
    analysis = auditor.analyze(khmer_transcript, english_transcript)
    logger.info(f"Complaint Verdict: is_complaint={analysis.is_complaint}, cat={analysis.category}, sev={analysis.severity}")

    # 7. Generate IDs and save artifacts
    now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    complaint_id = f"CMP-{random.randint(100000, 999999)}"

    proof_dir = Path("proofs/complaints")
    audio_dir = proof_dir / "audio"
    snap_dir = proof_dir / "snapshots"
    audio_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)

    wav_filename = f"complaint_{now_str}_{complaint_id}.wav"
    jpg_filename = f"snapshot_{now_str}_{complaint_id}.jpg"
    wav_path = audio_dir / wav_filename
    jpg_path = snap_dir / jpg_filename

    # Save audio WAV (16kHz 16-bit PCM)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        pcm_int16 = (np.clip(audio_pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
        wf.writeframes(pcm_int16.tobytes())

    # Save snapshot JPG
    cv2.imwrite(str(jpg_path), snapshot_bgr)
    logger.info(f"Saved artifacts: audio='{wav_path}', snapshot='{jpg_path}'")

    # 8. Store in SQLite DB
    conn = connect(Path("events.db"))
    saved = insert_customer_complaint(
        conn=conn,
        complaint_id=complaint_id,
        camera_id=bay_id,
        audio_path=str(wav_path),
        screenshot_path=str(jpg_path),
        khmer_transcript=khmer_transcript or "(No transcript detected)",
        english_transcript=english_transcript or "(No translation available)",
        is_complaint=analysis.is_complaint,
        category=analysis.category,
        severity=analysis.severity,
        summary=analysis.summary,
    )
    conn.close()
    logger.info(f"Persisted to SQLite events.db: success={saved}, Complaint ID: {complaint_id}")

    # 9. Dispatch to Telegram
    import yaml
    cfg_path = Path("config.yaml")
    tg_sent = False
    if cfg_path.exists():
        with cfg_path.open("r") as f:
            cfg = yaml.safe_load(f) or {}
        tg_token = cfg.get("telegram_bot_token", "")
        tg_chat = cfg.get("telegram_chat_id", "")
        if tg_token and tg_chat:
            tg = TelegramOut(token=tg_token, chat_id=tg_chat)
            logger.info(f"Dispatching complaint alert to Telegram chat {tg_chat}...")
            tg_sent = tg.send_complaint_alert(
                complaint={
                    "complaint_id": complaint_id,
                    "bay_id": bay_id,
                    "category": analysis.category,
                    "severity": analysis.severity,
                    "khmer_transcript": khmer_transcript,
                    "english_transcript": english_transcript,
                    "summary": analysis.summary,
                    "timestamp": now_str,
                },
                audio_path=wav_path,
                photo_path=jpg_path,
            )
            logger.info(f"Telegram dispatch result: {tg_sent}")

    tg_payload = {
        "complaint_id": complaint_id,
        "bay_id": bay_id,
        "category": analysis.category,
        "severity": analysis.severity,
        "khmer_transcript": khmer_transcript,
        "english_transcript": english_transcript,
        "summary": analysis.summary,
        "audio_clip_path": str(wav_path),
        "snapshot_path": str(jpg_path),
        "telegram_sent": tg_sent,
    }

    return {
        "complaint_id": complaint_id,
        "bay_id": bay_id,
        "audio_duration_sec": audio_duration_sec,
        "speech_detected": has_speech,
        "khmer_transcript": khmer_transcript,
        "english_transcript": english_transcript,
        "is_complaint": analysis.is_complaint,
        "category": analysis.category,
        "severity": analysis.severity,
        "summary": analysis.summary,
        "audio_path": str(wav_path),
        "snapshot_path": str(jpg_path),
        "telegram_payload": tg_payload,
    }


if __name__ == "__main__":
    video_input = sys.argv[1] if len(sys.argv) > 1 else "/home/ratanakvichea/Ibound-survillence-V2/IMG_1053.MOV"
    result = process_video_file(video_input)
    print("\n" + "=" * 60)
    print("VIDEO COMPLAINT ANALYSIS RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
