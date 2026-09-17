#!/usr/bin/env python3
"""Realtime laptop/USB mic health checker (is my microphone actually heard?).

Isolates the FIRST link of the complaint/audio pipeline before you ever
involve Whisper STT, Ollama, or Telegram:
    1. Lists every input device discoverable by sounddevice and arecord.
    2. Opens the chosen mic source using the SAME classes the launcher
       uses (LaptopMicrophoneSource / USBMicrophoneSource + arecord fallback).
    3. Shows a live, updating:
         * input RMS level (how loud the raw signal is)
         * Silero VAD speech probability (is a human voice present?)
    4. Prints a clear verdict:  MIC HEARD / MIC SILENT / NO-MIC-FEED.

Empty WAV proofs, missing STT output or a "silence" verdict from the full
pipeline almost always trace back to this first link, so check this FIRST.

Usage:
    venv/bin/python mic_check.py                     # laptop mic (default)
    venv/bin/python mic_check.py --source usb       # USB mic (first found)
    venv/bin/python mic_check.py --source usb --device 2
    venv/bin/python mic_check.py --source video --video IMG_1053.MOV
    venv/bin/python mic_check.py --list             # just list devices, don't listen

Exit codes:
    0  mic heard / VAD speech detected
    1  mic opened but stayed silent (energy) within the timeout
    2  failed to open any mic source
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional

import numpy as np

logger = logging.getLogger("mic_check")


def list_devices() -> None:
    """Print every audio input device discoverable on this machine."""
    print("=== Input devices (sounddevice / PortAudio) ===")
    sd_devs = 0
    try:
        import sounddevice as sd
        for dev in sd.query_devices():
            idx = dev["index"]
            name = dev["name"]
            inputs = int(dev.get("max_input_channels", 0))
            sr = dev.get("default_samplerate")
            mark = " *" if inputs > 0 else ""
            if inputs > 0:
                sd_devs += 1
            print(f"  [{idx}] {name}  (in={inputs}, sr={sr}){mark}")
        if sd_devs == 0:
            print("  (no input devices reported by PortAudio)")
    except Exception as exc:
        print(f"  sounddevice unavailable: {exc}")

    print("\n=== Input devices (arecord / ALSA) ===")
    import shutil
    if not shutil.which("arecord"):
        print("  arecord not installed")
        return
    import subprocess
    try:
        out = subprocess.run(
            ["arecord", "-l"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        print(out.stdout.strip() if out.stdout.strip() else "  {out.stderr.strip() or '(no capture cards found)'}")
    except Exception as exc:
        print(f"  arecord -l failed: {exc}")


def _build_source(args) -> Optional[object]:
    """Instantiate the same audio source classes the launcher uses."""
    from audio_source import (
        LaptopMicrophoneSource,
        USBMicrophoneSource,
        VideoFileAudioSource,
    )

    src_type = (args.source or "laptop").lower()
    if src_type == "usb":
        return USBMicrophoneSource(
            sample_rate=16000,
            device_index=args.device,
        )
    if src_type in ("video", "videofile"):
        if not args.video:
            print("error: --video <file> is required with --source video", file=sys.stderr)
            sys.exit(2)
        return VideoFileAudioSource(video_path=args.video, sample_rate=16000, loop=args.loop)
    return LaptopMicrophoneSource(sample_rate=16000, device_index=args.device)


def run_mic_check(args) -> int:
    source = _build_source(args)
    if source is None:
        print("❌ Could not create audio source.", file=sys.stderr)
        return 2

    from vad import SileroVAD
    vad = SileroVAD(
        sample_rate=16000,
        threshold=max(0.2, min(args.threshold, 0.95)),
        min_speech_duration_seconds=0.3,
        max_speech_duration_seconds=10.0,
    )

    peak_rms = 0.0
    peak_vad = 0.0
    speech_hits = 0
    chunk_count = 0
    last_level_bar = ""
    locked = False  # becomes True forever once speech is heard above threshold

    def on_chunk(audio: np.ndarray, _ts: float) -> None:
        nonlocal peak_rms, peak_vad, speech_hits, chunk_count, last_level_bar, locked
        chunk_count += 1
        try:
            rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        except Exception:
            rms = 0.0
        peak_rms = max(peak_rms, rms)
        prob = 0.0
        try:
            prob = float(vad.calculate_speech_prob(audio))
        except Exception:
            pass
        peak_vad = max(peak_vad, prob)
        if prob >= args.threshold:
            speech_hits += 1
            locked = True

        n_level = min(60, int(round(rms / 0.1 * 60)))  # normalize ~0.1 RMS = full bar
        n_vad = min(60, int(round(prob * 60)))
        last_level_bar = (
            "  RMS " + ("█" * n_level + "░" * (60 - n_level)).ljust(61)
            + "  VAD " + ("█" * n_vad + "░" * (60 - n_vad)).ljust(61)
            + f"  RMS={rms:.4f} VAD={prob:.3f}"
        )

    if not source.start(on_chunk):
        print("❌ Audio source failed to start. Run with --list to see devices.", file=sys.stderr)
        return 2

    print(f"\n🎙  Listening ({'laptop mic' if args.source and args.source != 'usb' else 'USB mic'} @16kHz) — speak, clap, or hum. Ctrl+C to stop.\n")
    try:
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            if last_level_bar:
                sys.stdout.write("\r" + last_level_bar)
                sys.stdout.flush()
            time.sleep(0.15)
    except KeyboardInterrupt:
        print()
    finally:
        source.stop()

    print("\n\n========================= MIC CHECK VERDICT =========================")
    print(f"  peak RMS : {peak_rms:.4f}")
    print(f"  peak VAD : {peak_vad:.3f}   (threshold {args.threshold})")
    print(f"  total chunks : {chunk_count}   speech hits: {speech_hits}")
    print("------------------------------------------------------------------")
    if chunk_count == 0:
        print("  ❌ NO-MIC-FEED  — the mic produced zero audio chunks.")
        print("     => Run with --list; pick device via --device, or check USB/Laptop mic cable.")
        return 2
    if locked and peak_rms >= 0.005:
        print("  ✅ MIC HEARD + VAD SPEECH DETECTED  — the pipeline's first link is healthy.")
        return 0
    if peak_rms >= 0.005:
        print("  ⚠️  MIC HEARD (signal present) but NO VAD SPEECH — check input level/threshold.")
        return 1
    print("  ❌ MIC SILENT  — signal baseline too quiet to classify as speech.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Realtime laptop/USB mic health checker.")
    parser.add_argument("--source", default="laptop", help="laptop | usb | video")
    parser.add_argument("--device", type=int, default=None, help="sounddevice device index (see --list)")
    parser.add_argument("--video", default=None, help="video file path for --source video")
    parser.add_argument("--loop", action="store_true", help="loop the video file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Silero VAD speech threshold (0-1)")
    parser.add_argument("--timeout", type=float, default=8.0, help="listen duration in seconds")
    parser.add_argument("--list", action="store_true", help="only list devices and exit")
    args = parser.parse_args()

    if args.list:
        list_devices()
        return 0
    return run_mic_check(args)


if __name__ == "__main__":
    sys.exit(main())
