"""Manual probe: transcribe a clip, translate it, and audit it as a complaint.

This is an investigation tool, not a test -- it needs a real media file, a
fine-tuned Khmer STT checkpoint, and a running Ollama. Run it directly:

    python probe_user_sound.py /path/to/clip.mov

It used to be named test_user_sound.py, which made `unittest discover` import
it during collection: the module-level `av.open()` and Ollama call ran on every
test run, and because the path came from `sys.argv[1]` (the test pattern under
discovery) the whole module died with
`FileNotFoundError: ... 'test_user_sound'`. Everything now lives under main().
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import av
import numpy as np

DEFAULT_CLIP = Path(__file__).resolve().parent.parent / "IMG_1053.MOV"
SAMPLE_RATE = 16000


def decode_audio(path: Path) -> np.ndarray:
    """Decode the first audio stream to mono 16 kHz float PCM."""
    container = av.open(str(path))
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
    chunks = [
        resampled.to_ndarray()[0]
        for frame in container.decode(audio=0)
        for resampled in resampler.resample(frame)
    ]
    container.close()
    if not chunks:
        raise ValueError(f"{path.name} has no decodable audio stream")
    return np.concatenate(chunks)


def resolve_clip(arg: str | None) -> Path:
    clip = Path(arg) if arg else DEFAULT_CLIP
    if not clip.is_file():
        raise SystemExit(
            f"audio clip not found: {clip}\n"
            f"usage: python probe_user_sound.py /path/to/clip.mov"
        )
    return clip


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", help=f"media file (default: {DEFAULT_CLIP})")
    args = parser.parse_args(argv)
    clip = resolve_clip(args.audio)

    from complaint_auditor import OllamaComplaintAuditor
    from speech_pipeline import KhmerSTTService, KhmerTranslationService

    print(f"Loading media from: {clip}")
    audio_pcm = decode_audio(clip)
    print(f"\n[1] Decoded Audio Length: {len(audio_pcm)} samples ({len(audio_pcm) / SAMPLE_RATE:.2f}s)")

    print("\n[2] Transcribing with fine-tuned Khmer model...")
    km_text = KhmerSTTService().transcribe(audio_pcm)
    print(f'--> Khmer Transcript: "{km_text}"')

    print("\n[3] Translating to English...")
    en_text = KhmerTranslationService().translate(km_text)
    print(f'--> English Translation: "{en_text}"')

    print("\n[4] Auditing Complaint with LLM...")
    analysis = OllamaComplaintAuditor().analyze(km_text, en_text)

    print("\n================ FINAL COMPLAINT VERDICT ================")
    print(json.dumps(analysis.as_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
