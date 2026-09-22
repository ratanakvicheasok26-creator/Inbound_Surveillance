"""Extract video frames for YOLO fine-tuning and dataset training.

Usage:
    python tools/extract_dataset.py --video path/to/garage_video.mp4 --fps 1 --out dataset/images
"""

from __future__ import annotations

import argparse
from pathlib import Path
import cv2


def extract_frames(video_path: str | Path, output_dir: str | Path, target_fps: float = 1.0) -> int:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"[Error] Video file not found: {video_path}")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {video_path}")
        return 0

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(source_fps / max(0.1, target_fps))))

    frame_idx = 0
    saved_count = 0
    stem = video_path.stem

    print(f"[Dataset Extractor] Processing {video_path.name} (source FPS: {source_fps:.1f}, sampling every {step} frames)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            out_file = output_dir / f"{stem}_f{saved_count:05d}.jpg"
            cv2.imwrite(str(out_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"[Dataset Extractor] Successfully saved {saved_count} frames to {output_dir}")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Extract video frames for YOLO model training.")
    parser.add_argument("--video", type=str, required=True, help="Path to video file (MP4, MKV, AVI, etc.)")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract (default: 1.0)")
    parser.add_argument("--out", type=str, default="dataset/images", help="Output directory for extracted frames")
    args = parser.parse_args()

    extract_frames(args.video, args.out, target_fps=args.fps)


if __name__ == "__main__":
    main()
