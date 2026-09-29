"""Automated Person Crop Collector for Role & Uniform Training.

Extracts cropped person images from CCTV video files or RTSP streams,
bounding-boxed by YOLO11-pose, for fine-tuning the Role Classifier.

Usage:
    python tools/collect_role_dataset.py --video videos/document_6161011834561243252.mp4 --out dataset/roles --default-label security_guard
"""

from __future__ import annotations

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO


def extract_person_crops(
    video_path: str | Path,
    out_dir: str | Path,
    default_label: str = "unassigned",
    conf_thresh: float = 0.35,
    sample_every_n_frames: int = 5,
) -> int:
    video_path = Path(video_path)
    out_dir = Path(out_dir) / default_label
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"[Error] Video file not found: {video_path}")
        return 0

    model = YOLO("yolo11n-pose.pt")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {video_path}")
        return 0

    frame_idx = 0
    crop_idx = 0
    stem = video_path.stem

    print(f"[Collector] Extracting person crops from {video_path.name} to {out_dir}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every_n_frames == 0:
            results = model(frame, verbose=False)[0]
            boxes = results.boxes
            for b in boxes:
                conf = float(b.conf[0].cpu().numpy())
                if conf < conf_thresh:
                    continue
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].cpu().numpy()]
                h_f, w_f = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_f, x2), min(h_f, y2)
                bw, bh = x2 - x1, y2 - y1
                if bw < 25 or bh < 50:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_file = out_dir / f"{stem}_f{frame_idx:05d}_p{crop_idx:04d}.jpg"
                    cv2.imwrite(str(crop_file), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    crop_idx += 1

        frame_idx += 1

    cap.release()
    print(f"[Collector] Finished! Saved {crop_idx} person crops in '{out_dir}'.")
    return crop_idx


def main():
    parser = argparse.ArgumentParser(description="Extract person crops for role classifier dataset.")
    parser.add_argument("--video", type=str, required=True, help="Video path")
    parser.add_argument("--out", type=str, default="dataset/roles", help="Dataset output folder")
    parser.add_argument("--default-label", type=str, default="unassigned", choices=["customer", "security_guard", "spa_staff", "delivery", "unassigned"], help="Category label for crops in this video")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--step", type=int, default=5, help="Sample every N frames")
    args = parser.parse_args()

    extract_person_crops(
        video_path=args.video,
        out_dir=args.out,
        default_label=args.default_label,
        conf_thresh=args.conf,
        sample_every_n_frames=args.step,
    )


if __name__ == "__main__":
    main()
