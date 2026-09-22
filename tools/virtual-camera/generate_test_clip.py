"""Generates a synthetic garage simulation video for testing when real camera video is unavailable."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def generate_garage_sample(output_path: str = "videos/sample_garage_demo.mp4", duration_sec: int = 15, fps: int = 30) -> str:
    """Renders a synthetic video showing garage bays, a parking car, a technician, and an IP camera timestamp."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    width = 1280
    height = 720
    total_frames = duration_sec * fps

    # H.264 or mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, float(fps), (width, height))

    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(out_file), fourcc, float(fps), (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_file}")

    print(f"[Generator] Rendering {total_frames} frames ({duration_sec}s @ {fps}fps) to {out_file}...")

    # Static garage floor layout
    bg = np.full((height, width, 3), (35, 38, 42), dtype=np.uint8)

    # Floor grid lines
    for y in range(0, height, 40):
        cv2.line(bg, (0, y), (width, y), (45, 48, 54), 1)
    for x in range(0, width, 40):
        cv2.line(bg, (x, 0), (x, height), (45, 48, 54), 1)

    # Bay 1 (Left Lift Bay)
    bay1_box = (150, 180, 500, 620)
    cv2.rectangle(bg, (bay1_box[0], bay1_box[1]), (bay1_box[2], bay1_box[3]), (60, 180, 240), 2)
    cv2.putText(bg, "BAY 1: LIFT STATION", (bay1_box[0] + 15, bay1_box[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 180, 240), 2)

    # Bay 2 (Right Tire/Brake Bay)
    bay2_box = (700, 180, 1100, 620)
    cv2.rectangle(bg, (bay2_box[0], bay2_box[1]), (bay2_box[2], bay2_box[3]), (100, 220, 100), 2)
    cv2.putText(bg, "BAY 2: TIRES & BRAKES", (bay2_box[0] + 15, bay2_box[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 220, 100), 2)

    # Tool Station
    tool_box = (40, 180, 120, 450)
    cv2.rectangle(bg, (tool_box[0], tool_box[1]), (tool_box[2], tool_box[3]), (200, 140, 40), 2)
    cv2.putText(bg, "TOOLS", (tool_box[0] + 5, tool_box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 140, 40), 1)

    t0 = time.time()
    for f in range(total_frames):
        frame = bg.copy()
        t = f / float(fps)  # current timestamp in video (seconds)

        # Vehicle simulation:
        # Car enters from bottom, drives into Bay 1, stops and gets worked on
        car_progress = min(1.0, t / 4.0)
        car_y = int(680 - car_progress * 350)
        car_x = 220
        car_w = 210
        car_h = 140

        # Draw vehicle body (blue sedan top-down perspective)
        cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), (140, 90, 40), -1)
        cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), (220, 180, 100), 2)
        # Windshields & roof
        cv2.rectangle(frame, (car_x + 35, car_y + 25), (car_x + car_w - 35, car_y + car_h - 25), (70, 50, 30), -1)
        cv2.putText(frame, "VEHICLE #402", (car_x + 35, car_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Mechanic / Technician simulation:
        # Walks from Tool Station to Bay 1, inspects car, bends over wheels
        if t >= 3.0:
            tech_t = min(1.0, (t - 3.0) / 3.0)
            tech_x = int(100 + tech_t * 220)
            tech_y = int(300 + np.sin(t * 5.0) * 15)

            # Technician body (head + shoulders + torso circle)
            cv2.circle(frame, (tech_x, tech_y), 18, (80, 120, 230), -1)
            cv2.circle(frame, (tech_x, tech_y), 18, (255, 255, 255), 1)
            cv2.circle(frame, (tech_x, tech_y - 6), 8, (200, 200, 255), -1)  # head
            cv2.putText(frame, "TECH (ALEX)", (tech_x - 30, tech_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 220, 240), 1)

        # IP Camera OSD (On-Screen Display) Header
        cv2.rectangle(frame, (0, 0), (width, 42), (15, 15, 18), -1)
        camera_label = f"CAM-01 [GARAGE MAIN]  1080p@30fps  VIRTUAL RTSP STREAM"
        timer_label = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0 + t)) + f".{int((t % 1) * 1000):03d}"
        cv2.putText(frame, camera_label, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        cv2.putText(frame, timer_label, (width - 320, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 230, 80), 1)

        writer.write(frame)

    writer.release()
    print(f"[Generator] Generated sample demo video: {out_file} ({out_file.stat().st_size} bytes)")
    return str(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic garage camera video")
    parser.add_argument("--output", default="videos/sample_garage_demo.mp4", help="Output file path")
    parser.add_argument("--duration", type=int, default=15, help="Duration in seconds")
    args = parser.parse_args()

    generate_garage_sample(args.output, duration_sec=args.duration)
