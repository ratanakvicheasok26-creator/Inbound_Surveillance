"""Verification script to test that Inbound Surveillance can ingest the virtual RTSP camera."""

from __future__ import annotations

import argparse
import os
import sys
import time

# Set FFmpeg options matching Inbound Surveillance RTSP adapter
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;2000000|max_delay;500000")

import cv2


def verify_rtsp_stream(url: str, num_frames: int = 15, timeout_s: float = 10.0) -> bool:
    print(f"[Verify] Connecting to virtual camera stream: {url}")
    t0 = time.time()

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[Verify] ERROR: Could not open stream at {url}")
        return False

    print("[Verify] Successfully connected! Reading frames...")
    frames_received = 0
    start_capture = time.time()

    while frames_received < num_frames:
        if time.time() - t0 > timeout_s:
            print(f"[Verify] ERROR: Timed out waiting for {num_frames} frames.")
            cap.release()
            return False

        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        frames_received += 1
        h, w, c = frame.shape
        fps = frames_received / max(0.001, (time.time() - start_capture))
        print(f"  Frame {frames_received:02d}/{num_frames}: {w}x{h} ({c} channels) @ {fps:.1f} FPS")

    cap.release()
    elapsed = time.time() - start_capture
    avg_fps = frames_received / max(0.001, elapsed)
    print(f"[Verify] SUCCESS: Read {frames_received} frames in {elapsed:.2f}s ({avg_fps:.1f} FPS).")
    print(f"[Verify] Stream is fully functional and ready for Inbound Surveillance ML inference!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify RTSP stream readability")
    parser.add_argument("--url", default="rtsp://127.0.0.1:8556/garage", help="RTSP stream URL to test")
    parser.add_argument("--frames", type=int, default=15, help="Number of frames to verify")
    args = parser.parse_args()

    success = verify_rtsp_stream(args.url, num_frames=args.frames)
    sys.exit(0 if success else 1)
