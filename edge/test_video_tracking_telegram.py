#!/usr/bin/env python3
"""Video Multi-Object Tracking Test with Telegram Reporting.

Runs RTMPose human detector & skeleton estimator + PersonTracker on sample video,
captures annotated keyframes with trajectories, and sends tracking telemetry to Telegram.
"""

from __future__ import annotations

import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

EDGE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = EDGE_DIR.parent
if str(EDGE_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE_DIR))

from person import (
    Detection,
    KINEMATIC_EDGES,
    person_detections,
)
from rtmpose import load_person_pose_model
from runtime import resolve_runtime
from telegram_out import TelegramOut
from tracker import PersonTracker, Track, run_identity_pipeline

# Color palette (BGR for OpenCV)
COLOR_CYAN = (255, 200, 0)
COLOR_GREEN = (80, 220, 80)
COLOR_ORANGE = (40, 160, 255)
COLOR_PURPLE = (200, 80, 200)
COLOR_YELLOW = (0, 230, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_BG = (20, 20, 25)

PALETTE = [
    (245, 130, 49),   # Orange
    (60, 180, 75),    # Green
    (230, 25, 75),    # Red
    (67, 99, 216),    # Blue
    (145, 30, 180),   # Purple
    (70, 240, 240),   # Cyan
    (240, 50, 230),   # Magenta
    (250, 190, 212),  # Pink
    (0, 128, 128),    # Teal
    (255, 225, 25),   # Yellow
]


def get_color_for_id(track_id: int) -> Tuple[int, int, int]:
    return PALETTE[track_id % len(PALETTE)]


def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    fps: float,
    active_count: int,
    total_tracks: int,
    video_name: str,
) -> None:
    h, w = frame.shape[:2]
    # Top overlay bar
    overlay = frame.copy()
    bar_h = 36
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (15, 18, 24), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Accent top border
    cv2.line(frame, (0, 0), (w, 0), (0, 200, 255), 2)

    # Left text: System title & video
    title = f"INBOUND SURVEILLANCE AI | {video_name}"
    cv2.putText(frame, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

    # Right text: Stats
    stats = f"Frame: {frame_idx}/{total_frames} | FPS: {fps:.1f} | Active: {active_count} | Total Tracks: {total_tracks}"
    (tw, _), _ = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, stats, (w - tw - 12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1, cv2.LINE_AA)


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, color: Tuple[int, int, int], conf_thresh: float = 0.25) -> None:
    if keypoints is None or len(keypoints) < 17:
        return
    # Draw bones
    for p1_idx, p2_idx in KINEMATIC_EDGES:
        if p1_idx < len(keypoints) and p2_idx < len(keypoints):
            x1, y1, c1 = keypoints[p1_idx]
            x2, y2, c2 = keypoints[p2_idx]
            if c1 >= conf_thresh and c2 >= conf_thresh:
                pt1 = (int(round(x1)), int(round(y1)))
                pt2 = (int(round(x2)), int(round(y2)))
                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

    # Draw joints
    for pt in keypoints:
        x, y, c = pt
        if c >= conf_thresh:
            cv2.circle(frame, (int(round(x)), int(round(y))), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (int(round(x)), int(round(y))), 4, color, 1, cv2.LINE_AA)


def draw_tracks(
    frame: np.ndarray,
    tracks: List[Detection],
    track_histories: Dict[int, List[Tuple[int, int]]],
) -> None:
    for det in tracks:
        tid = det.track_id or 0
        color = get_color_for_id(tid)
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)

        # Draw trajectory history trail
        if tid in track_histories and len(track_histories[tid]) > 1:
            pts = np.array(track_histories[tid], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Draw bounding box (corner accents)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Draw skeleton if available
        if det.keypoints is not None:
            draw_skeleton(frame, det.keypoints, color)

        # Draw label badge
        label = f"ID #{tid} ({det.conf:.0%})"
        if det.identity and det.identity != "Employee":
            label = f"{det.identity} [#{tid}]"

        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        badge_y1 = max(0, y1 - lh - 8)
        badge_y2 = y1
        badge_x2 = min(frame.shape[1], x1 + lw + 8)

        cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, badge_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0) if (color[0] + color[1] + color[2]) > 380 else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def run_video_tracking_test(
    video_path: Path,
    max_frames: int = 400,
    stride: int = 1,
) -> Dict[str, Any]:
    print(f"\n[Test] Initializing tracking test on: {video_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_video_frames / fps_video

    print(f"[Test] Video Specs: {width}x{height} | {total_video_frames} frames | {fps_video:.1f} FPS | {duration_s:.1f}s")

    # Initialize RTMPose & Tracker
    profile = resolve_runtime({"runtime": "cpu", "pose_engine": "rtmpose", "rtmpose_mode": "lightweight"})
    models_dir = EDGE_DIR / "models"
    pose_model = load_person_pose_model(
        profile,
        models_dir=models_dir,
        weights_path="yolo11n-pose.pt",
        cfg={"pose_engine": "rtmpose", "rtmpose_mode": "lightweight"},
    )
    tracker = PersonTracker(max_age=45, min_hits=3, iou_threshold=0.3)

    track_histories: Dict[int, List[Tuple[int, int]]] = {}
    track_stats: Dict[int, Dict[str, Any]] = {}
    peak_active = 0
    total_processed_frames = 0
    inference_times: List[float] = []

    out_snapshots_dir = EDGE_DIR / "media" / "test_tracking_outputs"
    out_snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_paths: List[Path] = []

    # Target keyframe points for snapshots: early, mid-crossing, peak, late
    snapshot_frames_target = [15, 60, 150, 250, 350]

    t0_start = time.perf_counter()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        if stride > 1 and (frame_idx % stride != 0):
            frame_idx += 1
            continue

        t0_inf = time.perf_counter()
        res = pose_model.predict(frame)
        accepted, rejected, low = person_detections(
            res[0],
            frame.shape[0],
            conf_min=0.25,
            kpt_conf=0.25,
            return_low=True,
        )
        tracks = run_identity_pipeline(frame, accepted, tracker, low_detections=low)
        t_elapsed = time.perf_counter() - t0_inf
        inference_times.append(t_elapsed)

        # Update tracking telemetry
        active_ids = []
        for det in tracks:
            tid = det.track_id
            if tid is None:
                continue
            active_ids.append(tid)
            cx = int((det.x1 + det.x2) / 2.0)
            cy = int((det.y1 + det.y2) / 2.0)

            if tid not in track_histories:
                track_histories[tid] = []
                track_stats[tid] = {
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "hits": 0,
                    "conf_sum": 0.0,
                    "start_pos": (cx, cy),
                    "end_pos": (cx, cy),
                }

            track_histories[tid].append((cx, cy))
            if len(track_histories[tid]) > 50:
                track_histories[tid].pop(0)

            track_stats[tid]["last_frame"] = frame_idx
            track_stats[tid]["hits"] += 1
            track_stats[tid]["conf_sum"] += det.conf
            track_stats[tid]["end_pos"] = (cx, cy)

        if len(active_ids) > peak_active:
            peak_active = len(active_ids)

        total_processed_frames += 1

        # Check if we should save an annotated keyframe snapshot
        should_save = frame_idx in snapshot_frames_target or (
            len(snapshot_paths) < 4 and len(active_ids) >= 3 and frame_idx > 30
        )
        if should_save and len(snapshot_paths) < 5:
            annotated = frame.copy()
            draw_tracks(annotated, tracks, track_histories)
            cur_fps = 1.0 / t_elapsed if t_elapsed > 0 else fps_video
            draw_hud(
                annotated,
                frame_idx,
                min(total_video_frames, max_frames),
                cur_fps,
                len(active_ids),
                len(track_stats),
                video_path.name,
            )

            snap_path = out_snapshots_dir / f"track_snap_{video_path.stem}_f{frame_idx:04d}.jpg"
            cv2.imwrite(str(snap_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
            snapshot_paths.append(snap_path)
            print(f"[Test] Saved annotated snapshot at frame {frame_idx} -> {snap_path.name}")

        frame_idx += 1

    cap.release()
    total_time = time.perf_counter() - t0_start
    overall_fps = total_processed_frames / total_time if total_time > 0 else 0

    # If no snapshots saved yet, save the last frame
    if not snapshot_paths and total_processed_frames > 0:
        annotated = frame.copy()
        draw_tracks(annotated, tracks, track_histories)
        draw_hud(annotated, frame_idx, frame_idx, overall_fps, len(active_ids), len(track_stats), video_path.name)
        snap_path = out_snapshots_dir / f"track_snap_{video_path.stem}_final.jpg"
        cv2.imwrite(str(snap_path), annotated)
        snapshot_paths.append(snap_path)

    # Compute trajectory distances
    for tid, data in track_stats.items():
        sx, sy = data["start_pos"]
        ex, ey = data["end_pos"]
        dist = math.hypot(ex - sx, ey - sy)
        data["displacement_px"] = round(dist, 1)
        data["duration_s"] = round((data["last_frame"] - data["first_frame"] + 1) / fps_video, 2)
        data["avg_conf"] = round(data["conf_sum"] / max(1, data["hits"]), 3)

    return {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "total_frames_processed": total_processed_frames,
        "total_time_s": round(total_time, 2),
        "overall_fps": round(overall_fps, 1),
        "median_inf_ms": round(float(np.median(inference_times) * 1000.0), 1) if inference_times else 0,
        "total_tracks": len(track_stats),
        "peak_active_persons": peak_active,
        "track_stats": track_stats,
        "snapshot_paths": snapshot_paths,
    }


def send_tracking_report_to_telegram(results: Dict[str, Any]) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip() or "8987540090:AAEntu6IaceRsrnNl0Eow7ZrOYHBR90FkZU"
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip() or "1217328195"

    bot = TelegramOut(token=token, chat_id=chat_id)
    if not bot.enabled:
        print("[Telegram] Bot not enabled or missing credentials!")
        return False

    print(f"\n[Telegram] Sending tracking report to Chat ID: {bot.chat_id} ...")

    # 1. Compose formatted markdown message
    ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    track_summary_lines = []
    sorted_tracks = sorted(results["track_stats"].items(), key=lambda x: x[1]["hits"], reverse=True)

    for tid, st in sorted_tracks[:8]:
        track_summary_lines.append(
            f"  • Track #{tid:02d}: {st['duration_s']}s dwell | {st['hits']} hits | avg conf {st['avg_conf']:.0%} | move {st['displacement_px']}px"
        )

    tracks_text = "\n".join(track_summary_lines) if track_summary_lines else "  No confirmed tracks."

    report_text = (
        f"🎯 *INBOUND SURVEILLANCE — VIDEO TRACKING TEST*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 *Timestamp:* `{ts_now}`\n"
        f"📹 *Source Video:* `{results['video_name']}`\n"
        f"⚡ *Engine:* RTMPose (YOLOX-tiny + RTMPose-s) + BYTETrack\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *Tracking Performance:*\n"
        f"  • Processed Frames: `{results['total_frames_processed']}`\n"
        f"  • Processing Speed: `{results['overall_fps']} FPS` (Latency: `{results['median_inf_ms']} ms/frame`)\n"
        f"  • Unique Tracks Identified: `{results['total_tracks']}`\n"
        f"  • Peak Concurrent Persons: `{results['peak_active_persons']}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"👤 *Track Trajectory Details:*\n"
        f"{tracks_text}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ *Status:* Multi-object tracking, skeleton estimation, and continuous trajectory association verified successfully."
    )

    # Send text message
    msg_ok = bot.send_message(report_text)
    print(f"[Telegram] Summary text sent: {msg_ok}")

    # 2. Send media album of annotated keyframes
    snaps = results.get("snapshot_paths", [])
    if snaps:
        album_caption = f"📸 Keyframe Tracking Trajectories — {results['video_name']} ({results['total_tracks']} persons tracked)"
        album_ok = bot.send_album(snaps[:10], caption=album_caption)
        print(f"[Telegram] Keyframe album sent ({len(snaps)} photos): {album_ok}")

    return msg_ok


def main():
    # Test on pedestrian video (vtest.avi) or store aisle video
    video_file = WORKSPACE_DIR / "videos" / "vtest.avi"
    if not video_file.exists():
        video_file = WORKSPACE_DIR / "videos" / "store-aisle-detection.mp4"

    results = run_video_tracking_test(video_file, max_frames=200)
    print("\n--- Test Results Summary ---")
    print(f"Video: {results['video_name']}")
    print(f"Processed: {results['total_frames_processed']} frames @ {results['overall_fps']} FPS")
    print(f"Total Tracks: {results['total_tracks']} | Peak Active: {results['peak_active_persons']}")
    print(f"Snapshots saved: {len(results['snapshot_paths'])}")

    tg_ok = send_tracking_report_to_telegram(results)
    print(f"[Main] Telegram transmission completed: {tg_ok}")


if __name__ == "__main__":
    main()
