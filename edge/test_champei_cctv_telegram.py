#!/usr/bin/env python3
"""Champei Spa CCTV Front Door Customer Entrance Tracker & Telegram Alert.

Processes real front-door CCTV video, captures high-resolution arrival photos,
generates clean Customer IDs (CUST-0001, CUST-0002...), formats arrival timestamps,
and sends rich Telegram alerts with the photo.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

EDGE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = EDGE_DIR.parent
if str(EDGE_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE_DIR))

from person import person_detections
from rtmpose import load_person_pose_model
from runtime import resolve_runtime
from telegram_out import TelegramOut
from tracker import PersonTracker, run_identity_pipeline


def main():
    video_path = WORKSPACE_DIR / "tools" / "virtual-camera" / "videos" / "document_6161011834561243252 (1).mp4"
    if not video_path.exists():
        video_path = Path("/home/ratanakvichea/Downloads/document_6161011834561243252 (1).mp4")

    print(f"[Champei CCTV] Loading video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Champei CCTV] Failed to open: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0

    profile = resolve_runtime({"runtime": "cpu", "pose_engine": "rtmpose", "rtmpose_mode": "lightweight"})
    models_dir = EDGE_DIR / "models"
    pose_model = load_person_pose_model(
        profile,
        models_dir=models_dir,
        weights_path="yolo11n-pose.pt",
        cfg={"pose_engine": "rtmpose", "rtmpose_mode": "lightweight"},
    )
    tracker = PersonTracker(max_age=45, min_hits=3, iou_threshold=0.3)

    out_dir = EDGE_DIR / "media" / "test_tracking_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip() or "8987540090:AAEntu6IaceRsrnNl0Eow7ZrOYHBR90FkZU"
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip() or "1217328195"
    bot = TelegramOut(token=token, chat_id=chat_id)

    tracked_customers = set()
    alerts_sent = 0

    print(f"[Champei CCTV] Running entrance detection & Telegram dispatch...")

    for i in range(min(500, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        res = pose_model.predict(frame)
        accepted, rejected, low = person_detections(
            res[0],
            frame.shape[0],
            conf_min=0.25,
            kpt_conf=0.25,
            return_low=True,
        )
        tracks = run_identity_pipeline(frame, accepted, tracker, low_detections=low)

        for t in tracks:
            tid = t.track_id
            if tid is None:
                continue

            cust_id = f"CUST-{tid:04d}"

            if cust_id not in tracked_customers:
                tracked_customers.add(cust_id)
                alerts_sent += 1

                # Look up customer in visitor_registry
                from visitor_registry import get_visitor_display_name
                db_path = EDGE_DIR / "events.db"
                meta = get_visitor_display_name(cust_id, db_path=db_path)

                is_named = meta.get("is_named", False)
                display_name = meta.get("display_name", cust_id)
                tier = meta.get("tier", "Standard")
                notes = meta.get("notes", "")

                now_str = datetime.now().strftime("%I:%M %p")
                ann = frame.copy()
                x1 = max(0, int(t.x1))
                y1 = max(0, int(t.y1))
                x2 = min(frame.shape[1], int(t.x2))
                y2 = min(frame.shape[0], int(t.y2))

                # Green box + badge
                cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 120), 2)
                badge_title = f"{display_name} [{tier}]" if is_named else f"{cust_id} [Visit #1]"
                cv2.rectangle(ann, (x1, max(0, y1 - 28)), (x1 + 260, max(0, y1)), (0, 255, 120), -1)
                cv2.putText(
                    ann,
                    badge_title,
                    (x1 + 6, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                snap_path = out_dir / f"champei_guest_entry_{cust_id}.jpg"
                cv2.imwrite(str(snap_path), ann, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Construct clean, rich alert
                if is_named:
                    caption = (
                        f"🌿 [champei-pp-01] 👑 VIP ARRIVAL\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"👤 Customer: {display_name} ({tier})\n"
                        f"🔢 Customer ID: {cust_id} (Visit #2)\n"
                        f"🕒 Arrival Time: {now_str}\n"
                        f"📝 Notes: {notes or 'Preferred client'}\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"✨ Welcome back {display_name}!"
                    )
                else:
                    caption = (
                        f"🌿 [champei-pp-01] 👋 GUEST ARRIVAL\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"👤 Customer ID: {cust_id}\n"
                        f"🔢 Visit Count: Visit #1 (New Guest)\n"
                        f"🕒 Arrival Time: {now_str}\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"💡 Staff Quick-Name: Reply with:\n"
                        f"/name {cust_id} <Guest_Name>"
                    )

                print(f"[Champei CCTV] Dispatching alert for {cust_id} ({now_str}) to Telegram...")
                ok = bot.send_photo(snap_path, caption=caption)
                print(f"[Champei CCTV] Telegram send result: {ok}")

                # Rate-limit between alerts so Telegram doesn't throttle
                time.sleep(1.0)

    cap.release()
    print(f"\n[Champei CCTV] Finished! Total Customers Detected: {len(tracked_customers)}, Alerts Sent: {alerts_sent}")


if __name__ == "__main__":
    main()
