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
    video_path = EDGE_DIR / "videos" / "document_6161011834561243252 (1).mp4"
    if not video_path.exists():
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

    profile = resolve_runtime({"runtime": "cpu", "pose_engine": "yolo", "weights": "yolo11n-pose.pt", "imgsz": 960})
    models_dir = EDGE_DIR / "models"
    pose_model = load_person_pose_model(
        profile,
        models_dir=models_dir,
        weights_path="yolo11n-pose.pt",
        cfg={"pose_engine": "yolo", "weights": "yolo11n-pose.pt", "imgsz": 960},
    )
    tracker = PersonTracker(max_age=60, min_hits=3, iou_threshold=0.30)


    out_dir = EDGE_DIR / "media" / "test_tracking_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip() or "8987540090:AAEntu6IaceRsrnNl0Eow7ZrOYHBR90FkZU"
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip() or "1217328195"
    bot = TelegramOut(token=token, chat_id=chat_id)

    from role_classifier import GuardDwellTracker
    from shoe_gate import ShoeChangeMonitor

    shoe_cfg = {
        "enabled": True,
        "zone": [0.05, 0.35, 0.45, 0.95],
        "wrist_ankle_dist": 0.08,
        "confirm_seconds": 1.0,
        "cooldown_seconds": 30.0,
        "proofs": True,
    }
    shoe_monitor = ShoeChangeMonitor(shoe_cfg)
    guard_tracker = GuardDwellTracker(guard_dwell_s=10.0)

    tracked_customers = set()
    suppressed_guards = set()
    alerts_sent = 0

    print(f"[Champei CCTV] Running entrance detection & Telegram dispatch with Bodyguard Suppression...")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= min(500, total_frames):
            break
        frame_idx += 1
        t_now = time.time()

        res = pose_model.predict(frame, conf=0.30, imgsz=960, verbose=False)
        accepted, rejected, low = person_detections(
            res[0],
            frame.shape[0],
            conf_min=0.30,
            kpt_conf=0.30,
            track_low_thresh=0.15,
            return_low=True,
        )

        # Filter out tiny noise / glass reflection artifacts
        valid_accepted = []
        for d in accepted:
            dw = d.x2 - d.x1
            dh = d.y2 - d.y1
            # Real people at this camera are >= 120px tall and inside the entrance / floor area
            if dh >= 100 and dw >= 40:
                valid_accepted.append(d)

        tracks = run_identity_pipeline(frame, valid_accepted, tracker, low_detections=low)

        h, w = frame.shape[:2]
        # Check shoe change kinematics
        shoe_events = shoe_monitor.update(tracks, w, h, t_now, frame=frame)
        customer_track_ids = {ev.track_id for ev in shoe_events if getattr(ev, "customer", False)}

        for t in tracks:
            tid = t.track_id
            if tid is None or getattr(t, "coasting", False) or t.hits < 3:
                continue

            box = (t.x1, t.y1, t.x2, t.y2)
            norm_cx = (t.x1 + t.x2) / (2.0 * max(1, w))
            norm_cy = (t.y1 + t.y2) / (2.0 * max(1, h))

            # Check if this person is dwelling at the entrance / waiting bench as a bodyguard or driver
            in_waiting_zone = (norm_cx >= 0.30 and norm_cy >= 0.40)
            is_guard = guard_tracker.update(tid, box, now=t_now, in_guard_zone=in_waiting_zone)

            # Customer qualification:
            # 1. Did shoe change at rack, OR
            # 2. Entered inner doorway (norm_cx <= 0.40 and norm_cy <= 0.45) and is NOT a lingering guard
            is_customer = (tid in customer_track_ids) or (norm_cx <= 0.40 and norm_cy <= 0.45 and not is_guard)

            if is_guard and not is_customer:
                if tid not in suppressed_guards:
                    suppressed_guards.add(tid)
                    print(f"[GuardFilter] Track #{tid} (cx={norm_cx:.2f}, cy={norm_cy:.2f}) classified as Security Guard / Escort -> SUPPRESSED from Customer VIP alerts.")
                continue

            cust_id = f"CUST-{tid:04d}"

            if is_customer and cust_id not in tracked_customers:
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

                # 1. Annotate all other confirmed active people / tracks in the scene
                drawn_boxes = []
                for other_tr in tracks:
                    if other_tr.track_id is None or other_tr.track_id == tid:
                        continue
                    if getattr(other_tr, "coasting", False) or other_tr.hits < 3:
                        continue
                    o_tid = other_tr.track_id
                    ox1 = max(0, min(w - 5, int(other_tr.x1)))
                    oy1 = max(0, min(h - 5, int(other_tr.y1)))
                    ox2 = max(ox1 + 10, min(w, int(other_tr.x2)))
                    oy2 = max(oy1 + 10, min(h, int(other_tr.y2)))
                    drawn_boxes.append((ox1, oy1, ox2, oy2))

                    o_cx = (other_tr.x1 + other_tr.x2) / (2.0 * max(1, w))
                    o_cy = (other_tr.y1 + other_tr.y2) / (2.0 * max(1, h))
                    o_is_guard = (o_tid in suppressed_guards) or (o_cx >= 0.30 and o_cy >= 0.40)

                    if o_is_guard:
                        # Amber / Orange box for Bodyguards / Drivers
                        cv2.rectangle(ann, (ox1, oy1), (ox2, oy2), (0, 140, 255), 2)
                        badge_lbl = f"Guard #{o_tid} [Escort]"
                        cv2.rectangle(ann, (ox1, max(0, oy1 - 22)), (ox1 + 175, max(0, oy1)), (0, 140, 255), -1)
                        cv2.putText(ann, badge_lbl, (ox1 + 4, max(16, oy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                    else:
                        # Other customer / guest in scene -> Green box with Customer ID
                        o_cust_id = f"CUST-{o_tid:04d}"
                        o_meta = get_visitor_display_name(o_cust_id, db_path=db_path)
                        o_is_named = o_meta.get("is_named", False)
                        o_display = o_meta.get("display_name", o_cust_id)
                        o_tier = o_meta.get("tier", "Guest")
                        o_badge = f"{o_display} [{o_tier}]" if o_is_named else f"{o_cust_id} [Guest]"
                        o_bw = 200 if o_is_named else 165

                        cv2.rectangle(ann, (ox1, oy1), (ox2, oy2), (0, 220, 100), 2)
                        cv2.rectangle(ann, (ox1, max(0, oy1 - 22)), (ox1 + o_bw, max(0, oy1)), (0, 220, 100), -1)
                        cv2.putText(ann, o_badge, (ox1 + 4, max(16, oy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

                # 2. Annotate the triggering customer with prominent Green box & badge
                x1 = max(0, min(w - 5, int(t.x1)))
                y1 = max(0, min(h - 5, int(t.y1)))
                x2 = max(x1 + 10, min(w, int(t.x2)))
                y2 = max(y1 + 10, min(h, int(t.y2)))

                cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 120), 3)
                badge_title = f"{display_name} [{tier}]" if is_named else f"{cust_id} [Guest]"
                badge_w = 240 if is_named else 180
                cv2.rectangle(ann, (x1, max(0, y1 - 28)), (x1 + badge_w, max(0, y1)), (0, 255, 120), -1)
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

                # 3. Find all accompanying customers present in the scene
                accompanying_customers = []
                for other_tr in tracks:
                    if other_tr.track_id is None or other_tr.track_id == tid:
                        continue
                    o_tid = other_tr.track_id
                    o_cx = (other_tr.x1 + other_tr.x2) / (2.0 * max(1, w))
                    o_cy = (other_tr.y1 + other_tr.y2) / (2.0 * max(1, h))
                    o_is_guard = (o_tid in suppressed_guards) or (o_cx >= 0.30 and o_cy >= 0.40)
                    if not o_is_guard:
                        o_cust_id = f"CUST-{o_tid:04d}"
                        o_meta = get_visitor_display_name(o_cust_id, db_path=db_path)
                        o_is_named = o_meta.get("is_named", False)
                        o_disp = o_meta.get("display_name", o_cust_id)
                        o_tr = o_meta.get("tier", "Guest")
                        accompanying_customers.append((o_cust_id, o_disp, o_tr, o_is_named))

                snap_path = out_dir / f"champei_guest_entry_{cust_id}.jpg"
                cv2.imwrite(str(snap_path), ann, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Construct clean, rich alert for single or group arrivals
                if accompanying_customers:
                    total_group = 1 + len(accompanying_customers)
                    party_lines = [f"👤 Primary: {display_name} ({tier if is_named else 'Guest'}) [{cust_id}]"]
                    quick_name_lines = []
                    if not is_named:
                        quick_name_lines.append(f"/name {cust_id} <Name>")
                    for acc_id, acc_disp, acc_tier, acc_named in accompanying_customers:
                        party_lines.append(f"👤 Accompanying: {acc_disp} ({acc_tier if acc_named else 'Guest'}) [{acc_id}]")
                        if not acc_named:
                            quick_name_lines.append(f"/name {acc_id} <Name>")

                    party_block = "\n".join(party_lines)
                    qn_block = "\n".join(quick_name_lines) if quick_name_lines else "All customers recognized!"

                    caption = (
                        f"🌿 [champei-pp-01] 👥 GROUP ARRIVAL ({total_group} Guests)\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"{party_block}\n"
                        f"🕒 Arrival Time: {now_str}\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"💡 Staff Quick-Name:\n"
                        f"{qn_block}"
                    )
                elif is_named:
                    caption = (
                        f"🌿 [champei-pp-01] 👑 VIP ARRIVAL\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"👤 Customer: {display_name} ({tier})\n"
                        f"🔢 Customer ID: {cust_id}\n"
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
                        f"🔢 Visit Status: Verified Guest Entrance\n"
                        f"🕒 Arrival Time: {now_str}\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"💡 Staff Quick-Name: Reply with:\n"
                        f"/name {cust_id} <Guest_Name>"
                    )

                print(f"[Champei CCTV] Dispatching alert for {cust_id} ({now_str}) to Telegram...")
                ok = bot.send_photo(snap_path, caption=caption)
                print(f"[Champei CCTV] Telegram send result: {ok}")

                time.sleep(1.0)


    cap.release()
    print(f"\n[Champei CCTV] Finished! Total Customers Detected: {len(tracked_customers)}, Alerts Sent: {alerts_sent}")


if __name__ == "__main__":
    main()
