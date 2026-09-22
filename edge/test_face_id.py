"""Standalone Local Face Recognition Test & Enrollment Tool.

Usage:
    python test_face_id.py              # Test with default webcam 0
    python test_face_id.py --source 1   # Test with secondary camera
    python test_face_id.py --enroll "John"  # Capture and enroll a face directly

Hotkeys in live window:
    - [e] : Enter staff name in terminal to enroll the current face
    - [r] : Reload enrolled faces from the faces/ directory
    - [q] : Quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_id import FaceRecognizer
from paths import data_dir

DATA = data_dir()


def main():
    parser = argparse.ArgumentParser(description="Test Local Face Recognition")
    parser.add_argument("--source", default=0, help="Camera index (0) or RTSP / video file")
    parser.add_argument("--faces-dir", default=str(DATA / "faces"), help="Directory storing enrolled faces")
    parser.add_argument("--models-dir", default=str(DATA / "models"), help="Directory for ONNX models")
    parser.add_argument("--threshold", type=float, default=0.60, help="Cosine similarity threshold (0.0 - 1.0)")
    parser.add_argument("--enroll", type=str, default=None, help="Directly enroll a staff name from live camera")
    args = parser.parse_args()

    # Parse source as int if digit
    source = int(args.source) if str(args.source).isdigit() else args.source

    print("[*] Initializing FaceRecognizer...")
    face_rec = FaceRecognizer(
        faces_dir=args.faces_dir,
        models_dir=args.models_dir,
        match_threshold=args.threshold,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video source {source}")
        return

    win_name = "Local Face Recognition Test (Press 'e' to enroll, 'r' to reload, 'q' to quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    print("\n" + "=" * 60)
    print(" Local Face Recognition Ready (0 Tokens / 100% Offline)")
    print(" Controls:")
    print("   [e] : Enroll face under a new staff name")
    print("   [r] : Reload faces folder")
    print("   [q] : Quit")
    print("=" * 60 + "\n")

    fps_time = time.time()
    fps_count = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame.")
            break

        fps_count += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps = fps_count / (now - fps_time)
            fps_count = 0
            fps_time = now

        # Run face recognition on the frame
        match = face_rec.recognize_in_crop(frame)

        # Draw results on frame
        if match.bbox is not None:
            fx, fy, fw, fh = match.bbox
            color = (0, 255, 0) if match.is_staff else (0, 165, 255)
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)

            label = f"{match.name} ({match.confidence:.2f})"
            # Background pill for text readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (fx, fy - th - 8), (fx + tw + 6, fy), color, -1)
            cv2.putText(
                frame,
                label,
                (fx + 3, fy - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0) if match.is_staff else (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Draw HUD stats
        hud_text = f"FPS: {fps:.1f} | Enrolled Staff: {len(face_rec.known_embeddings)} | Threshold: {args.threshold}"
        cv2.putText(frame, hud_text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            count = face_rec.reload_enrolled_faces()
            print(f"[*] Reloaded database: {count} staff enrolled.")
        elif key == ord("e"):
            # Interactive enrollment
            cv2.putText(frame, "SAVING FACE... Enter name in terminal", (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(win_name, frame)
            cv2.waitKey(1)

            name = input("\n[Enroll] Enter employee name (e.g. Alice): ").strip()
            if name:
                person_folder = Path(args.faces_dir) / name
                person_folder.mkdir(parents=True, exist_ok=True)
                existing = len(list(person_folder.glob("*.jpg")))
                save_path = person_folder / f"{existing + 1}.jpg"
                cv2.imwrite(str(save_path), frame)
                print(f"[+] Saved face snapshot to: {save_path}")
                face_rec.reload_enrolled_faces()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
