"""Ghost Counter on webcam 0 or an NVR RTSP URL.

    cd edge
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python main.py                  # preview; drag ROI, q quit, r report, o rotate, f fullscreen
    python main.py --report         # send today's report without a camera

Telegram: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID, or copy
config.example.yaml to config.yaml and fill the fields.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from paths import data_dir, get_resource_path, resource_dir


def _prepare_qt_for_opencv() -> None:
    """OpenCV 5 highgui ships Qt without fonts; GNOME Wayland often never
    creates the window, so setMouseCallback dies with NULL window handler.
    Linux-only — xcb / DejaVu paths break macOS and Windows.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    if "QT_QPA_FONTDIR" in os.environ:
        return
    for candidate in (
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/truetype/liberation",
        "/usr/share/fonts",
    ):
        if Path(candidate).is_dir():
            os.environ["QT_QPA_FONTDIR"] = candidate
            return


if sys.platform.startswith("linux"):
    _prepare_qt_for_opencv()

import cv2
import numpy as np
import yaml

ROOT = resource_dir()
DATA_DIR = data_dir()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corroborate import veto_vehicle_interior
from db import connect, has_opened_today, insert_event, upsert_minute
from face_id import till_status_label, try_create_face_recognizer
from occupancy import GhostCounter, GhostState, OccupancyGate
from person import Detection, draw_detection, person_detections
from proof import save_proof
from reid import try_create_body_reid
from report import build_report
from roi_edit import RoiEditor, draw_roi_handles
from runtime import (
    resolve_imgsz,
    resolve_kpt_conf,
    resolve_min_aspect,
    resolve_min_keypoints,
    resolve_min_person_height,
    resolve_occupy_clear_seconds,
    resolve_person_conf,
    resolve_runtime,
    resolve_weights_file,
)
from liveness import LivenessProbe
from telegram_out import TelegramOut
from tracker import PersonTracker, run_identity_pipeline
from vehicle import extract_vehicle_detections

WIN = "Inbound Garage Floor"
ROTATES = (0, 90, 180, 270)
ROTATE_CODE = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}
FIT_MAX_H = 900


def open_preview_window(win: str, on_mouse) -> None:
    """Qt highgui does not create a handle until imshow; namedWindow alone is not enough."""
    flags = cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL
    cv2.namedWindow(win, flags)
    cv2.imshow(win, np.zeros((2, 2, 3), dtype=np.uint8))
    cv2.waitKey(1)
    try:
        cv2.setMouseCallback(win, on_mouse)
    except cv2.error as exc:
        raise SystemExit(
            "Could not open the preview window (OpenCV Qt). "
            "From the edge folder run: QT_QPA_PLATFORM=xcb python main.py"
        ) from exc


def default_rotate(source: int | str) -> int:
    """Default to native orientation (0) across all camera sources."""
    return 0


def parse_rotate(value) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "auto"):
        return None
    try:
        deg = int(value)
    except (TypeError, ValueError):
        return None
    deg = ((deg % 360) + 360) % 360
    return deg if deg in ROTATES else None


def resolve_rotate(value, source: int | str) -> int:
    parsed = parse_rotate(value)
    return default_rotate(source) if parsed is None else parsed


def parse_flip(value) -> str:
    text = str(value or "none").strip().lower()
    if text in ("h", "horizontal", "x"):
        return "h"
    if text in ("v", "vertical", "y"):
        return "v"
    return "none"


def orient_frame(frame, rotate_deg: int, flip: str):
    code = ROTATE_CODE.get(rotate_deg)
    if code is not None:
        frame = cv2.rotate(frame, code)
    if flip == "h":
        frame = cv2.flip(frame, 1)
    elif flip == "v":
        frame = cv2.flip(frame, 0)
    return frame


def fit_window_size(width: int, height: int, max_h: int = FIT_MAX_H) -> tuple[int, int]:
    scale = min(1.0, max_h / max(height, 1))
    return max(1, int(width * scale)), max(1, int(height * scale))


def load_config(path: Path) -> dict:
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    token = os.environ.get("TELEGRAM_BOT_TOKEN", data.get("telegram_bot_token") or "")
    chat = os.environ.get("TELEGRAM_CHAT_ID", data.get("telegram_chat_id") or "")
    data["telegram_bot_token"] = token
    data["telegram_chat_id"] = chat
    return data


def parse_source(value) -> int | str:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


def resolve_weights(cfg: dict) -> str:
    return resolve_weights_file(cfg, get_resource_path, DATA_DIR)


def resolve_detect_fps(cfg: dict, sample_fps: float) -> float:
    if cfg.get("detect_fps") is not None:
        return max(0.5, float(cfg["detect_fps"]))
    return max(0.5, min(sample_fps, 15.0))


def draw_overlay(
    frame,
    roi_px: tuple[int, int, int, int],
    occupied: bool,
    empty_elapsed: float,
    absent_seconds: float,
    status: str | None = None,
) -> None:
    x1, y1, x2, y2 = roi_px
    color = (80, 200, 80) if occupied else (40, 180, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    draw_roi_handles(frame, roi_px, color)
    if status is None:
        status = "STAFF IN ROI" if occupied else f"EMPTY {empty_elapsed:.0f}/{absent_seconds:.0f}s"
    cv2.putText(
        frame,
        status,
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "drag box / handles / empty=new ROI",
        (12, frame.shape[0] - 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "q quit  r report  o rotate  f fullscreen",
        (12, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def send_shift_report(conn, cfg: dict, bot: TelegramOut) -> None:
    day = datetime.now().date()
    text, paths = build_report(
        conn,
        day,
        venue=str(cfg.get("garage_name") or cfg.get("venue") or "Garage"),
        open_time=str(cfg.get("open_time") or "08:00"),
        close_time=str(cfg.get("close_time") or "18:00"),
        bay_ids=[b.get("id") for b in (cfg.get("bays") or []) if isinstance(b, dict)],
    )
    print(text)
    bot.send_message(text)
    if paths:
        bot.send_album(paths, caption=f"Proof stills — {day.isoformat()}")
    else:
        print("[report] no proof stills for today")


def roi_persist_path(cfg_path: Path) -> Path:
    example = get_resource_path("config.example.yaml").resolve()
    real = DATA_DIR / "config.yaml"
    if cfg_path.resolve() == example and real.exists():
        return real
    return cfg_path


def run_camera(cfg: dict, conn, bot: TelegramOut, cfg_path: Path) -> None:
    from ultralytics import YOLO

    source = parse_source(cfg.get("source", 0))
    roi = list(cfg.get("roi") or [0.3, 0.2, 0.4, 0.6])
    absent = float(cfg.get("absent_seconds") or 10)
    cooldown = float(cfg.get("cooldown_seconds") or 30)
    sample_fps = float(cfg.get("sample_fps") or 2)
    detect_fps = resolve_detect_fps(cfg, sample_fps)
    interval = 1.0 / detect_fps
    proofs = DATA_DIR / "proofs"
    rotate_deg = resolve_rotate(cfg.get("rotate"), source)
    flip = parse_flip(cfg.get("flip"))
    person_conf = resolve_person_conf(cfg)
    min_person_height = resolve_min_person_height(cfg)
    min_aspect = resolve_min_aspect(cfg)
    min_keypoints = resolve_min_keypoints(cfg)
    kpt_conf = resolve_kpt_conf(cfg)
    profile = resolve_runtime(cfg)
    imgsz = resolve_imgsz(cfg, profile)
    confirm = float(
        cfg.get("occupy_confirm_seconds")
        if cfg.get("occupy_confirm_seconds") is not None
        else 1.0
    )
    clear = resolve_occupy_clear_seconds(cfg)

    if isinstance(source, int):
        from launcher import _open_webcam_index

        cap = _open_webcam_index(source)
        if cap is None:
            raise SystemExit(f"Cannot open video source: {source}")
    else:
        cap = cv2.VideoCapture(source)
        if isinstance(source, str) and source.startswith("rtsp"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open video source: {source}")

    editor = RoiEditor(roi, WIN, roi_persist_path(cfg_path))
    open_preview_window(WIN, editor.on_mouse)

    weights = resolve_weights(cfg)
    from rtmpose import load_person_pose_model, resolve_models_dir

    model = load_person_pose_model(
        profile,
        models_dir=resolve_models_dir(get_resource_path, DATA_DIR),
        weights_path=weights,
        cfg=cfg,
        yolo_cls=YOLO,
    )
    # Dual Neural Network Law: the COCO detector runs alongside pose so vehicle
    # boxes are available to veto skeletons hallucinated onto engines and bikes.
    try:
        vehicle_model = YOLO(str(get_resource_path("yolo11n.pt")))
    except Exception as exc:
        print(f"[Vehicle] Detector unavailable ({exc}); skipping interior veto")
        vehicle_model = None
    ghost = GhostCounter(absent, cooldown)
    gate = OccupancyGate(confirm, clear)
    face_rec = try_create_face_recognizer(cfg)
    reid = try_create_body_reid(cfg)
    probe = LivenessProbe()
    tracker = PersonTracker(
        max_age=profile.track_max_age,
        min_hits=profile.track_min_hits,
        iou_threshold=profile.track_iou_threshold,
        reid_threshold=profile.reid_match_threshold,
    )
    last_accepted: list[Detection] = []
    last_rejected: list[Detection] = []
    last_state = GhostState(False, 0.0, False)
    last_infer = 0.0
    last_wh: tuple[int, int] | None = None
    fullscreen = False

    print(f"Ingest {source}  ROI {roi}  absent>={absent}s  Telegram={'on' if bot.enabled else 'off'}")
    print(f"Orient rotate={rotate_deg} flip={flip}")
    print(
        f"Pose {weights} runtime={profile.name} device={profile.yolo_device} "
        f"conf>={person_conf} kpt>={kpt_conf} n>={min_keypoints} "
        f"aspect>={min_aspect} detect={detect_fps:.1f}fps imgsz={imgsz} "
        f"gate {confirm}s/{clear}s"
    )
    print("Drag the box or its handles; drag on empty space to draw a new ROI.")
    print("q quit  r report  o rotate  m flip  f fullscreen")
    if face_rec is not None:
        print(f"Face ID on — {len(face_rec.known_embeddings)} enrolled in {face_rec.faces_dir}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ingest] frame read failed")
            time.sleep(0.25)
            continue

        frame = orient_frame(frame, rotate_deg, flip)
        h, w = frame.shape[:2]
        editor.set_frame_size(w, h)
        if not fullscreen and (w, h) != last_wh:
            last_wh = (w, h)
            nw, nh = fit_window_size(w, h)
            cv2.resizeWindow(WIN, nw, nh)

        now = time.time()
        roi_px = editor.pixels()
        if now - last_infer >= interval:
            last_infer = now
            result = model.predict(
                frame,
                imgsz=imgsz,
                conf=person_conf,
                device=profile.yolo_device,
                verbose=False,
            )[0]
            last_accepted, last_rejected = person_detections(
                result,
                h,
                conf_min=person_conf,
                min_height_frac=min_person_height,
                min_aspect=min_aspect,
                min_keypoints=min_keypoints,
                kpt_conf=kpt_conf,
            )
            vehicles = []
            if vehicle_model is not None:
                try:
                    veh_res = vehicle_model.predict(
                        frame,
                        imgsz=imgsz,
                        classes=[2, 3, 5, 7],
                        conf=0.18,
                        device=profile.yolo_device,
                        verbose=False,
                    )[0]
                    vehicles = extract_vehicle_detections(veh_res, w, h, conf_min=0.18)
                except Exception as exc:
                    print(f"[Vehicle] Prediction error: {exc}")
            last_accepted, vetoed = veto_vehicle_interior(
                last_accepted, vehicles, kpt_conf=kpt_conf
            )
            last_rejected.extend(vetoed)
            last_accepted = run_identity_pipeline(
                frame,
                last_accepted,
                tracker,
                face_rec=face_rec,
                reid=reid,
                probe=probe,
            )
            detected = any(det.in_roi(roi_px, kpt_conf) for det in last_accepted)
            occupied = gate.update(detected, now)
            last_state = ghost.update(occupied, now)
            stamp = datetime.now()

            upsert_minute(
                conn,
                stamp.strftime("%Y-%m-%d %H:%M"),
                len(last_accepted),
                last_state.occupied,
            )
            if last_state.occupied and not has_opened_today(conn, stamp.date()):
                insert_event(conn, "opened", stamp)

            if last_state.should_alert:
                path = save_proof(frame, roi_px, stamp, proofs, kind="abandoned")
                insert_event(conn, "abandoned", stamp, str(path))
                caption = (
                    f"{cfg.get('garage_name', cfg.get('venue', 'Garage'))}: shop floor idle "
                    f"for {int(absent)}s.\n{stamp.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                print(f"[alert] {path}")
                bot.send_photo(path, caption)

        preview = frame.copy()
        for det in last_rejected:
            draw_detection(
                preview,
                det,
                in_roi=det.in_roi(roi_px, kpt_conf),
                kpt_conf=kpt_conf,
            )
        for det in last_accepted:
            draw_detection(
                preview,
                det,
                in_roi=det.in_roi(roi_px, kpt_conf),
                kpt_conf=kpt_conf,
            )
        draw_overlay(
            preview,
            roi_px,
            last_state.occupied,
            last_state.empty_elapsed,
            absent,
            status=till_status_label(
                last_state.occupied,
                [det for det in last_accepted if det.in_roi(roi_px, kpt_conf)],
                last_state.empty_elapsed,
                absent,
                face_id_enabled=face_rec is not None,
            ),
        )
        cv2.imshow(WIN, preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            send_shift_report(conn, cfg, bot)
        elif key == ord("o"):
            rotate_deg = ROTATES[(ROTATES.index(rotate_deg) + 1) % len(ROTATES)]
            last_wh = None
            print(f"[orient] rotate={rotate_deg} flip={flip}")
        elif key == ord("m"):
            flip = "none" if flip == "h" else "h"
            print(f"[orient] rotate={rotate_deg} flip={flip}")
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WIN,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )
            if not fullscreen:
                last_wh = None

    cap.release()
    cv2.destroyAllWindows()


from launcher import start_unified_server


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inbound Garage — live AI shop-floor monitor (webcam or RTSP)."
    )
    parser.add_argument(
        "--config",
        default="",
        help="YAML config path (default: edge/config.yaml then config.example.yaml)",
    )
    parser.add_argument(
        "--direct",
        "--desktop",
        action="store_true",
        help="Run standalone OpenCV desktop window instead of the web dashboard.",
    )
    parser.add_argument(
        "--source",
        default="",
        help="Directly specify or override the camera source (e.g. 0 or rtsp://...).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Send today's shift report and exit (no camera).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Preferred HTTP port for the camera hub (falls back if busy).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a system browser (used by the desktop sidecar).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else DATA_DIR / "config.yaml"
    if not cfg_path.exists():
        cfg_path = get_resource_path("config.example.yaml")

    if args.report:
        cfg = load_config(cfg_path)
        conn = connect(DATA_DIR / "events.db")
        bot = TelegramOut(cfg["telegram_bot_token"], cfg["telegram_chat_id"])
        send_shift_report(conn, cfg, bot)
        return

    # Default mode: Unified Live Web Surveillance Dashboard (Camera streams on the right)
    if not args.direct and not args.source:
        start_unified_server(port=args.port, open_browser=not args.no_browser)
        return

    # Standalone OpenCV Desktop Mode
    cfg = load_config(cfg_path)
    if args.source:
        cfg["source"] = args.source

    conn = connect(DATA_DIR / "events.db")
    bot = TelegramOut(cfg.get("telegram_bot_token", ""), cfg.get("telegram_chat_id", ""))
    run_camera(cfg, conn, bot, cfg_path)


if __name__ == "__main__":
    main()


