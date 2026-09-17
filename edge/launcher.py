"""Inbound Garage — live AI auto-shop operations hub.

Serves the garage command-center dashboard (edge/hub.html):
- Live view with YOLO11 pose overlay and multi-bay service ROIs.
- Face ID attendance, wrench-time classification, Wi-Fi presence.
- End-of-day scorecards for Telegram and the owner dashboard.
"""

from __future__ import annotations

import argparse
import json
import os

# Pin OpenMP and MKL threads globally to avoid thread explosion
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from paths import (
    INBOUND_BUILD_ID,
    data_dir,
    fatal_boot,
    get_resource_path,
    is_frozen,
    log_boot_banner,
    resource_dir,
)

# Default FFmpeg RTSP options for any leftover OpenCV opens (CLI preview).
# The grabber's RTSPAdapter overrides per-attempt with a 2s stimeout.
# Set before importing cv2 / adapters so FFmpeg picks the timeout up.
_RTSP_STIMEOUT_US = 2_000_000
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    f"rtsp_transport;tcp|stimeout;{_RTSP_STIMEOUT_US}|max_delay;500000",
)
if sys.platform.startswith("linux"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

ROOT = resource_dir()
DATA_DIR = data_dir()
VIDEOS_DIR = DATA_DIR / "videos"
log_boot_banner()


def resolve_static_asset(url_path: str) -> Path | None:
    """Resolve ``/static/...`` to a bundled file. Drops ``..`` so Windows
    mixed separators cannot walk out of the static folder.
    """
    if not url_path.startswith("/static/"):
        return None
    rel = urllib.parse.unquote(url_path[len("/static/") :]).replace("\\", "/")
    parts = [part for part in rel.split("/") if part and part not in (".", "..")]
    if not parts:
        return None
    candidates = [
        get_resource_path(str(Path("static", *parts))),
        DATA_DIR.joinpath("static", *parts),
    ]
    if not is_frozen():
        candidates.append(ROOT.parent.joinpath("static", *parts))
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    return None


try:
    import cv2
    import numpy as np
    import requests
    import yaml
except Exception as _boot_err:
    if __name__ == "__main__":
        fatal_boot(_boot_err)
    raise

import re


DEFAULT_SUPABASE_URL = "https://rmepwjywobowktdmkusu.supabase.co"
DEFAULT_SUPABASE_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJtZXB3anl3b2Jvd2t0ZG1rdXN1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODg3NDA0MDYsImV4cCI6MjEwNDMxNjQwNn0."
    "u_hg9uXPSK7AE1kD3ol9LUmYgyddVWrELACWh5Z9zr4"
)
DEFAULT_TELEGRAM_BOT_TOKEN = "8987540090:AAEntu6IaceRsrnNl0Eow7ZrOYHBR90FkZU"


def load_dotenv_files() -> None:
    """Load KEY=VALUE pairs from .env without overriding a real environment."""
    candidates = [
        DATA_DIR / ".env",
        Path(sys.executable).parent / ".env",
        Path.cwd() / ".env",
        ROOT / ".env",
        ROOT.parent / ".env",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv_files()


def open_host_url(url: str) -> bool:
    """Open a URL / custom protocol with the host OS (Windows, macOS, Linux)."""
    target = str(url or "").strip()
    if not target:
        return False
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform.startswith("win"):
            os.startfile(target)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as exc:
        print(f"[telegram] host open failed: {exc}", flush=True)
        return False


def public_supabase_config() -> dict[str, Any]:
    url = (
        os.environ.get("VITE_SUPABASE_URL")
        or os.environ.get("SUPABASE_URL")
        or DEFAULT_SUPABASE_URL
    ).strip()
    key = (
        os.environ.get("VITE_SUPABASE_ANON_KEY")
        or os.environ.get("VITE_SUPABASE_PUBLISHABLE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or DEFAULT_SUPABASE_ANON_KEY
    ).strip()
    return {
        "supabaseUrl": url,
        "supabaseAnonKey": key,
        "configured": bool(url and key),
    }


def init_videos_dir() -> Path:
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    target = VIDEOS_DIR / "sample_garage_demo.mp4"
    sample = None
    if not is_frozen():
        sample = ROOT.parent / "tools" / "virtual-camera" / "videos" / "sample_garage_demo.mp4"
    if not target.exists() and sample is not None and sample.is_file():
        try:
            target.symlink_to(sample.resolve())
        except Exception:
            try:
                import shutil
                shutil.copy2(sample, target)
            except Exception:
                pass
    return VIDEOS_DIR


init_videos_dir()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from adapters import create_adapter, ingest_kind
    from adapters.base import (
        BaseCameraAdapter,
        FramePacket,
        parse_source,
        protocol_from_source,
        redact_source,
        unwrap_local_video_source,
        webcam_index,
    )
    from adapters.onvif import onvif_xaddr
    from adapters.webcam import open_webcam_index as _open_webcam_index
    from capture import AsyncFrameGrabber, request_still
    from discovery import DiscoveryEngine
    from media.go2rtc import Go2RtcManager, sanitize_stream_id
    from db import (
        add_wifi_minutes,
        close_bay_sessions,
        close_empty_bays,
        complete_vehicle_job,
        connect,
        customer_visit_counts,
        get_daily_garage_summary,
        get_or_create_vehicle_job,
        get_recent_ai_audits,
        get_recent_complaints,
        get_vehicle_job_history,
        has_opened_today,
        insert_customer_complaint,
        insert_event,
        list_vehicle_jobs,
        record_ai_audit_verdict,
        record_face_clock_in,
        record_face_clock_out,
        update_complaint_telegram_status,
        update_technician_activity,
        update_vehicle_job_activity,
        upsert_minute,
    )
    from complaint_service import ComplaintMonitoringService
    from face_id import (
        create_identity,
        delete_identity,
        delete_identity_photo,
        get_identity,
        identity_photo_path,
        list_identities,
        save_identity_photo,
        till_status_label,
        try_create_face_recognizer,
    )
    from ai_auditor import AIAuditorQueue, AIAuditVerdict, TokenSaverGate
    from occupancy import (
        DEFAULT_BAYS,
        UNKNOWN_WORKER,
        BayZoneManager,
        GhostCounter,
        GhostState,
        detection_in_bay,
        normalize_bays,
        roi_to_pixels,
    )
    from workplaces import normalize_workplace_zones, parse_workplace_id
    from workplaces.customer_visits import CustomerVisitMonitor
    from bay_zoom import occupancy_hints, zoom_empty_bays
    from corroborate import veto_vehicle_interior
    from liveness import LivenessProbe
    from negatives import bank_hard_negatives
    from person import Detection, draw_detection, person_detections, person_detections_split
    from proof import save_proof, scale_roi_px
    from reid import PersistentReIDGallery, try_create_body_reid
    from report import build_report
    from runtime import (
        resolve_bay_zoom,
        resolve_bay_zoom_pad,
        resolve_imgsz,
        resolve_kpt_conf,
        resolve_min_aspect,
        resolve_min_keypoints,
        resolve_min_person_height,
        resolve_occupy_clear_seconds,
        resolve_person_conf,
        resolve_runtime,
        resolve_under_car_grace_seconds,
        resolve_weights_file,
    )
    from sensors.wifi_tracker import WifiTracker, normalize_wifi_devices, presence_status
    from service_patterns import KNOWLEDGE_BASE, evaluate_completed_vehicle_job
    from telegram_link import TelegramLinkService
    from telegram_out import TelegramOut, normalize_chat_id
    from tracker import PersonTracker, run_identity_pipeline
    from vehicle import VehicleDetection, extract_vehicle_detections
except Exception as _boot_err:
    if __name__ == "__main__":
        fatal_boot(_boot_err)
    raise


def find_free_port(default_port: int = 8765) -> int:
    for attempt in range(5):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", default_port))
                return default_port
        except OSError:
            if attempt < 4:
                time.sleep(0.3)
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Map legacy Whisper size strings -> local fine-tuned Khmer model ids.
_KHMER_STT_MODEL_BY_SIZE = {
    "base": "sengtha/whisper-base-khmer",
    "sengtha/whisper-base-khmer": "sengtha/whisper-base-khmer",
}


def resolve_khmer_stt_model(value: str) -> str:
    """Resolve a config STT model value to a loadable model identifier.

    Accepts a full HuggingFace model id (pass-through) or a legacy size
    ("base", "small", ...) that is mapped to the fine-tuned Khmer model.
    """
    text = str(value or "").strip().lower()
    if not text:
        return "sengtha/whisper-base-khmer"
    if text in _KHMER_STT_MODEL_BY_SIZE:
        return _KHMER_STT_MODEL_BY_SIZE[text]
    for key in ("base", "small", "medium", "large"):
        if text == key or text == f"whisper-{key}":
            return f"sengtha/whisper-{key}-khmer"
    return str(value).strip()


def get_config_path() -> Path:
    real = DATA_DIR / "config.yaml"
    if real.exists():
        return real
    example = get_resource_path("config.example.yaml")
    if example.exists():
        return example
    return real


def read_config() -> dict[str, Any]:
    path = get_config_path()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    token = (
        os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        or str(data.get("telegram_bot_token") or "").strip()
        or DEFAULT_TELEGRAM_BOT_TOKEN
    )
    chat = normalize_chat_id(os.environ.get("TELEGRAM_CHAT_ID", data.get("telegram_chat_id") or ""))
    data["telegram_bot_token"] = token
    data["telegram_chat_id"] = chat
    if os.environ.get("TELEGRAM_BOT_TOKEN", "").strip():
        data["telegram_bot_token"] = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    data["telegram_bot_configured"] = bool(data["telegram_bot_token"])
    data["telegram_bot_username"] = str(data.get("telegram_bot_username") or "")

    data["workplace_type"] = parse_workplace_id(data.get("workplace_type"))
    data["cameras"] = _normalize_cameras(data.get("cameras"), workplace=data["workplace_type"])
    data["active_camera_id"] = str(data.get("active_camera_id") or "")
    data["garage_name"] = str(data.get("garage_name") or data.get("venue") or "Demo Garage")
    data["venue"] = str(data.get("venue") or data["garage_name"])
    data["close_time"] = str(data.get("close_time") or "18:00")
    active_cid = data["active_camera_id"]
    active_cam = next((c for c in data["cameras"] if c["id"] == active_cid), None)
    if active_cam and active_cam.get("bays"):
        data["bays"] = active_cam["bays"]
    else:
        fallback_roi = parse_roi(active_cam.get("roi")) if active_cam else parse_roi(data.get("roi"))
        data["bays"] = parse_bays(
            data.get("bays"),
            fallback_roi=fallback_roi,
            workplace=data["workplace_type"],
        )
        if active_cam and not active_cam.get("bays"):
            active_cam["bays"] = data["bays"]
    data["wifi_devices"] = normalize_wifi_devices(data.get("wifi_devices"))
    hours = data.get("operating_hours")
    if not isinstance(hours, dict):
        data["operating_hours"] = {
            "open": str(data.get("open_time") or "08:00"),
            "close": str(data.get("close_time") or "18:00"),
        }
    else:
        data["operating_hours"] = {
            "open": str(hours.get("open") or data.get("open_time") or "08:00"),
            "close": str(hours.get("close") or data.get("close_time") or "18:00"),
        }
        data["open_time"] = data["operating_hours"]["open"]
        data["close_time"] = data["operating_hours"]["close"]
    return data


_SETTINGS_KEYS = (
    "telegram_bot_token",
    "telegram_chat_id",
    "venue",
    "garage_name",
    "workplace_type",
    "open_time",
    "close_time",
    "absent_seconds",
    "cooldown_seconds",
    "auto_create_bays",
)


def save_config(updates: dict[str, Any]) -> dict[str, Any]:
    target = DATA_DIR / "config.yaml"
    current = {}
    if target.exists():
        try:
            with target.open("r", encoding="utf-8") as f:
                current = yaml.safe_load(f) or {}
        except Exception:
            current = {}
    else:
        example = get_resource_path("config.example.yaml")
        if example.exists():
            try:
                with example.open("r", encoding="utf-8") as f:
                    current = yaml.safe_load(f) or {}
            except Exception:
                current = {}

    current.update(updates)
    if "cameras" in current:
        current["cameras"] = _normalize_cameras(
            current.get("cameras"),
            workplace=current.get("workplace_type"),
        )
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(current, f, sort_keys=False)
    return current


def _normalize_xaddrs(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _normalize_cameras(raw: Any, workplace: Any = None) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id") or "").strip()
        name = str(item.get("name") or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        source = item.get("source", "")
        if not isinstance(source, int):
            source = str(source or "")
        main_source = item.get("main_source", "")
        if not isinstance(main_source, int):
            main_source = str(main_source or "")
        cam_roi = parse_roi(item.get("roi")) or [0.30, 0.20, 0.40, 0.60]
        raw_bays = item.get("bays")
        if raw_bays and isinstance(raw_bays, list):
            cam_bays = parse_bays(
                raw_bays,
                fallback_roi=cam_roi,
                seed_if_empty=False,
                workplace=workplace,
            )
        else:
            cam_bays = []
        out.append(
            {
                "id": cid,
                "name": name or "Untitled camera",
                "source": source,
                "main_source": main_source,
                "protocol": str(item.get("protocol") or protocol_from_source(source)),
                "vendor": str(item.get("vendor") or "generic"),
                "username": str(item.get("username") or ""),
                "xaddrs": _normalize_xaddrs(item.get("xaddrs")),
                "rotate": item.get("rotate", "auto") if str(item.get("rotate", "")).lower() == "auto" else item.get("rotate", 0),
                "flip": str(item.get("flip") or "none"),
                "roi": cam_roi,
                "bays": cam_bays,
                "enabled": bool(item.get("enabled", True)),
                "trigger_mode": str(item.get("trigger_mode") or "roi_state_change"),
            }
        )
    return out


def upsert_camera(cfg: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    cameras = _normalize_cameras(cfg.get("cameras"), workplace=cfg.get("workplace_type"))
    cid = str(fields.get("id") or "").strip() or f"cam-{int(time.time() * 1000)}"
    name = str(fields.get("name") or "").strip() or "Untitled camera"
    existing: dict[str, Any] = {}
    for cam in cameras:
        if cam["id"] == cid:
            existing = cam
            break
    source = fields.get("source", existing.get("source", cfg.get("source", "")))
    if not isinstance(source, int):
        source = str(source or "")
    if not str(source).strip() and fields.get("xaddrs"):
        xaddrs_fallback = _normalize_xaddrs(fields.get("xaddrs"))
        if xaddrs_fallback:
            source = xaddrs_fallback[0]
    if "main_source" in fields:
        main_source = fields.get("main_source", "")
    else:
        main_source = existing.get("main_source", "")
    if not isinstance(main_source, int):
        main_source = str(main_source or "")
    if "xaddrs" in fields:
        xaddrs = _normalize_xaddrs(fields.get("xaddrs"))
    else:
        xaddrs = _normalize_xaddrs(existing.get("xaddrs"))
    roi = parse_roi(fields.get("roi"))
    if roi is None and existing:
        roi = parse_roi(existing.get("roi"))
    if roi is None:
        roi = [0.30, 0.20, 0.40, 0.60]
    rotate = fields.get("rotate", existing.get("rotate", cfg.get("rotate", "auto")))
    bays_val = fields.get("bays") if "bays" in fields else existing.get("bays")
    if bays_val and isinstance(bays_val, list):
        bays = parse_bays(bays_val, fallback_roi=roi, seed_if_empty=False)
    else:
        bays = list(existing.get("bays") or [])
    trigger_mode = str(
        fields.get("trigger_mode")
        or existing.get("trigger_mode")
        or "roi_state_change"
    )
    entry = {
        "id": cid,
        "name": name,
        "source": source,
        "main_source": main_source,
        "protocol": str(fields.get("protocol") or existing.get("protocol") or protocol_from_source(source)),
        "vendor": str(fields.get("vendor") or existing.get("vendor") or "generic"),
        "username": str(fields.get("username") if "username" in fields else existing.get("username") or ""),
        "xaddrs": xaddrs,
        "rotate": rotate,
        "flip": str(fields.get("flip") or existing.get("flip") or cfg.get("flip") or "none"),
        "roi": roi,
        "bays": bays,
        "enabled": bool(fields.get("enabled", existing.get("enabled", True))),
        "ml_enabled": bool(fields.get("ml_enabled", existing.get("ml_enabled", True))),
        "trigger_mode": trigger_mode,
    }
    for i, cam in enumerate(cameras):
        if cam["id"] == cid:
            cameras[i] = entry
            break
    else:
        cameras.append(entry)
    cfg["cameras"] = cameras
    cfg["active_camera_id"] = cid
    cfg["roi"] = roi
    if bays:
        cfg["bays"] = bays
    return entry


def _box_intersects_roi(
    box: tuple[int, int, int, int], roi: list[float], frame_w: int, frame_h: int
) -> bool:
    try:
        x1, y1, x2, y2 = box
        bx1 = int(roi[0] * frame_w)
        by1 = int(roi[1] * frame_h)
        bx2 = int((roi[0] + roi[2]) * frame_w)
        by2 = int((roi[1] + roi[3]) * frame_h)
        ix1 = max(x1, bx1)
        iy1 = max(y1, by1)
        ix2 = min(x2, bx2)
        iy2 = min(y2, by2)
        if ix2 > ix1 and iy2 > iy1:
            inter = (ix2 - ix1) * (iy2 - iy1)
            area = max(1.0, float((x2 - x1) * (y2 - y1)))
            return (inter / area) >= 0.2
        return False
    except Exception:
        return False


def parse_rotate(value: Any) -> int:
    if value is None or str(value).strip().lower() in ("", "auto"):
        return 0
    try:
        deg = int(value)
        deg = ((deg % 360) + 360) % 360
        return deg if deg in (0, 90, 180, 270) else 0
    except (TypeError, ValueError):
        return 0


def is_auto_rotate(value: Any) -> bool:
    return value is None or str(value).strip().lower() in ("", "auto")


def suggest_rotate(source: Any, frame) -> int:
    """Return default rotation for auto-orient.

    Default to 0 (native camera orientation) so streams are not unexpectedly
    rotated sideways. Users can select CW (90°), CCW (270°), or 180° for
    mounted phone orientations.
    """
    return 0


def resolve_orient(
    rotate_value: Any, source: Any = None, frame=None, flip_value: Any = "none"
) -> tuple[int, str]:
    rot = suggest_rotate(source, frame) if is_auto_rotate(rotate_value) else parse_rotate(rotate_value)
    return rot, parse_flip(flip_value)


def parse_flip(value: Any) -> str:
    text = str(value or "none").strip().lower()
    if text in ("both", "hv", "vh", "all"):
        return "both"
    if text in ("h", "horizontal", "x"):
        return "h"
    if text in ("v", "vertical", "y"):
        return "v"
    return "none"


def orient_frame(frame, rotate_deg: int, flip: str):
    if rotate_deg == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotate_deg == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotate_deg == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    flp = parse_flip(flip)
    if flp == "both":
        frame = cv2.flip(frame, -1)
    elif flp == "h":
        frame = cv2.flip(frame, 1)
    elif flp == "v":
        frame = cv2.flip(frame, 0)
    return frame


def transform_point_hflip(x: float, y: float) -> tuple[float, float]:
    return round(max(0.0, min(1.0, 1.0 - x)), 4), round(max(0.0, min(1.0, y)), 4)


def transform_point_vflip(x: float, y: float) -> tuple[float, float]:
    return round(max(0.0, min(1.0, x)), 4), round(max(0.0, min(1.0, 1.0 - y)), 4)


def transform_point_rotate(x: float, y: float, deg: int) -> tuple[float, float]:
    d = ((deg % 360) + 360) % 360
    if d == 90:
        return round(max(0.0, min(1.0, 1.0 - y)), 4), round(max(0.0, min(1.0, x)), 4)
    elif d == 180:
        return round(max(0.0, min(1.0, 1.0 - x)), 4), round(max(0.0, min(1.0, 1.0 - y)), 4)
    elif d == 270:
        return round(max(0.0, min(1.0, y)), 4), round(max(0.0, min(1.0, 1.0 - x)), 4)
    return round(max(0.0, min(1.0, x)), 4), round(max(0.0, min(1.0, y)), 4)


def transform_bay_geometry(bay: dict[str, Any], transform_fn) -> dict[str, Any]:
    b = dict(bay)
    poly = b.get("polygon")
    if isinstance(poly, list) and len(poly) >= 3:
        new_poly = [list(transform_fn(float(p[0]), float(p[1]))) for p in poly]
        b["polygon"] = new_poly
        xs = [p[0] for p in new_poly]
        ys = [p[1] for p in new_poly]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        b["roi"] = [round(min_x, 4), round(min_y, 4), round(max(0.02, max_x - min_x), 4), round(max(0.02, max_y - min_y), 4)]
    elif isinstance(b.get("roi"), (list, tuple)) and len(b["roi"]) == 4:
        x, y, w, h = (float(v) for v in b["roi"])
        p1 = transform_fn(x, y)
        p2 = transform_fn(x + w, y)
        p3 = transform_fn(x + w, y + h)
        p4 = transform_fn(x, y + h)
        xs = [p1[0], p2[0], p3[0], p4[0]]
        ys = [p1[1], p2[1], p3[1], p4[1]]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        b["roi"] = [round(min_x, 4), round(min_y, 4), round(max(0.02, max_x - min_x), 4), round(max(0.02, max_y - min_y), 4)]
        if poly is not None:
            b["polygon"] = [list(p1), list(p2), list(p3), list(p4)]
    return b




def parse_roi(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x, y, w, h = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    w = max(0.02, min(1.0, w))
    h = max(0.02, min(1.0, h))
    x = max(0.0, min(1.0 - w, x))
    y = max(0.0, min(1.0 - h, y))
    return [x, y, w, h]


def parse_bays(
    value: Any,
    fallback_roi: list[float] | None = None,
    *,
    seed_if_empty: bool = True,
    workplace: Any = None,
) -> list[dict[str, Any]]:
    if workplace is not None and parse_workplace_id(workplace) != "garage":
        return normalize_workplace_zones(
            workplace, value, fallback_roi=fallback_roi, seed_if_empty=seed_if_empty
        )
    return normalize_bays(value, fallback_roi=fallback_roi, seed_if_empty=seed_if_empty)


def parse_clock(value: Any, default: str = "08:00") -> str:
    text = str(value or default).strip()
    parts = text.split(":")
    try:
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            return default
        return f"{hour:02d}:{minute:02d}"
    except (TypeError, ValueError, IndexError):
        return default


def shop_is_open(cfg: dict[str, Any], now: datetime | None = None) -> bool:
    stamp = now or datetime.now()
    hours = cfg.get("operating_hours") if isinstance(cfg.get("operating_hours"), dict) else {}
    open_s = parse_clock(hours.get("open") or cfg.get("open_time"), "08:00")
    close_s = parse_clock(hours.get("close") or cfg.get("close_time"), "18:00")
    current = stamp.strftime("%H:%M")
    if open_s <= close_s:
        return open_s <= current < close_s
    return current >= open_s or current < close_s


def garage_name_of(cfg: dict[str, Any]) -> str:
    return str(cfg.get("garage_name") or cfg.get("venue") or "Demo Garage")


def _needs_onvif_onboard(protocol: Any, source: Any, xaddrs: Any) -> bool:
    if str(protocol or "").strip().lower() != "onvif":
        return False
    text = str(source or "").strip().lower()
    if text.startswith("rtsp://"):
        return False
    if _normalize_xaddrs(xaddrs) or "onvif" in text or text.startswith("http://") or text.startswith("https://"):
        return True
    return False


class _ConnectErrorAdapter(BaseCameraAdapter):
    """Queued after a failed onboard so telemetry becomes FAILED without crashing."""

    def __init__(self, message: str):
        self.error: str | None = message

    def connect(self) -> bool:
        return False

    def read_frame(self) -> FramePacket | None:
        return None

    def release(self) -> None:
        return None

    def is_connected(self) -> bool:
        return False


class CameraStreamWorker:
    """Maintains a background grabber and latest-frame cache for a single camera.

    Keeps the camera stream alive across layout switches and active-camera changes,
    enabling all cameras in the multi-camera grid to stream simultaneously.
    """

    def __init__(
        self,
        camera_id: str,
        camera_cfg: dict[str, Any],
        gateway: Any = None,
        eval_callback: Any = None,
    ):
        self.camera_id = str(camera_id)
        self.cfg = dict(camera_cfg)
        self.gateway = gateway
        self.eval_callback = eval_callback
        self.grabber = AsyncFrameGrabber()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.latest_jpeg: bytes | None = None
        self.latest_ts: float = 0.0
        self.latest_packet: FramePacket | None = None
        self.is_active_ai: bool = False
        self._prev_gray: np.ndarray | None = None
        self._last_bg_eval: float = 0.0
        self._bg_eval_pending: bool = False
        self._bg_eval_frame: np.ndarray | None = None

    def update_cfg(self, new_cfg: dict[str, Any]) -> bool:
        """Update config. Reconnects adapter if source, credentials, or protocol changed."""
        with self._lock:
            old_src = self.cfg.get("source")
            old_proto = self.cfg.get("protocol")
            old_user = self.cfg.get("username")
            old_pass = self.cfg.get("password")
            old_main = self.cfg.get("main_source")
            self.cfg.update(new_cfg)
            need_reconnect = (
                old_src != self.cfg.get("source")
                or old_proto != self.cfg.get("protocol")
                or old_user != self.cfg.get("username")
                or old_pass != self.cfg.get("password")
                or old_main != self.cfg.get("main_source")
            )
        if need_reconnect and self._thread and self._thread.is_alive():
            self._reconnect()
            return True
        return False

    def _build_adapter(self) -> BaseCameraAdapter:
        src = self.cfg.get("source")
        proto = self.cfg.get("protocol")
        user = str(self.cfg.get("username") or "")
        pwd = str(self.cfg.get("password") or "")
        xaddrs = self.cfg.get("xaddrs") or []
        main_src = str(self.cfg.get("main_source") or "")
        stream_id = sanitize_stream_id(self.camera_id)
        return create_adapter(
            src,
            gateway=self.gateway if stream_id else None,
            stream_id=stream_id,
            protocol=proto,
            username=user,
            password=pwd,
            xaddrs=xaddrs,
            main_source=main_src,
        )

    def _reconnect(self) -> None:
        try:
            adapter = self._build_adapter()
            self.grabber.switch_source(adapter)
        except Exception as exc:
            print(f"[CameraStreamWorker:{self.camera_id}] Reconnect error: {exc}", flush=True)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.grabber.start()
        self._reconnect()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"CamWorker-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.grabber.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None

    def _loop(self) -> None:
        last_encode_time = 0.0
        last_reconnect_time = time.time()
        last_packet_ts = 0.0
        while not self._stop_event.is_set():
            now = time.time()
            if (
                self.grabber.connection_state in ("FAILED", "STANDBY", "RECONNECTING")
                or getattr(self.grabber, "_adapter", None) is None
                or (now - self.latest_ts >= 3.5 and self.grabber.connection_state != "CONNECTED")
            ) and (now - last_reconnect_time >= 2.5):
                last_reconnect_time = now
                self._reconnect()

            packet = self.grabber.peek_latest_frame()
            if packet is None:
                time.sleep(0.04)
                continue
            if packet.frame is None:
                time.sleep(0.04)
                continue
            pkt_ts = float(getattr(packet, "timestamp", 0.0) or 0.0)
            if pkt_ts and pkt_ts == last_packet_ts:
                time.sleep(0.04)
                continue
            try:
                self.grabber.output_rotate = int(self.cfg.get("rotate") or 0)
            except (ValueError, TypeError):
                self.grabber.output_rotate = 0
            self.grabber.output_flip = str(self.cfg.get("flip") or "none")
            jpeg = getattr(packet, "jpeg", None)
            if jpeg:
                with self._lock:
                    self.latest_jpeg = jpeg
                    self.latest_ts = pkt_ts or now
                    self.latest_packet = packet
                last_packet_ts = pkt_ts
            elif packet.frame is not None and (now - last_encode_time) >= 0.09:
                try:
                    try:
                        rot = int(self.cfg.get("rotate") or 0)
                    except (ValueError, TypeError):
                        rot = 0
                    flp = str(self.cfg.get("flip") or "none")
                    fr = packet.frame
                    if rot or flp not in ("none", "", "0"):
                        fr = orient_frame(fr, rot, flp)
                    ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 72])
                    if ok:
                        enc_jpeg = buf.tobytes()
                        last_encode_time = now
                        with self._lock:
                            self.latest_jpeg = enc_jpeg
                            self.latest_ts = pkt_ts or now
                            self.latest_packet = packet
                        last_packet_ts = pkt_ts
                except Exception:
                    pass

            # Flag background ML for the engine thread. Do not call cv2 here —
            # FFmpeg-backed cameras deadlock if the worker encodes while the
            # grabber is reading.
            ml_on = bool(self.cfg.get("ml_enabled", True))
            if ml_on and not self.is_active_ai and (now - self._last_bg_eval >= 1.0):
                self._last_bg_eval = now
                self._bg_eval_frame = packet.frame
                self._bg_eval_pending = True
            time.sleep(0.04)


class CameraStreamPool:
    """Manages concurrent CameraStreamWorkers for all configured cameras."""

    def __init__(self, gateway: Any = None, eval_callback: Any = None):
        self.gateway = gateway
        self.eval_callback = eval_callback
        self._workers: dict[str, CameraStreamWorker] = {}
        self._lock = threading.Lock()
        self.active_camera_id: str = ""

    def set_active_camera(self, camera_id: str | None) -> None:
        cid = str(camera_id or "").strip()
        with self._lock:
            self.active_camera_id = cid
            for wid, worker in self._workers.items():
                worker.is_active_ai = (wid == cid)

    def get_worker(self, camera_id: str | None) -> CameraStreamWorker | None:
        cid = str(camera_id or "").strip()
        with self._lock:
            return self._workers.get(cid)

    def get_latest_jpeg(self, camera_id: str | None) -> tuple[bytes | None, str]:
        worker = self.get_worker(camera_id)
        if worker and worker.latest_jpeg:
            return worker.latest_jpeg, "image/jpeg"
        return None, "image/jpeg"

    def sync_cameras(self, cameras: list[dict[str, Any]]) -> None:
        active_id = self.active_camera_id
        configured_ids = set()
        with self._lock:
            for cam in cameras:
                cid = str(cam.get("id") or "").strip()
                src = cam.get("source")
                if not cam.get("enabled", True):
                    continue
                if not cid or src is None or str(src).strip() == "":
                    continue
                configured_ids.add(cid)
                if cid not in self._workers:
                    worker = CameraStreamWorker(
                        cid, cam, gateway=self.gateway, eval_callback=self.eval_callback
                    )
                    worker.is_active_ai = (cid == active_id)
                    self._workers[cid] = worker
                    worker.start()
                else:
                    self._workers[cid].update_cfg(cam)
                    self._workers[cid].eval_callback = self.eval_callback
                    self._workers[cid].is_active_ai = (cid == active_id)

            to_remove = [wid for wid in self._workers if wid not in configured_ids]
            workers_to_stop = []
            for wid in to_remove:
                worker = self._workers.pop(wid)
                workers_to_stop.append(worker)

        if workers_to_stop:
            def _async_teardown(workers):
                for w in workers:
                    try:
                        w.stop()
                    except Exception as ex:
                        print(f"[CameraStreamPool] Async teardown error for {w.camera_id}: {ex}", flush=True)

            threading.Thread(
                target=_async_teardown,
                args=(workers_to_stop,),
                name="CameraWorkerTeardown",
                daemon=True,
            ).start()

    def stop(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            try:
                worker.stop()
            except Exception:
                pass



class LiveStreamEngine:
    """Background engine that captures video, performs YOLO11 pose inference,

    tracks garage bay occupancy and wrench time, saves proofs, and delivers
    frames to the web UI. Media ingest (AsyncFrameGrabber + go2rtc) is unchanged.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.is_streaming = False
        self.thread: threading.Thread | None = None
        self.current_frame_jpeg: bytes | None = None
        self.current_frame_bgr: np.ndarray | None = None
        self.mjpeg_generation = 0
        self.frame_seq = 0
        self.new_frame_event = threading.Event()
        self.infer_lock = threading.Lock()
        self.latest_camera_event: dict[str, Any] | None = None
        self._bg_camera_state: dict[str, dict[str, Any]] = {}
        self.active_roi_cameras: dict[str, float] = {}
        self._last_bg_roi_infer: dict[str, float] = {}
        self.media = Go2RtcManager()
        self.camera_pool = CameraStreamPool(
            gateway=self.media, eval_callback=self._evaluate_background_camera
        )
        self._fallback_grabber = AsyncFrameGrabber()
        self.discovery = DiscoveryEngine()
        self._gateway_stream_id: str | None = None
        self._gateway_main_stream_id: str | None = None
        self.model = None
        self.face_rec = None
        self.tracker: PersonTracker | None = None
        self.shared_gallery = PersistentReIDGallery()
        self.reid = None
        self.liveness_probe: LivenessProbe | None = None
        self.runtime_profile = None
        self.staff_names: list[str] = []
        self.identities: list[str] = []
        self.force_infer = False

        # Telemetry stats
        self.fps = 0.0
        self.infer_ms = 0.0
        self.is_occupied = False
        self.empty_elapsed = 0.0
        self.person_count = 0
        self.status_text = "STANDBY"
        self.connection_state = "STANDBY"
        self.stream_resolution = "--"
        self.error_message: str | None = None
        self.bay_telemetry: list[dict[str, Any]] = []
        self.last_face_seen: dict[str, float] = {}
        self._wifi_last_tick: float | None = None
        self._camera_frame_cache: dict[str, tuple[bytes, str, float]] = {}

        # Load initial config. SQLite is opened on the worker thread —
        # connections cannot be shared across threads.
        self.cfg = read_config()
        self.conn = None
        self.bot = TelegramOut(self.cfg.get("telegram_bot_token", ""), self.cfg.get("telegram_chat_id", ""))
        self.telegram_links = TelegramLinkService()
        self.telegram_links.configure(self.cfg.get("telegram_bot_token", ""))
        if self.cfg.get("telegram_chat_id"):
            self.telegram_links.set_active_chat(
                str(self.cfg.get("telegram_chat_id")),
                str(self.cfg.get("venue") or self.cfg.get("garage_name") or "Operator"),
            )
        proofs_audit_dir = DATA_DIR / "proofs" / "ai_audits"
        audit_cooldown = float(self.cfg.get("ai_audit_cooldown_seconds") or 45.0)
        self.ai_auditor = AIAuditorQueue(
            gate=TokenSaverGate(cooldown_seconds=audit_cooldown),
            save_crops_dir=proofs_audit_dir,
            on_verdict_callback=self._on_ai_verdict_received,
        )
        self.bay_manager = BayZoneManager(
            self.cfg.get("bays"),
            fallback_roi=parse_roi(self.cfg.get("roi")),
            ai_auditor=self.ai_auditor,
            under_car_grace_seconds=resolve_under_car_grace_seconds(self.cfg),
        )
        self.visit_monitor = CustomerVisitMonitor(
            self.cfg.get("bays"),
            confirm_seconds=float(self.cfg.get("occupy_confirm_seconds") or 1.0),
            clear_seconds=float(self.cfg.get("occupy_clear_seconds") or 4.0),
        )
        self.wifi = WifiTracker(self.cfg.get("wifi_devices"))
        self.bay_telemetry = self.bay_manager.telemetry()
        cmp_cfg = self.cfg.get("complaint_monitoring") or {}
        active_source = str(self.cfg.get("source") or "")
        active_proto = str(self.cfg.get("protocol") or "")
        is_video_source = (
            active_proto == "video"
            or ingest_kind(active_source, active_proto) == "video"
            or str(cmp_cfg.get("audio_source_type", "")).lower() == "video"
        )
        audio_src_type = "video" if is_video_source and active_source else str(cmp_cfg.get("audio_source_type", "laptop"))

        self.complaint_service = ComplaintMonitoringService(
            db_conn=connect(DATA_DIR / "events.db", check_same_thread=False),
            telegram_out=self.bot,
            get_camera_frame_fn=lambda: self.current_frame_bgr,
            audio_source_type=audio_src_type,
            video_path=active_source if is_video_source else None,
            video_loop=bool(cmp_cfg.get("video_loop", False)),
            device_index=cmp_cfg.get("device_index"),
            whisper_model=resolve_khmer_stt_model(
                str(cmp_cfg.get("stt_model_size") or self.cfg.get("whisper_model") or "sengtha/whisper-base-khmer")
            ),
            ollama_model=str(cmp_cfg.get("ollama_model") or self.cfg.get("ollama_model") or "qwen2.5:3b"),
            ollama_host=str(cmp_cfg.get("ollama_host") or "http://localhost:11434"),
            vad_threshold=float(cmp_cfg.get("vad_threshold", 0.35)),
            vad_min_speech_duration_ms=int(cmp_cfg.get("vad_min_speech_duration_ms", 400)),
            vad_max_speech_duration_s=float(cmp_cfg.get("vad_max_speech_duration_s", 30.0)),
            dedup_window_seconds=float(cmp_cfg.get("dedup_window_seconds", 60.0)),
            telegram_min_severity=str(cmp_cfg.get("telegram_min_severity", "low")).lower(),
            telegram_alert_enabled=bool(cmp_cfg.get("telegram_alert_enabled", True)),
            enabled=bool(cmp_cfg.get("enabled", self.cfg.get("complaint_monitoring_enabled", True))),
        )


    @property
    def grabber(self) -> AsyncFrameGrabber:
        worker = self.camera_pool.get_worker(self.cfg.get("active_camera_id"))
        if worker is not None:
            return worker.grabber
        return self._fallback_grabber


    def _on_ai_verdict_received(self, verdict: AIAuditVerdict) -> None:
        try:
            conn = connect(DATA_DIR / "events.db", check_same_thread=False)
            try:
                record_ai_audit_verdict(
                    conn,
                    bay_id=verdict.bay_id,
                    technician_name=verdict.technician_id,
                    action=verdict.action,
                    category=verdict.category,
                    confidence=verdict.confidence,
                    explanation=verdict.explanation,
                    crop_path=verdict.crop_path,
                    ts=verdict.timestamp,
                )
            finally:
                conn.close()
        except Exception as e:
            print(f"[AI Auditor DB Error] {e}", flush=True)

        # Trigger Telegram alert for confirmed distraction / phone usage
        if (
            self.bot
            and self.bot.enabled
            and (
                verdict.action in ("PHONE_USAGE", "DISTRACTION")
                or verdict.category == "DISTRACTION"
                or not verdict.is_work_activity
            )
            and verdict.action != "UNKNOWN"
        ):
            self._send_ai_distraction_telegram(verdict)

    def _send_ai_distraction_telegram(self, verdict: AIAuditVerdict) -> None:
        from graph.compile import pipeline_runtime_flags

        flags = pipeline_runtime_flags(self.cfg)
        if not flags["telegram"] or not flags["alerts"]:
            return
        now = time.time()
        if not hasattr(self, "_last_ai_tg_alerts"):
            self._last_ai_tg_alerts = {}
        # 60-second cooldown per bay to prevent alert flooding
        if (now - self._last_ai_tg_alerts.get(verdict.bay_id, 0.0)) < 60.0:
            return
        self._last_ai_tg_alerts[verdict.bay_id] = now

        venue = str(self.cfg.get("venue") or self.cfg.get("garage_name") or "Workshop").strip()
        bay_name = verdict.bay_id
        for b in (self.cfg.get("bays") or []):
            if isinstance(b, dict) and b.get("id") == verdict.bay_id:
                bay_name = b.get("name") or verdict.bay_id
                break

        conf_pct = int(round((verdict.confidence or 0.0) * 100))
        time_str = verdict.timestamp[11:19] if len(verdict.timestamp) >= 19 else verdict.timestamp
        action_label = verdict.action.replace("_", " ").title()

        caption = (
            f"📱 {venue} Alert: {action_label}\n"
            f"📍 Bay: {bay_name}\n"
            f"👤 Staff: {verdict.technician_id} ({conf_pct}% conf)\n"
            f"⏰ Time: {time_str}\n\n"
            f"🔍 AI Audit:\n{verdict.explanation}"
        )

        try:
            crop_path = Path(verdict.crop_path) if verdict.crop_path else None
            if crop_path and crop_path.is_file():
                sent = self.bot.send_photo(crop_path, caption)
            else:
                sent = self.bot.send_message(caption)
            if sent:
                print(f"[LiveStreamEngine Alert] Sent AI {verdict.action} alert to Telegram for {verdict.bay_id}", flush=True)
            else:
                print(f"[LiveStreamEngine Alert] Failed to send AI {verdict.action} alert to Telegram for {verdict.bay_id}", flush=True)
        except Exception as exc:
            print(f"[LiveStreamEngine Alert] Error sending AI Telegram alert: {exc}", flush=True)

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True
            try:
                self.media.start()
                self.register_all_cameras_in_gateway()
            except Exception as exc:
                print(f"[go2rtc] start failed: {exc}", flush=True)
            self._fallback_grabber.start()
            self.camera_pool.set_active_camera(self.cfg.get("active_camera_id") or "")
            if self.is_streaming:
                self.camera_pool.sync_cameras(list(self.cfg.get("cameras") or []))
            self.wifi.start()
            try:
                self.complaint_service.start()
            except Exception as exc:
                print(f"[complaint_service] start failed: {exc}", flush=True)
            self.thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.thread.start()

    def disconnect(self) -> None:
        """Disconnect active camera streams and return engine to STANDBY."""
        with self.lock:
            self.is_streaming = False
            self.current_frame_jpeg = None
            self.current_frame_bgr = None
            self.status_text = "STANDBY"
            self.connection_state = "STANDBY"
            self.mjpeg_generation += 1
            self.new_frame_event.set()
        self._drop_gateway_stream()
        self.camera_pool.stop()
        self._fallback_grabber.clear_source()

    def stop(self):
        with self.lock:
            self.running = False
            self.is_streaming = False
            self._drop_gateway_stream()
        self._fallback_grabber.stop()
        self.camera_pool.stop()
        try:
            self.complaint_service.stop()
        except Exception:
            pass
        try:
            self.wifi.stop()
        except Exception:
            pass

        try:
            self.media.stop()
        except Exception:
            pass
        try:
            self.telegram_links.stop()
        except Exception:
            pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def reload_face_id(self) -> int:
        if self.face_rec is None:
            self.face_rec = try_create_face_recognizer(self.cfg)
            if self.face_rec is None:
                return 0
        try:
            # Hold infer_lock so YuNet enrollment does not race live YOLO DNN.
            with self.infer_lock:
                count = self.face_rec.reload_enrolled_faces()
            if self.tracker is not None:
                self.tracker.reset(clear_gallery=True)
            if hasattr(self, "shared_gallery") and self.shared_gallery is not None:
                self.shared_gallery.clear()
            if self.bay_manager is not None:
                for bay in getattr(self.bay_manager, "_bays", []):
                    bay.locked_tracks.clear()
            self.force_infer = True
            return count
        except Exception as exc:
            print(f"[FaceID] Reload failed: {exc}")
            return 0

    def _drop_gateway_stream(self) -> None:
        sid = self._gateway_stream_id
        main_sid = self._gateway_main_stream_id
        self._gateway_stream_id = None
        self._gateway_main_stream_id = None
        for name in (sid, main_sid):
            if not name:
                continue
            try:
                self.media.client.remove_stream(name)
            except Exception:
                pass

    def register_all_cameras_in_gateway(self) -> None:
        """Register all configured network cameras and video files into go2rtc so multi-camera grid can stream simultaneously."""
        if not self.media.is_ready():
            return
        cameras = list(self.cfg.get("cameras") or [])
        for cam in cameras:
            cid = cam.get("id")
            src = cam.get("source")
            if not cid or not src:
                continue
            kind = ingest_kind(src, cam.get("protocol"))
            if kind == "webcam":
                continue
            stream_id = sanitize_stream_id(cid)
            if kind == "video":
                try:
                    from adapters.video_file import resolve_video_path
                    vp = resolve_video_path(src)
                    if vp.is_file():
                        # Forward slashes so go2rtc does not treat `C:` as a
                        # second scheme separator and YAML/URL parsers stay sane.
                        url = f"ffmpeg:{vp.resolve().as_posix()}#video=h264#loop"
                    else:
                        continue
                except Exception:
                    continue
            else:
                url = str(parse_source(src))
            try:
                self.media.client.register_stream(stream_id, url)
                main_src = str(cam.get("main_source") or "").strip()
                if main_src:
                    self.media.client.register_stream(f"{stream_id}-main", main_src)
            except Exception as exc:
                print(f"[go2rtc] background register failed for {stream_id}: {exc}", flush=True)

    def _bind_gateway(self, source: Any, camera_id: str, main_source: Any = None) -> str | None:
        """Register a network camera with go2rtc. Webcams skip the gateway."""
        kind = ingest_kind(source, None)
        if kind in ("webcam", "video") or not self.media.is_ready():
            return None
        stream_id = sanitize_stream_id(camera_id or "live")
        url = str(parse_source(source))
        try:
            ok = self.media.client.register_stream(stream_id, url)
        except Exception as exc:
            print(f"[go2rtc] register failed: {exc}", flush=True)
            ok = False
        if not ok:
            print(f"[go2rtc] could not register {stream_id}", flush=True)
            return None
        self._gateway_stream_id = stream_id
        main_url = str(main_source or "").strip()
        main_id = f"{stream_id}-main"
        if main_url:
            try:
                if self.media.client.register_stream(main_id, main_url):
                    self._gateway_main_stream_id = main_id
                    print(f"[go2rtc] stream {main_id} <- {redact_source(main_url)}", flush=True)
                else:
                    self._gateway_main_stream_id = None
            except Exception as exc:
                print(f"[go2rtc] main register failed: {exc}", flush=True)
                self._gateway_main_stream_id = None
        else:
            self._gateway_main_stream_id = None
        print(f"[go2rtc] stream {stream_id} <- {redact_source(url)}", flush=True)
        return stream_id

    def connect_camera(self, new_cfg: dict[str, Any]) -> dict[str, Any]:
        """Queue a camera switch on the grabber thread and return immediately.

        Never opens VideoCapture on the HTTP thread — V4L2/DirectShow handles
        are exclusive and OpenCV objects are thread-affine. Status moves
        CONNECTING → CONNECTED or FAILED via telemetry.
        """
        payload = dict(new_cfg)
        cid = str(payload.pop("camera_id", "") or "")
        cam_name = str(payload.pop("camera_name", "") or "")
        if cid:
            payload.pop("id", None)
        else:
            cid = str(payload.pop("id", "") or "")
        if cam_name:
            payload.pop("name", None)
        else:
            cam_name = str(payload.pop("name", "") or "")
        creds = payload.pop("credentials", None)
        if not isinstance(creds, dict):
            creds = {}
        password = str(payload.pop("password", "") or creds.get("password") or "")
        camera_fields = {
            "id": cid,
            "name": cam_name,
            "protocol": payload.pop("protocol", ""),
            "vendor": payload.pop("vendor", ""),
            "username": str(payload.pop("username", "") or creds.get("username") or ""),
            "enabled": True,
        }
        if "roi" in payload:
            camera_fields["roi"] = payload.pop("roi")
        if "main_source" in payload:
            camera_fields["main_source"] = payload.pop("main_source")
        if "xaddrs" in payload:
            camera_fields["xaddrs"] = payload.pop("xaddrs")
        if "trigger_mode" in payload:
            camera_fields["trigger_mode"] = payload.pop("trigger_mode")
        if "ml_enabled" in payload:
            camera_fields["ml_enabled"] = bool(payload.pop("ml_enabled"))
        payload.pop("enabled", None)
        with self.lock:
            self.cfg.update(payload)
            recovered = unwrap_local_video_source(self.cfg.get("source"))
            proto_hint = str(camera_fields.get("protocol") or self.cfg.get("protocol") or "").lower()
            if proto_hint == "webcam" or ingest_kind(self.cfg.get("source"), proto_hint) == "webcam":
                self.cfg["source"] = webcam_index(self.cfg.get("source"))
                self.cfg["protocol"] = "webcam"
                camera_fields["protocol"] = "webcam"
            if recovered or proto_hint in ("video", "file"):
                if recovered:
                    self.cfg["source"] = recovered
                self.cfg["protocol"] = "video"
                camera_fields["protocol"] = "video"
            if str(camera_fields.get("name") or "").strip() or str(camera_fields.get("id") or "").strip():
                camera_fields["source"] = self.cfg.get("source")
                camera_fields["rotate"] = self.cfg.get("rotate")
                camera_fields["flip"] = self.cfg.get("flip")
                if "ml_enabled" in camera_fields:
                    camera_fields["ml_enabled"] = bool(camera_fields["ml_enabled"])
                upsert_camera(self.cfg, camera_fields)
            cameras = list(self.cfg.get("cameras") or [])
            active_id = str(cid or self.cfg.get("active_camera_id") or "")
            if active_id:
                self.cfg["active_camera_id"] = active_id
            active: dict[str, Any] = {}
            for cam in cameras:
                if cam.get("id") == active_id:
                    active = cam
                    break
            protocol = str(
                active.get("protocol")
                or camera_fields.get("protocol")
                or self.cfg.get("protocol")
                or protocol_from_source(self.cfg.get("source"))
            )
            main_source = active.get("main_source") or self.cfg.get("main_source") or ""
            xaddrs = active.get("xaddrs") or camera_fields.get("xaddrs") or []
            username = str(active.get("username") or camera_fields.get("username") or "")
            self.cfg["protocol"] = protocol
            self.cfg["main_source"] = main_source
            cfg_to_save = dict(self.cfg)
            self.bot = TelegramOut(
                self.cfg.get("telegram_bot_token", ""),
                self.cfg.get("telegram_chat_id", ""),
            )
            source = self.cfg.get("source", 0)
            existing_worker = self.camera_pool.get_worker(active_id)
            worker_live = (
                existing_worker is not None
                and existing_worker.grabber.connection_state in ("CONNECTED", "CONNECTING", "RECONNECTING")
                and str(existing_worker.cfg.get("source")) == str(source)
            )
            self.error_message = None
            self.is_streaming = True
            self.current_frame_jpeg = None
            self.fps = 0.0
            self.is_occupied = False
            self.person_count = 0
            self.staff_names = []
            self.identities = []
            if worker_live:
                grabber_state = existing_worker.grabber.connection_state
                self.connection_state = grabber_state if grabber_state == "CONNECTED" else "CONNECTING"
                self.status_text = "CONNECTED" if grabber_state == "CONNECTED" else "CONNECTING"
            else:
                self.mjpeg_generation += 1
                self.status_text = "CONNECTING"
                self.connection_state = "CONNECTING"
            self.new_frame_event.set()
            active_bays = list(active.get("bays") or [])
            if not active_bays:
                active_roi = parse_roi(active.get("roi")) or parse_roi(self.cfg.get("roi")) or [0.30, 0.20, 0.40, 0.60]
                active_bays = parse_bays(None, fallback_roi=active_roi, seed_if_empty=True)
                active["bays"] = active_bays
            self.cfg["bays"] = active_bays
            roi = list(parse_roi(active.get("roi")) or self.cfg.get("roi") or (active_bays[0]["roi"] if active_bays else [0.30, 0.20, 0.40, 0.60]))
            self.cfg["roi"] = roi
            self.bay_manager.set_bays(active_bays)
            self.bay_telemetry = self.bay_manager.telemetry()
            bays = list(self.cfg.get("bays") or DEFAULT_BAYS)
            rotate = self.cfg.get("rotate", 0)
            flip = self.cfg.get("flip", "none")
            cfg_to_save = dict(self.cfg)

        # Perform disk write and network/gateway operations outside self.lock to avoid deadlock
        save_config(cfg_to_save)

        existing = self.camera_pool.get_worker(active_id)
        worker_live = (
            existing is not None
            and existing.grabber.connection_state in ("CONNECTED", "CONNECTING", "RECONNECTING")
            and str(existing.cfg.get("source")) == str(source)
        )

        if _needs_onvif_onboard(protocol, source, xaddrs) and not worker_live:
            self.grabber.mark_connecting()
            threading.Thread(
                target=self._onboard_onvif,
                kwargs={
                    "xaddrs": xaddrs,
                    "source": source,
                    "username": username,
                    "password": password,
                    "camera_id": active_id or cid,
                    "camera_name": cam_name or str(active.get("name") or ""),
                },
                name="OnvifOnboard",
                daemon=True,
            ).start()
            media = {"ready": self.media.is_ready()}
            return {
                "success": True,
                "pending": True,
                "connection": "CONNECTING",
                "status": "CONNECTING",
                "message": f"Resolving ONVIF '{redact_source(onvif_xaddr(source, xaddrs))}'…",
                "roi": roi,
                "bays": bays,
                "rotate": rotate,
                "flip": flip,
                "cameras": cameras,
                "active_camera_id": active_id,
                "stream_id": None,
                "media": media,
            }

        kind = ingest_kind(source, protocol)
        if kind in ("webcam", "video") or worker_live:
            stream_id = None
        else:
            stream_id = self._bind_gateway(source, active_id, main_source=main_source)
        self.camera_pool.set_active_camera(active_id)
        if self.running:
            self.camera_pool.sync_cameras(cameras)
            worker = self.camera_pool.get_worker(active_id)
            if worker is None:
                adapter = create_adapter(
                    source,
                    gateway=self.media if stream_id else None,
                    stream_id=stream_id,
                    protocol=protocol,
                    username=username,
                    password=password,
                    xaddrs=xaddrs,
                    main_source=main_source,
                )
                self.grabber.switch_source(adapter)
            else:
                # Pool owns capture; drop any leftover exclusive V4L2 handle.
                self._fallback_grabber.clear_source()
        else:
            adapter = create_adapter(
                source,
                gateway=self.media if stream_id else None,
                stream_id=stream_id,
                protocol=protocol,
                username=username,
                password=password,
                xaddrs=xaddrs,
                main_source=main_source,
            )
            self.grabber.switch_source(adapter)

        # Dynamic Audio Source Synchronization
        if self.complaint_service is not None:
            try:
                if kind == "video" or protocol == "video":
                    from audio_source import VideoFileAudioSource
                    cmp_cfg = self.cfg.get("complaint_monitoring") or {}
                    self.complaint_service.set_audio_source(
                        VideoFileAudioSource(
                            video_path=source,
                            sample_rate=16000,
                            loop=bool(cmp_cfg.get("video_loop", False)),
                        )
                    )
                else:
                    cmp_cfg = self.cfg.get("complaint_monitoring") or {}
                    src_type = str(cmp_cfg.get("audio_source_type", "laptop")).lower()
                    if src_type == "usb":
                        from audio_source import USBMicrophoneSource
                        self.complaint_service.set_audio_source(USBMicrophoneSource(device_index=cmp_cfg.get("device_index")))
                    else:
                        from audio_source import LaptopMicrophoneSource
                        self.complaint_service.set_audio_source(LaptopMicrophoneSource(device_index=cmp_cfg.get("device_index")))
            except Exception as audio_sw_err:
                print(f"[ComplaintService] Audio source switch notice: {audio_sw_err}", flush=True)

        media = self.media.status(stream_id) if stream_id else {"ready": self.media.is_ready()}
        connected = worker_live and self.connection_state == "CONNECTED"
        return {
            "success": True,
            "pending": not connected,
            "connection": "CONNECTED" if connected else "CONNECTING",
            "status": "CONNECTED" if connected else "CONNECTING",
            "message": (
                f"Switched to '{redact_source(source)}'."
                if connected
                else f"Connecting to '{redact_source(source)}'…"
            ),
            "roi": roi,
            "bays": bays,
            "rotate": rotate,
            "flip": flip,
            "cameras": cameras,
            "active_camera_id": active_id,
            "stream_id": stream_id,
            "media": media,
        }

    def _onboard_onvif(
        self,
        *,
        xaddrs: Any,
        source: Any,
        username: str,
        password: str,
        camera_id: str,
        camera_name: str,
    ) -> None:
        """SOAP on a worker: persist RTSP URLs, then queue the grabber switch."""
        from adapters.onvif import ONVIFAdapter

        adapter = ONVIFAdapter(
            onvif_xaddr(source, xaddrs),
            username=username,
            password=password,
            xaddrs=xaddrs,
            source=source,
        )
        try:
            ok = adapter.resolve()
        except Exception as exc:
            ok = False
            adapter.error = str(exc)
        if not ok:
            self.grabber.switch_source(
                _ConnectErrorAdapter(adapter.error or "ONVIF resolve failed.")
            )
            return
        sub = adapter.source
        main = adapter.main_source or sub
        try:
            with self.lock:
                self.cfg["source"] = sub
                self.cfg["main_source"] = main
                self.cfg["protocol"] = "onvif"
                if adapter.manufacturer:
                    self.cfg["vendor"] = adapter.manufacturer
                if camera_id or camera_name:
                    upsert_camera(
                        self.cfg,
                        {
                            "id": camera_id,
                            "name": camera_name or "Untitled camera",
                            "source": sub,
                            "main_source": main,
                            "protocol": "onvif",
                            "vendor": adapter.manufacturer or "generic",
                            "username": username,
                            "xaddrs": xaddrs,
                        },
                    )
                cfg_to_save = dict(self.cfg)
                active_id = str(self.cfg.get("active_camera_id") or camera_id or "")
            save_config(cfg_to_save)
            stream_id = self._bind_gateway(sub, active_id, main_source=main)
            inner = create_adapter(
                sub,
                gateway=self.media if stream_id else None,
                stream_id=stream_id,
                protocol="rtsp",
            )
            self.grabber.switch_source(inner)
        except Exception as exc:
            self.grabber.switch_source(_ConnectErrorAdapter(str(exc)))

    def apply_hub_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        if "telegram_chat_id" in payload:
            updates["telegram_chat_id"] = normalize_chat_id(payload.get("telegram_chat_id"))
        if "venue" in payload:
            updates["venue"] = str(payload.get("venue") or "").strip() or "Demo Garage"
            updates["garage_name"] = updates["venue"]
        if "garage_name" in payload:
            updates["garage_name"] = str(payload.get("garage_name") or "").strip() or "Demo Garage"
            updates["venue"] = updates["garage_name"]
        if "open_time" in payload:
            updates["open_time"] = parse_clock(payload.get("open_time"), "08:00")
        if "close_time" in payload:
            updates["close_time"] = parse_clock(payload.get("close_time"), "18:00")
        if "open_time" in updates or "close_time" in updates:
            updates["operating_hours"] = {
                "open": updates.get("open_time") or str(self.cfg.get("open_time") or "08:00"),
                "close": updates.get("close_time") or str(self.cfg.get("close_time") or "18:00"),
            }
        if "absent_seconds" in payload:
            try:
                updates["absent_seconds"] = max(5.0, min(600.0, float(payload.get("absent_seconds"))))
            except (TypeError, ValueError):
                return {"success": False, "error": "absent_seconds must be a number."}
        if "cooldown_seconds" in payload:
            try:
                updates["cooldown_seconds"] = max(5.0, min(600.0, float(payload.get("cooldown_seconds"))))
            except (TypeError, ValueError):
                return {"success": False, "error": "cooldown_seconds must be a number."}
        if "auto_create_bays" in payload:
            updates["auto_create_bays"] = bool(payload.get("auto_create_bays"))
        if not updates:
            return {"success": False, "error": "No settings provided."}
        with self.lock:
            self.cfg.update(updates)
            save_config(updates)
            if "telegram_bot_token" in updates or "telegram_chat_id" in updates:
                self.bot = TelegramOut(
                    self.cfg.get("telegram_bot_token", ""),
                    self.cfg.get("telegram_chat_id", ""),
                )
            if "telegram_bot_token" in updates:
                self.telegram_links.configure(self.cfg.get("telegram_bot_token", ""))
            if "telegram_chat_id" in updates:
                self.telegram_links.set_active_chat(
                    self.cfg.get("telegram_chat_id", ""),
                    str(self.cfg.get("venue") or self.cfg.get("garage_name") or "Operator"),
                )
            snapshot = {key: self.cfg.get(key) for key in _SETTINGS_KEYS}
        snapshot["success"] = True
        snapshot["bays"] = list(self.cfg.get("bays") or [])
        snapshot["wifi_devices"] = list(self.cfg.get("wifi_devices") or [])
        snapshot["operating_hours"] = self.cfg.get("operating_hours")
        snapshot["telegram_bot_configured"] = bool(str(self.cfg.get("telegram_bot_token") or "").strip())
        snapshot["telegram_bot_username"] = self.telegram_links.status().get("bot_username") or ""
        snapshot["telegram_bot_token"] = ""
        return snapshot

    def save_camera(self, fields: dict[str, Any]) -> dict[str, Any]:
        name = str(fields.get("name") or fields.get("camera_name") or "").strip()
        if not name:
            return {"success": False, "error": "Camera location name is required."}
        mapped = {
            "id": fields.get("id") or fields.get("camera_id") or "",
            "name": name,
            "source": fields.get("source", ""),
            "protocol": fields.get("protocol", ""),
            "vendor": fields.get("vendor", ""),
            "username": fields.get("username", ""),
            "rotate": fields.get("rotate", "auto"),
            "flip": fields.get("flip", "none"),
            "roi": fields.get("roi"),
        }
        if "main_source" in fields:
            mapped["main_source"] = fields.get("main_source", "")
        if "xaddrs" in fields:
            mapped["xaddrs"] = fields.get("xaddrs")
        if "bays" in fields:
            mapped["bays"] = fields.get("bays")
        if "enabled" in fields:
            mapped["enabled"] = bool(fields.get("enabled"))
        if "ml_enabled" in fields:
            mapped["ml_enabled"] = bool(fields.get("ml_enabled"))
        if "trigger_mode" in fields:
            mapped["trigger_mode"] = str(fields.get("trigger_mode") or "roi_state_change")
        with self.lock:
            entry = upsert_camera(self.cfg, mapped)
            save_config(self.cfg)
        try:
            self.register_all_cameras_in_gateway()
        except Exception:
            pass
        if self.running:
            try:
                self.camera_pool.sync_cameras(list(self.cfg.get("cameras") or []))
            except Exception:
                pass
        return {
            "success": True,
            "camera": entry,
            "cameras": list(self.cfg.get("cameras") or []),
            "active_camera_id": entry["id"],
            "bays": entry.get("bays") or [],
        }

    def delete_camera(self, camera_id: Any) -> dict[str, Any]:
        cid = str(camera_id or "").strip()
        if not cid:
            return {"success": False, "error": "Camera id is required."}
        with self.lock:
            cameras = [c for c in _normalize_cameras(self.cfg.get("cameras"), workplace=self.cfg.get("workplace_type")) if c["id"] != cid]
            self.cfg["cameras"] = cameras
            if str(self.cfg.get("active_camera_id") or "") == cid:
                self.cfg["active_camera_id"] = cameras[0]["id"] if cameras else ""
            if self._gateway_stream_id == sanitize_stream_id(cid):
                self._drop_gateway_stream()
            cfg_to_save = dict(self.cfg)
            active_id = str(self.cfg.get("active_camera_id") or "")
        save_config(cfg_to_save)
        if self.running:
            try:
                self.camera_pool.sync_cameras(cameras)
            except Exception:
                pass
        return {
            "success": True,
            "cameras": cameras,
            "active_camera_id": active_id,
        }

    def toggle_camera_port(self, camera_id: Any, enabled: bool | None = None) -> dict[str, Any]:
        cid = str(camera_id or "").strip()
        if not cid:
            return {"success": False, "error": "Camera id is required."}
        with self.lock:
            cameras = _normalize_cameras(self.cfg.get("cameras"), workplace=self.cfg.get("workplace_type"))
            target = None
            for cam in cameras:
                if cam["id"] == cid:
                    target = cam
                    break
            if target is None:
                return {"success": False, "error": f"Camera '{cid}' not found."}
            current = target.get("enabled", True)
            new_state = (not current) if enabled is None else bool(enabled)
            target["enabled"] = new_state
            self.cfg["cameras"] = cameras
            active_id = str(self.cfg.get("active_camera_id") or "")
            closing_active = (not new_state) and cid == active_id
            if closing_active:
                self.is_streaming = False
                self.current_frame_jpeg = None
                self.current_frame_bgr = None
                self.status_text = "STANDBY"
                self.connection_state = "STANDBY"
                self.mjpeg_generation += 1
                self.new_frame_event.set()
            cfg_to_save = dict(self.cfg)
        save_config(cfg_to_save)
        if self.running or bool(self.camera_pool._workers):
            try:
                self.camera_pool.sync_cameras(cameras)
            except Exception:
                pass
        if closing_active:
            self._fallback_grabber.clear_source()
        return {
            "success": True,
            "camera_id": cid,
            "enabled": new_state,
            "cameras": cameras,
            "active_camera_id": str(self.cfg.get("active_camera_id") or ""),
        }

    def toggle_camera_ml(self, camera_id: Any, enabled: bool | None = None) -> dict[str, Any]:
        cid = str(camera_id or "").strip()
        if not cid:
            return {"success": False, "error": "Camera id is required."}
        with self.lock:
            cameras = _normalize_cameras(self.cfg.get("cameras"), workplace=self.cfg.get("workplace_type"))
            target = None
            for cam in cameras:
                if cam["id"] == cid:
                    target = cam
                    break
            if target is None:
                return {"success": False, "error": f"Camera '{cid}' not found."}
            current = target.get("ml_enabled", True)
            new_state = (not current) if enabled is None else bool(enabled)
            target["ml_enabled"] = new_state
            self.cfg["cameras"] = cameras
            cfg_to_save = dict(self.cfg)
            if not new_state:
                self.active_roi_cameras.pop(cid, None)
                self._last_bg_roi_infer.pop(cid, None)
        save_config(cfg_to_save)
        worker = self.camera_pool.get_worker(cid)
        if worker is not None:
            worker.update_cfg({"ml_enabled": new_state})
        return {
            "success": True,
            "camera_id": cid,
            "ml_enabled": new_state,
            "cameras": cameras,
            "active_camera_id": str(self.cfg.get("active_camera_id") or ""),
        }

    def set_roi(self, roi_value: Any) -> dict[str, Any]:
        parsed = parse_roi(roi_value)
        if parsed is None:
            return {"success": False, "error": "ROI must be [x, y, width, height] fractions."}
        with self.lock:
            self.cfg["roi"] = parsed
            bays = parse_bays(self.cfg.get("bays"), fallback_roi=parsed)
            if bays:
                bays[0] = {**bays[0], "roi": parsed}
            self.cfg["bays"] = bays
            cameras = _normalize_cameras(self.cfg.get("cameras"), workplace=self.cfg.get("workplace_type"))
            active = str(self.cfg.get("active_camera_id") or "")
            for cam in cameras:
                if cam["id"] == active:
                    cam["roi"] = parsed
                    cam["bays"] = bays
                    break
            self.cfg["cameras"] = cameras
            self.bay_manager.set_bays(bays)
            self.bay_telemetry = self.bay_manager.telemetry()
            cfg_to_save = dict(self.cfg)
        save_config(cfg_to_save)
        return {"success": True, "roi": parsed, "bays": bays, "cameras": cameras, "active_camera_id": active}

    def set_bays(self, bays_value: Any) -> dict[str, Any]:
        parsed = parse_bays(
            bays_value,
            fallback_roi=parse_roi(self.cfg.get("roi")),
            seed_if_empty=False,
            workplace=self.cfg.get("workplace_type"),
        )
        with self.lock:
            previous = parse_bays(self.cfg.get("bays"), seed_if_empty=False)
            previous_ids = {str(bay.get("id") or "") for bay in previous}
            keep_ids = {str(bay.get("id") or "") for bay in parsed}
            removed_ids = [bay_id for bay_id in previous_ids if bay_id and bay_id not in keep_ids]
            self.cfg["bays"] = parsed
            if parsed:
                self.cfg["roi"] = list(parsed[0]["roi"])
            cameras = _normalize_cameras(self.cfg.get("cameras"), workplace=self.cfg.get("workplace_type"))
            active = str(self.cfg.get("active_camera_id") or "")
            for cam in cameras:
                if cam["id"] == active:
                    cam["roi"] = self.cfg.get("roi")
                    cam["bays"] = parsed
                    break
            self.cfg["cameras"] = cameras
            self.bay_manager.set_bays(parsed)
            self.visit_monitor.set_zones(parsed)
            self.bay_telemetry = self.bay_manager.telemetry()
            cfg_to_save = dict(self.cfg)
            roi = list(self.cfg.get("roi") or [0.30, 0.20, 0.40, 0.60])
        save_config(cfg_to_save)
        if removed_ids:
            def _async_close_sessions(bids):
                try:
                    import sqlite3
                    db_path = DATA_DIR / "events.db"
                    conn = sqlite3.connect(str(db_path), timeout=5.0)
                    try:
                        conn.execute("PRAGMA busy_timeout = 5000")
                        close_bay_sessions(conn, bids, datetime.now())
                    finally:
                        conn.close()
                except Exception as exc:
                    print(f"[set_bays] async close sessions failed: {exc}", flush=True)

            threading.Thread(
                target=_async_close_sessions,
                args=(list(removed_ids),),
                name="BaySessionsCloser",
                daemon=True,
            ).start()
        return {
            "success": True,
            "bays": parsed,
            "roi": roi,
            "cameras": cameras,
            "active_camera_id": active,
            "removed_bay_ids": removed_ids,
        }

    def import_bays(self, source_camera_id: Any, target_camera_id: Any = None) -> dict[str, Any]:
        src_id = str(source_camera_id or "").strip()
        if not src_id:
            return {"success": False, "error": "Source camera id is required."}
        with self.lock:
            cameras = _normalize_cameras(self.cfg.get("cameras"), workplace=self.cfg.get("workplace_type"))
            tgt_id = str(target_camera_id or self.cfg.get("active_camera_id") or "").strip()
            if not tgt_id:
                return {"success": False, "error": "Target camera id is required."}
            src_cam = next((c for c in cameras if c["id"] == src_id), None)
            if not src_cam:
                return {"success": False, "error": f"Source camera '{src_id}' not found."}
            tgt_cam = next((c for c in cameras if c["id"] == tgt_id), None)
            if not tgt_cam:
                return {"success": False, "error": f"Target camera '{tgt_id}' not found."}

            src_bays = src_cam.get("bays") or []
            if not src_bays:
                src_roi = src_cam.get("roi") or [0.30, 0.20, 0.40, 0.60]
                src_bays = parse_bays(None, fallback_roi=src_roi, seed_if_empty=True)

            cloned = []
            for b in src_bays:
                item = dict(b)
                if "roi" in item and isinstance(item["roi"], list):
                    item["roi"] = list(item["roi"])
                if "polygon" in item and isinstance(item["polygon"], list):
                    item["polygon"] = [list(pt) for pt in item["polygon"]]
                cloned.append(item)

            tgt_cam["bays"] = cloned
            if tgt_id == str(self.cfg.get("active_camera_id") or ""):
                self.cfg["bays"] = cloned
                if cloned:
                    self.cfg["roi"] = list(cloned[0]["roi"])
                self.bay_manager.set_bays(cloned)
                self.bay_telemetry = self.bay_manager.telemetry()

            self.cfg["cameras"] = cameras
            save_config(self.cfg)
            active = str(self.cfg.get("active_camera_id") or "")
            roi = list(self.cfg.get("roi") or [0.30, 0.20, 0.40, 0.60])
        return {
            "success": True,
            "imported_bays_count": len(cloned),
            "bays": cloned,
            "cameras": cameras,
            "active_camera_id": active,
            "roi": roi,
            "message": f"Imported {len(cloned)} bays from {src_cam.get('name', src_id)}.",
        }

    def get_camera_frame(self, camera_id: str | None = None) -> tuple[bytes | None, str]:
        """Fetch latest JPEG frame for a specific camera.

        Returns (jpeg_bytes, content_type).
        - If camera_id is None, empty, or active_camera_id: returns the active AI-annotated frame.
        - If another camera: pulls from CameraStreamPool JPEG cache (never a competing RTSP open).
        """
        cid = str(camera_id or "").strip()
        active_id = str(self.cfg.get("active_camera_id") or "").strip()
        target_id = cid or active_id
        with self.lock:
            cams = list(self.cfg.get("cameras") or [])
        target_cam = next((c for c in cams if str(c.get("id")) == target_id), None)
        if target_cam and not target_cam.get("enabled", True):
            return None, "image/jpeg"
        now = time.time()

        if not cid or cid == active_id:
            with self.lock:
                frame_data = self.current_frame_jpeg
            if frame_data:
                self._camera_frame_cache[cid or active_id] = (frame_data, "image/jpeg", now)
                return frame_data, "image/jpeg"
            cached = self._camera_frame_cache.get(cid or active_id)
            if cached and (now - cached[2]) < 2.5:
                return cached[0], cached[1]
            worker = self.camera_pool.get_worker(cid or active_id)
            if worker and worker.latest_jpeg:
                return worker.latest_jpeg, "image/jpeg"
            pkt = self.grabber.peek_latest_frame()
            if pkt is not None:
                if getattr(pkt, "jpeg", None):
                    return pkt.jpeg, "image/jpeg"
                if getattr(pkt, "frame", None) is not None:
                    try:
                        with self.lock:
                            rot_val = self.cfg.get("rotate")
                            flp_val = self.cfg.get("flip")
                        rot = 0 if is_auto_rotate(rot_val) else parse_rotate(rot_val)
                        flp = parse_flip(flp_val)
                        fr = pkt.frame
                        if rot or flp != "none":
                            fr = orient_frame(fr, rot, flp)
                        ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 72])
                        if ok:
                            enc_jpeg = buf.tobytes()
                            self._camera_frame_cache[cid or active_id] = (enc_jpeg, "image/jpeg", now)
                            return enc_jpeg, "image/jpeg"
                    except Exception:
                        pass
            if worker:
                wpkt = worker.grabber.peek_latest_frame()
                if wpkt is not None:
                    if getattr(wpkt, "jpeg", None):
                        return wpkt.jpeg, "image/jpeg"
                    if getattr(wpkt, "frame", None) is not None:
                        try:
                            rot = int(worker.cfg.get("rotate") or 0)
                            flp = str(worker.cfg.get("flip") or "none")
                            fr = wpkt.frame
                            if rot or flp not in ("none", "", "0"):
                                fr = orient_frame(fr, rot, flp)
                            ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 72])
                            if ok:
                                enc_jpeg = buf.tobytes()
                                self._camera_frame_cache[cid or active_id] = (enc_jpeg, "image/jpeg", now)
                                return enc_jpeg, "image/jpeg"
                        except Exception:
                            pass
            if cached:
                return cached[0], cached[1]
            return None, "image/jpeg"

        # Background camera: serve the worker JPEG cache only.
        # Do not steal grabber frames or open a competing RTSP capture from
        # the HTTP thread — both freeze every other network camera in the grid.
        worker = self.camera_pool.get_worker(cid)
        if worker is None:
            with self.lock:
                cams = list(self.cfg.get("cameras") or [])
            self.camera_pool.sync_cameras(cams)
            worker = self.camera_pool.get_worker(cid)

        if worker:
            pkt = worker.grabber.peek_latest_frame()
            pkt_ts = float(getattr(pkt, "timestamp", 0.0) or 0.0) if pkt is not None else 0.0
            pkt_jpeg = getattr(pkt, "jpeg", None) if pkt is not None else None

            # 1. Grabber has a newer pre-encoded JPEG frame
            if pkt_jpeg and pkt_ts > (worker.latest_ts or 0.0):
                with worker._lock:
                    worker.latest_jpeg = pkt_jpeg
                    worker.latest_ts = pkt_ts
                    worker.latest_packet = pkt
                self._camera_frame_cache[cid] = (pkt_jpeg, "image/jpeg", pkt_ts)
                return pkt_jpeg, "image/jpeg"

            # 2. Worker cache is fresh (within last 0.35s)
            if worker.latest_jpeg and (now - (worker.latest_ts or 0.0)) <= 0.35:
                self._camera_frame_cache[cid] = (worker.latest_jpeg, "image/jpeg", worker.latest_ts)
                return worker.latest_jpeg, "image/jpeg"

            # 3. Worker cache is stale, but grabber has a pre-encoded JPEG
            if pkt_jpeg:
                with worker._lock:
                    worker.latest_jpeg = pkt_jpeg
                    worker.latest_ts = pkt_ts or now
                    worker.latest_packet = pkt
                self._camera_frame_cache[cid] = (pkt_jpeg, "image/jpeg", worker.latest_ts)
                return pkt_jpeg, "image/jpeg"

            # 4. Fallback: encode raw frame from grabber if needed
            if pkt is not None and pkt.frame is not None:
                try:
                    rot = int(worker.cfg.get("rotate") or 0)
                    flp = str(worker.cfg.get("flip") or "none")
                    fr = pkt.frame
                    if rot or flp not in ("none", "", "0"):
                        fr = orient_frame(fr, rot, flp)
                    ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 72])
                    if ok:
                        enc_jpeg = buf.tobytes()
                        with worker._lock:
                            worker.latest_jpeg = enc_jpeg
                            worker.latest_ts = pkt_ts or now
                            worker.latest_packet = pkt
                        self._camera_frame_cache[cid] = (enc_jpeg, "image/jpeg", worker.latest_ts)
                        return enc_jpeg, "image/jpeg"
                except Exception:
                    pass

            # 5. Serve existing worker latest_jpeg if available
            if worker.latest_jpeg:
                self._camera_frame_cache[cid] = (worker.latest_jpeg, "image/jpeg", worker.latest_ts or now)
                return worker.latest_jpeg, "image/jpeg"

        cached = self._camera_frame_cache.get(cid)
        if cached and (now - cached[2]) < 5.0:
            return cached[0], cached[1]
        if cached:
            return cached[0], cached[1]
        return None, "image/jpeg"

    def set_wifi_devices(self, devices_value: Any) -> dict[str, Any]:
        parsed = self.wifi.set_devices(devices_value)
        with self.lock:
            self.cfg["wifi_devices"] = parsed
            save_config(self.cfg)
        return {"success": True, "wifi_devices": parsed}

    def _bank_hard_negatives(self, frame, tracks: list) -> None:
        if not tracks:
            return
        bank_hard_negatives(frame, tracks, DATA_DIR)

    def _apply_bay_zoom(
        self,
        frame,
        bays: list,
        accepted: list[Detection],
        rejected: list[Detection],
        *,
        imgsz: int,
        person_conf: float,
        min_person_height: float,
        min_aspect: float,
        min_keypoints: int,
        kpt_conf: float,
        occupancy_by_id: dict | None = None,
    ) -> tuple[list[Detection], list[Detection]]:
        cfg = self.cfg or {}
        if not resolve_bay_zoom(cfg) or self.model is None:
            return accepted, rejected
        return zoom_empty_bays(
            self.model,
            frame,
            bays or [],
            accepted,
            rejected,
            imgsz=imgsz,
            device=self.runtime_profile.yolo_device if self.runtime_profile else None,
            person_conf=person_conf,
            min_height_frac=min_person_height,
            min_aspect=min_aspect,
            min_keypoints=min_keypoints,
            kpt_conf=kpt_conf,
            pad=resolve_bay_zoom_pad(cfg),
            occupancy_by_id=occupancy_by_id,
        )

    def _evaluate_background_camera(
        self, camera_id: str, frame: np.ndarray, cam_cfg: dict[str, Any]
    ) -> bool:
        if not cam_cfg.get("ml_enabled", True):
            return True
        if self.model is None or not self.running:
            return False
        if not self.infer_lock.acquire(blocking=False):
            return False
        try:
            now = time.time()
            state = self._bg_camera_state.setdefault(
                camera_id,
                {
                    "was_occupied": False,
                    "last_trigger_ts": 0.0,
                    "last_person_count": 0,
                    "last_vehicle_count": 0,
                },
            )

            h, w = frame.shape[:2]
            trigger_mode = str(cam_cfg.get("trigger_mode") or "roi_state_change")

            vehicles = []
            if getattr(self, "vehicle_model", None) is not None:
                try:
                    veh_res = self.vehicle_model.predict(
                        frame,
                        imgsz=512,
                        classes=[2, 3, 5, 7],
                        conf=0.18,
                        device=None,
                        verbose=False,
                    )[0]
                    vehicles = extract_vehicle_detections(veh_res, w, h, conf_min=0.18)
                except Exception:
                    pass

            cfg = dict(self.cfg)
            person_conf = resolve_person_conf(cfg)
            min_person_height = resolve_min_person_height(cfg)
            min_aspect = resolve_min_aspect(cfg)
            min_keypoints = resolve_min_keypoints(cfg)
            kpt_conf = resolve_kpt_conf(cfg)
            imgsz = resolve_imgsz(cfg, self.runtime_profile)

            bays = cam_cfg.get("bays") or []
            if not bays:
                roi = parse_roi(cam_cfg.get("roi")) or [0.30, 0.20, 0.40, 0.60]
                bays = [{"id": f"{camera_id}_bay", "name": "Bay 1", "roi": roi, "type": "vehicle_bay"}]

            person_res = self.model.predict(
                frame,
                imgsz=imgsz,
                conf=person_conf,
                device=self.runtime_profile.yolo_device if self.runtime_profile else None,
                verbose=False,
            )[0]
            persons, rejected = person_detections(
                person_res,
                h,
                conf_min=person_conf,
                min_height_frac=min_person_height,
                min_aspect=min_aspect,
                min_keypoints=min_keypoints,
                kpt_conf=kpt_conf,
            )
            persons, rejected = self._apply_bay_zoom(
                frame,
                bays,
                persons,
                rejected,
                imgsz=imgsz,
                person_conf=person_conf,
                min_person_height=min_person_height,
                min_aspect=min_aspect,
                min_keypoints=min_keypoints,
                kpt_conf=kpt_conf,
            )
            persons, vetoed = veto_vehicle_interior(persons, vehicles, kpt_conf=kpt_conf)
            rejected.extend(vetoed)

            triggered = False
            event_desc = ""

            if trigger_mode == "any_detection":
                if persons:
                    with self.lock:
                        self.active_roi_cameras[camera_id] = now
                    if (now - state["last_trigger_ts"]) >= 20.0:
                        triggered = True
                        event_desc = f"{len(persons)} person{'s' if len(persons) > 1 else ''} detected"
                elif vehicles and (now - state["last_trigger_ts"]) >= 20.0:
                    triggered = True
                    event_desc = f"{len(vehicles)} vehicle{'s' if len(vehicles) > 1 else ''} detected"
            else:
                persons_in_bays = []
                for det in persons:
                    for b in bays:
                        if detection_in_bay(det, b, w, h, kpt_conf=0.4):
                            persons_in_bays.append((det, b))
                            break

                vehicles_in_bays = []
                for veh in vehicles:
                    vbox = getattr(veh, "box", None)
                    if vbox is None:
                        continue
                    for b in bays:
                        broi = b.get("roi") or [0.3, 0.2, 0.4, 0.6]
                        if _box_intersects_roi(vbox, broi, w, h):
                            vehicles_in_bays.append((veh, b))
                            break

                now_occupied = bool(persons_in_bays or vehicles_in_bays)
                if persons_in_bays:
                    with self.lock:
                        self.active_roi_cameras[camera_id] = now

                if (now - state["last_trigger_ts"]) >= 20.0:
                    if now_occupied and not state["was_occupied"]:
                        triggered = True
                        if persons_in_bays:
                            bay_name = persons_in_bays[0][1].get("name") or "Bay"
                            event_desc = f"Movement in {bay_name}"
                        else:
                            bay_name = vehicles_in_bays[0][1].get("name") or "Bay"
                            event_desc = f"Vehicle entered {bay_name}"
                    elif now_occupied and (now - state["last_trigger_ts"] > 60.0):
                        triggered = True
                        event_desc = f"Active presence in {bays[0].get('name') or 'Bay'}"

                state["was_occupied"] = now_occupied
                state["last_person_count"] = len(persons_in_bays)
                state["last_vehicle_count"] = len(vehicles_in_bays)

            if triggered:
                state["last_trigger_ts"] = now
                cam_name = str(cam_cfg.get("name") or f"Camera {camera_id}").strip()
                event_obj = {
                    "event_id": f"{camera_id}_{int(now * 1000)}",
                    "camera_id": camera_id,
                    "camera_name": cam_name,
                    "event": event_desc,
                    "trigger_mode": trigger_mode,
                    "timestamp": now,
                    "handled": False,
                }
                with self.lock:
                    self.latest_camera_event = event_obj
                print(
                    f"[BackgroundML] Event triggered on {event_obj['camera_name']}: {event_obj['event']}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[BackgroundML] Eval error on {camera_id}: {exc}", flush=True)
        finally:
            self.infer_lock.release()
        return True

    def _drain_background_evals(self) -> None:
        """Run pending background-camera ML on the engine thread.

        Camera JPEG workers only set a flag. YOLO stays on the thread that
        loaded the model so grid encoding cannot deadlock inside predict().
        """
        if self.model is None or not self.running:
            return
        with self.camera_pool._lock:
            workers = list(self.camera_pool._workers.values())
        for worker in workers:
            if worker.is_active_ai:
                worker._bg_eval_pending = False
                continue
            if not getattr(worker, "_bg_eval_pending", False):
                continue
            frame = getattr(worker, "_bg_eval_frame", None)
            if frame is None:
                pkt = worker.latest_packet or worker.grabber.peek_latest_frame()
                if pkt is None or pkt.frame is None:
                    continue
                frame = pkt.frame
            ran = self._evaluate_background_camera(
                worker.camera_id, frame, dict(worker.cfg)
            )
            if ran:
                worker._bg_eval_pending = False
                worker._bg_eval_frame = None
                return

    def acknowledge_event(self, event_id: str | None = None) -> None:
        with self.lock:
            if self.latest_camera_event is not None:
                if not event_id or self.latest_camera_event.get("event_id") == event_id:
                    self.latest_camera_event["handled"] = True

    def garage_telemetry(self) -> dict[str, Any]:
        with self.lock:
            bays = list(self.bay_telemetry or self.bay_manager.telemetry())
            cfg = dict(self.cfg)
            fps = self.fps
            staff = list(self.staff_names)
            identities = list(self.identities)
            last_seen = dict(self.last_face_seen)
            latest_event = dict(self.latest_camera_event) if self.latest_camera_event else None
        wifi_rows = self.wifi.snapshot()
        wifi_by_name = {row["name"]: row for row in wifi_rows}
        now = time.time()
        for bay in bays:
            name = bay.get("mechanic_name")
            face_recent = bool(name) and (now - last_seen.get(name, 0)) < 30
            wifi_on = self.wifi.is_connected(name) if name else False
            bay["presence"] = presence_status(face_recent, wifi_on)
            bay["wifi_connected"] = wifi_on
        active = sum(1 for bay in bays if bay.get("state") != "EMPTY")
        gname = garage_name_of(cfg)
        gopen = shop_is_open(cfg)
        return {
            "bays": bays,
            "fps": fps,
            "garage_name": gname,
            "shop_open": gopen,
            "garage": {
                "name": gname,
                "shop_open": gopen,
                "active_bay_count": active,
                "total_bays": len(bays),
            },
            "open_time": str(cfg.get("open_time") or "08:00"),
            "close_time": str(cfg.get("close_time") or "18:00"),
            "active_bay_count": active,
            "total_bays": len(bays),
            "staff_names": staff,
            "identities": identities,
            "wifi_devices": wifi_rows,
            "roi": list(cfg.get("roi") or [0.30, 0.20, 0.40, 0.60]),
            "active_camera_id": str(cfg.get("active_camera_id") or ""),
            "active_roi_cameras": list(self.active_roi_cameras.keys()),
            "latest_camera_event": latest_event,
        }

    def workplace_telemetry(self) -> dict[str, Any]:
        data = self.garage_telemetry()
        workplace = parse_workplace_id(self.cfg.get("workplace_type"))
        data["workplace_type"] = workplace
        data["zones"] = list(data.get("bays") or [])
        if workplace == "massage":
            data["bays"] = self.visit_monitor.telemetry()
            data["zones"] = data["bays"]
            counts = customer_visit_counts(self.conn)
            counts["open_sessions"] = self.visit_monitor.open_sessions()
            data["visits"] = counts
        return data

    def _vehicle_weights_path(self) -> Path:
        path = get_resource_path("yolo11n_improved.pt")
        if not path.exists():
            path = DATA_DIR / "yolo11n_improved.pt"
        if not path.exists():
            path = get_resource_path("yolo11n.pt")
        if not path.exists():
            path = DATA_DIR / "yolo11n.pt"
        return path

    def _sync_pipeline_models(self) -> None:
        from graph.compile import pipeline_runtime_flags

        flags = pipeline_runtime_flags(self.cfg)
        if flags["vehicle_detect"]:
            if getattr(self, "vehicle_model", None) is None:
                try:
                    from ultralytics import YOLO

                    path = self._vehicle_weights_path()
                    self.vehicle_model = YOLO(str(path) if path.exists() else "yolo11n.pt", task="detect")
                    print("[Pipeline] Vehicle YOLO loaded", flush=True)
                except Exception as exc:
                    print(f"[Pipeline] Vehicle YOLO not loaded: {exc}", flush=True)
                    self.vehicle_model = None
            return
        if getattr(self, "vehicle_model", None) is not None:
            print("[Pipeline] Vehicle YOLO unloaded (no VehicleDetect node)", flush=True)
        self.vehicle_model = None
        self.cached_vehicles = []

    def apply_pipeline(self, graph: Any) -> dict[str, Any]:
        from graph.compile import compile_to_config, validate_graph

        errors = validate_graph(graph if isinstance(graph, dict) else None)
        if errors:
            return {"ok": False, "errors": errors, "pipeline_valid": False}
        try:
            patch = compile_to_config(graph)
        except ValueError as exc:
            return {"ok": False, "errors": [str(exc)], "pipeline_valid": False}

        existing_bays = [b for b in (self.cfg.get("bays") or []) if isinstance(b, dict)]
        existing_by_id = {str(b.get("id")): b for b in existing_bays}
        existing_by_name = {str(b.get("name")): b for b in existing_bays}
        compiled_bays = patch.get("bays")
        if isinstance(compiled_bays, list):
            merged: list[dict[str, Any]] = []
            for zone in compiled_bays:
                if not isinstance(zone, dict):
                    continue
                prev = existing_by_id.get(str(zone.get("id"))) or existing_by_name.get(str(zone.get("name")))
                if prev:
                    compiled_roi = list(zone.get("roi") or [])
                    if compiled_roi == [0.2, 0.2, 0.3, 0.4] and prev.get("roi"):
                        zone = {**zone, "roi": list(prev["roi"])}
                    if prev.get("polygon") and not zone.get("polygon"):
                        zone = {**zone, "polygon": prev.get("polygon")}
                merged.append(zone)
            patch["bays"] = merged

        with self.lock:
            self.cfg.update(patch)
            self.cfg["pipeline_graph"] = graph
            parsed = parse_bays(
                self.cfg.get("bays"),
                fallback_roi=parse_roi(self.cfg.get("roi")),
                seed_if_empty=False,
                workplace=patch.get("workplace_type"),
            )
            if parsed:
                self.cfg["bays"] = parsed
                self.bay_manager.set_bays(parsed)
                self.visit_monitor.set_zones(parsed)
            cfg_to_save = dict(self.cfg)
        self._sync_pipeline_models()
        save_config(cfg_to_save)
        return {
            "ok": True,
            "pipeline_valid": True,
            "config": {
                "workplace_type": patch.get("workplace_type"),
                "person_detect": patch.get("person_detect"),
                "vehicle_detect": patch.get("vehicle_detect"),
                "face_id_enabled": patch.get("face_id_enabled"),
                "reid_enabled": patch.get("reid_enabled"),
                "employee_labor": patch.get("employee_labor"),
                "customer_visits": patch.get("customer_visits"),
            },
        }

    def garage_scorecard(self) -> dict[str, Any]:
        cfg = dict(self.cfg)
        day = datetime.now().date()
        conn = connect(DATA_DIR / "events.db", check_same_thread=False)
        try:
            summary = get_daily_garage_summary(
                conn,
                day,
                open_time=str(cfg.get("open_time") or "08:00"),
                close_time=str(cfg.get("close_time") or "18:00"),
                bay_ids=[b["id"] for b in (cfg.get("bays") or DEFAULT_BAYS)],
            )
        finally:
            conn.close()
        wifi_rows = self.wifi.snapshot()
        wifi_by = {row["name"].lower(): row for row in wifi_rows}
        live = {b.get("mechanic_name"): b for b in self.garage_telemetry().get("bays") or [] if b.get("mechanic_name")}
        now = time.time()
        technicians = []
        seen_names: set[str] = set()
        for tech in summary["technicians"]:
            name = str(tech.get("staff_name") or "").strip()
            if not name or name.lower() in seen_names:
                continue
            seen_names.add(name.lower())
            wifi = wifi_by.get(name.lower(), {})
            face_recent = (now - self.last_face_seen.get(name, 0)) < 45
            wifi_on = bool(wifi.get("connected"))
            bay = live.get(name) or {}
            if tech["clocked_in"] and bay.get("state") == "WORKING":
                attendance = f"Active in {bay.get('name') or bay.get('bay_id')}"
            elif tech["clocked_in"] and bay.get("state") == "IDLE":
                attendance = "On Break"
            elif tech["clocked_in"] and wifi_on:
                attendance = "Clocked In"
            elif tech["clocked_in"]:
                attendance = "Clocked In"
            else:
                attendance = "Clocked Out"
            if not wifi_on and tech["clocked_in"] and not face_recent:
                attendance = "Off-site break"
            score = float(tech["performance_score"])
            technicians.append(
                {
                    **tech,
                    "attendance": attendance,
                    "wifi_connected": wifi_on,
                    "presence": presence_status(face_recent, wifi_on),
                    "current_bay": bay.get("bay_id"),
                    "wrench_pct": score,
                    "badge": "high" if score >= 70 else ("normal" if score >= 40 else "low"),
                }
            )
        enrolled = []
        try:
            enrolled = [row["name"] for row in list_identities(cfg)]
        except Exception:
            enrolled = list({t["staff_name"] for t in technicians})
        return {
            "date": summary["date"],
            "garage_name": garage_name_of(cfg),
            "shop": summary["shop"],
            "technicians": technicians,
            "bays": summary["bays"],
            "wifi_devices": wifi_rows,
            "enrolled": enrolled,
            "attendance_logs": technicians,
        }

    def apply_telegram_link(self, chat_id: str) -> dict[str, Any]:
        chat = normalize_chat_id(chat_id)
        if not chat:
            return {"success": False, "error": "chat_id required."}
        self.telegram_links.set_active_chat(
            chat,
            str(self.cfg.get("venue") or self.cfg.get("garage_name") or "Operator"),
        )
        return self.apply_hub_settings({"telegram_chat_id": chat})

    def begin_telegram_link(
        self, user_id: str, display_name: str = "", *, open_client: bool = False
    ) -> dict[str, Any]:
        token = str(self.cfg.get("telegram_bot_token") or "").strip()
        if not token:
            return {
                "ok": False,
                "error": "TELEGRAM_BOT_TOKEN is not configured on the engine.",
            }
        self.telegram_links.configure(token)
        result = self.telegram_links.begin_link(user_id, display_name)
        status = self.telegram_links.status()
        result["telegram_chat_id"] = self.cfg.get("telegram_chat_id") or ""
        result["status"] = status
        opened = False
        if open_client and result.get("ok"):
            # Prefer desktop protocol on the host machine running the engine.
            opened = open_host_url(str(result.get("app_link") or ""))
            if not opened:
                opened = open_host_url(str(result.get("deep_link") or ""))
        result["opened_on_host"] = opened
        return result

    def telegram_link_status(self, user_id: str = "") -> dict[str, Any]:
        status = self.telegram_links.status()
        consumed = None
        if user_id:
            consumed = self.telegram_links.consume_link_for_user(user_id)
            if consumed:
                self.apply_telegram_link(consumed["chat_id"])
                status = self.telegram_links.status()
                status["just_linked"] = consumed
        status["telegram_chat_id"] = self.cfg.get("telegram_chat_id") or ""
        status["telegram_bot_token_set"] = bool(str(self.cfg.get("telegram_bot_token") or "").strip())
        return status

    def redacted_config(self) -> dict[str, Any]:
        cfg = dict(read_config())
        cfg["telegram_bot_configured"] = bool(str(cfg.get("telegram_bot_token") or "").strip())
        cfg["telegram_bot_username"] = self.telegram_links.status().get("bot_username") or ""
        cfg["telegram_bot_token"] = ""
        return cfg

    def set_orient(
        self,
        rotate: Any = None,
        flip: Any = None,
        bays: Any = None,
        roi: Any = None,
    ) -> dict[str, Any]:
        with self.lock:
            old_rotate = self.cfg.get("rotate")
            old_flip = self.cfg.get("flip")

            if rotate is not None:
                if is_auto_rotate(rotate):
                    self.cfg["rotate"] = "auto"
                else:
                    self.cfg["rotate"] = parse_rotate(rotate)
            if flip is not None:
                self.cfg["flip"] = parse_flip(flip)

            new_rotate = self.cfg.get("rotate")
            new_flip = self.cfg.get("flip")

            if bays is not None:
                parsed_bays = parse_bays(bays, fallback_roi=parse_roi(roi), seed_if_empty=False)
                self.cfg["bays"] = parsed_bays
                if parsed_bays:
                    self.cfg["roi"] = list(parsed_bays[0]["roi"])
                elif roi is not None:
                    pr = parse_roi(roi)
                    if pr:
                        self.cfg["roi"] = pr
                self.bay_manager.set_bays(self.cfg["bays"])
                self.bay_telemetry = self.bay_manager.telemetry()
            elif (new_rotate != old_rotate or new_flip != old_flip):
                cur_bays = parse_bays(self.cfg.get("bays"), seed_if_empty=False)
                if cur_bays:
                    def _safe_rot(r):
                        try:
                            return int(r) % 360
                        except (TypeError, ValueError):
                            return 0

                    delta_r = 0
                    if not is_auto_rotate(new_rotate) and not is_auto_rotate(old_rotate):
                        old_r = _safe_rot(old_rotate)
                        new_r = _safe_rot(new_rotate)
                        delta_r = ((new_r - old_r) % 360 + 360) % 360

                    transformed = []
                    for b in cur_bays:
                        tb = dict(b)
                        if delta_r != 0:
                            tb = transform_bay_geometry(tb, lambda x, y, dr=delta_r: transform_point_rotate(x, y, dr))
                        if old_flip != new_flip:
                            h_toggled = (old_flip == "h" or new_flip == "h") and (old_flip != new_flip)
                            v_toggled = (old_flip == "v" or new_flip == "v") and (old_flip != new_flip)
                            if h_toggled:
                                tb = transform_bay_geometry(tb, transform_point_hflip)
                            if v_toggled:
                                tb = transform_bay_geometry(tb, transform_point_vflip)
                        transformed.append(tb)

                    self.cfg["bays"] = transformed
                    if transformed:
                        self.cfg["roi"] = list(transformed[0]["roi"])
                    self.bay_manager.set_bays(self.cfg["bays"])
                    self.bay_telemetry = self.bay_manager.telemetry()

            cameras = _normalize_cameras(self.cfg.get("cameras"), workplace=self.cfg.get("workplace_type"))
            active = str(self.cfg.get("active_camera_id") or "")
            for cam in cameras:
                if cam["id"] == active:
                    cam["rotate"] = self.cfg.get("rotate")
                    cam["flip"] = self.cfg.get("flip")
                    if "bays" in self.cfg:
                        cam["bays"] = self.cfg.get("bays")
                    if "roi" in self.cfg:
                        cam["roi"] = self.cfg.get("roi")
                    break
            self.cfg["cameras"] = cameras
            cfg_to_save = dict(self.cfg)

        save_config(cfg_to_save)

        rot_val = cfg_to_save.get("rotate")
        try:
            rot_int = int(rot_val) if rot_val != "auto" else 0
        except (ValueError, TypeError):
            rot_int = 0
        flp_val = str(cfg_to_save.get("flip") or "none")

        if hasattr(self, "grabber") and self.grabber is not None:
            self.grabber.output_rotate = rot_int
            self.grabber.output_flip = flp_val

        if hasattr(self, "camera_pool") and self.camera_pool is not None:
            active_cid = str(self.cfg.get("active_camera_id") or "")
            worker = self.camera_pool.get_worker(active_cid)
            if worker is not None:
                worker.update_cfg({"rotate": rot_val, "flip": flp_val})
                if hasattr(worker, "grabber") and worker.grabber is not None:
                    worker.grabber.output_rotate = rot_int
                    worker.grabber.output_flip = flp_val

        return {
            "success": True,
            "rotate": cfg_to_save.get("rotate"),
            "flip": cfg_to_save.get("flip"),
            "bays": cfg_to_save.get("bays"),
            "roi": cfg_to_save.get("roi"),
            "cameras": cameras,
            "active_camera_id": active,
        }

    def _sync_connection_from_grabber(self) -> tuple[str, str | None, int]:
        grabber = self.grabber
        state = grabber.connection_state
        error = grabber.error
        generation = grabber.generation
        self.connection_state = state
        if error:
            self.error_message = error
        if state == "CONNECTING":
            self.status_text = "CONNECTING"
        elif state == "RECONNECTING":
            self.status_text = "RECONNECTING..."
        elif state == "FAILED":
            self.status_text = "FAILED"
        return state, error, generation

    def _worker_loop(self):
        from ultralytics import YOLO

        print("[LiveStreamEngine] Loading person pose model...")
        self.conn = connect(DATA_DIR / "events.db")
        self.visit_monitor.conn = self.conn
        self.runtime_profile = resolve_runtime(self.cfg)
        weights_path = resolve_weights_file(self.cfg, get_resource_path, DATA_DIR)
        veh_weights_path = get_resource_path("yolo11n_improved.pt")
        if not veh_weights_path.exists():
            veh_weights_path = DATA_DIR / "yolo11n_improved.pt"
        if not veh_weights_path.exists():
            veh_weights_path = get_resource_path("yolo11n.pt")
        if not veh_weights_path.exists():
            veh_weights_path = DATA_DIR / "yolo11n.pt"
        try:
            from rtmpose import load_person_pose_model, resolve_models_dir

            self.model = load_person_pose_model(
                self.runtime_profile,
                models_dir=resolve_models_dir(get_resource_path, DATA_DIR),
                weights_path=weights_path,
                cfg=self.cfg,
                yolo_cls=YOLO,
            )

            if getattr(self.model, "task", None) != "pose":
                fallback = get_resource_path("yolo11n-pose.pt")
                print(
                    f"[LiveStreamEngine] Person weights {weights_path} are "
                    f"task={getattr(self.model, 'task', None)!r}, not pose; "
                    f"falling back to {fallback}",
                    flush=True,
                )
                weights_path = fallback
                self.model = YOLO(str(weights_path), task="pose")
            from graph.compile import pipeline_runtime_flags

            boot_flags = pipeline_runtime_flags(self.cfg)
            if boot_flags["vehicle_detect"]:
                self.vehicle_model = YOLO(
                    str(veh_weights_path) if veh_weights_path.exists() else "yolo11n.pt",
                    task="detect",
                )
            else:
                self.vehicle_model = None
                print("[LiveStreamEngine] Vehicle YOLO skipped (pipeline has no VehicleDetect)", flush=True)
            self.cached_vehicles: list = []
            self.last_vehicle_infer: float = 0.0
            self.vehicle_infer_interval: float = float(self.cfg.get("vehicle_infer_interval", 1.0))
            print(
                f"[LiveStreamEngine] Models ready ({self.runtime_profile.name}, "
                f"engine={getattr(self.runtime_profile, 'pose_engine', 'rtmpose')}, "
                f"device={self.runtime_profile.yolo_device}, weights={weights_path})"
            )
            self.face_rec = try_create_face_recognizer(self.cfg)
            self.reid = try_create_body_reid(self.cfg)
            self.liveness_probe = LivenessProbe()
            self.tracker = PersonTracker(
                max_age=self.runtime_profile.track_max_age,
                min_hits=self.runtime_profile.track_min_hits,
                iou_threshold=self.runtime_profile.track_iou_threshold,
                reid_threshold=self.runtime_profile.reid_match_threshold,
                gallery=self.shared_gallery,
                camera_id=str(self.cfg.get("active_camera_id") or "cam-1"),
            )
        except Exception as e:
            self.error_message = f"Failed to load YOLO model: {e}"
            with self.lock:
                self.connection_state = "FAILED"
                self.status_text = "FAILED"
            return

        proofs = DATA_DIR / "proofs"
        session_gen = -1
        source: int | str = 0
        absent = 10.0
        interval = 1.0 / 8.0
        person_conf = resolve_person_conf(self.cfg)
        min_person_height = resolve_min_person_height(self.cfg)
        min_aspect = resolve_min_aspect(self.cfg)
        min_keypoints = resolve_min_keypoints(self.cfg)
        kpt_conf = resolve_kpt_conf(self.cfg)
        imgsz = resolve_imgsz(self.cfg, profile=self.runtime_profile)
        ghost = GhostCounter(absent, 30.0)
        last_accepted: list[Detection] = []
        last_rejected: list[Detection] = []
        last_state = GhostState(False, 0.0, False)
        last_infer = 0.0
        prev_gray = None
        frame_count = 0
        t_fps = time.time()
        clock_out_grace = 600.0
        if self.conn is not None:
            summary = get_daily_garage_summary(
                self.conn,
                datetime.now().date(),
                bay_ids=[b["id"] for b in self.bay_manager.configs()],
            )
            self.bay_manager.hydrate_today(summary.get("bays") or [])

        while self.running:
            with self.lock:
                streaming = self.is_streaming
                cfg = dict(self.cfg)

            if not streaming:
                session_gen = -1
                time.sleep(0.2)
                continue

            state, err, generation = self._sync_connection_from_grabber()
            if state == "FAILED" and generation != session_gen:
                with self.lock:
                    if self.grabber.generation != generation:
                        continue
                    session_gen = generation
                    self.is_streaming = False
                    self.current_frame_jpeg = None
                    self.error_message = err
                    self.status_text = "FAILED"
                    self.connection_state = "FAILED"
                    self.new_frame_event.set()
                continue

            packet = self.grabber.get_latest_frame(timeout=0.1)
            if packet is None:
                continue

            generation = self.grabber.generation
            active_cid = str(self.cfg.get("active_camera_id") or "")
            if generation != session_gen or active_cid != getattr(self, "_last_worker_cam_id", None):
                session_gen = generation
                self._last_worker_cam_id = active_cid
                source = parse_source(cfg.get("source", 0))
                absent = float(cfg.get("absent_seconds") or 10)
                cooldown = float(cfg.get("cooldown_seconds") or 30)
                detect_fps = max(0.5, float(cfg.get("detect_fps") or 8.0))
                interval = 1.0 / detect_fps
                person_conf = resolve_person_conf(cfg)
                min_person_height = resolve_min_person_height(cfg)
                min_aspect = resolve_min_aspect(cfg)
                min_keypoints = resolve_min_keypoints(cfg)
                kpt_conf = resolve_kpt_conf(cfg)
                imgsz = resolve_imgsz(cfg, self.runtime_profile)
                confirm = float(
                    cfg.get("occupy_confirm_seconds")
                    if cfg.get("occupy_confirm_seconds") is not None
                    else 1.0
                )
                clear = resolve_occupy_clear_seconds(cfg)
                clock_out_grace = float(cfg.get("clock_out_seconds") or 600)
                idle_hold = float(cfg.get("idle_stationary_seconds") or 120)
                print(f"[LiveStreamEngine] Ingesting camera stream: {redact_source(source)}")
                ghost = GhostCounter(absent, cooldown)
                self.bay_manager.confirm = confirm
                self.bay_manager.clear = clear
                self.bay_manager.idle_stationary_seconds = idle_hold
                self.bay_manager.under_car_grace_seconds = resolve_under_car_grace_seconds(cfg)
                self.bay_manager.set_bays(cfg.get("bays"), fallback_roi=parse_roi(cfg.get("roi")))
                last_accepted = []
                last_rejected = []
                last_state = GhostState(False, 0.0, False)
                last_infer = 0.0
                frame_count = 0
                t_fps = time.time()
                if self.tracker is not None:
                    self.tracker.camera_id = str(self.cfg.get("active_camera_id") or "cam-1")
                    self.tracker.reset(clear_gallery=False)
                rotate_deg, flip = resolve_orient(cfg.get("rotate"), source, packet.frame, cfg.get("flip"))
                first_oriented = orient_frame(packet.frame, rotate_deg, flip)
                h0, w0 = first_oriented.shape[:2]
                with self.lock:
                    self.error_message = None
                    self.status_text = "CONNECTED"
                    self.connection_state = "CONNECTED"
                    self.stream_resolution = f"{w0}x{h0}"

            with self.lock:
                venue = garage_name_of(self.cfg)
                rotate_now = self.cfg.get("rotate")
                flip_now = self.cfg.get("flip")

            frame = packet.frame
            rotate_deg, flip = resolve_orient(rotate_now, source, frame, flip_now)
            frame = orient_frame(frame, rotate_deg, flip)
            h, w = frame.shape[:2]
            self.stream_resolution = f"{w}x{h}"

            now = time.time()
            frame_count += 1
            if now - t_fps >= 1.0:
                self.fps = frame_count / (now - t_fps)
                frame_count = 0
                t_fps = now

            snapshots = self.bay_manager.snapshots()
            any_occupied = any(s.state != "EMPTY" for s in snapshots)
            protected_ids = self.bay_manager.protected_track_ids()
            if self.tracker is not None:
                self.tracker.protected_ids = set(protected_ids)

            # Fast Motion Scanner (<0.05ms CPU on 160x120 grayscale)
            try:
                small_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120))
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, small_gray)
                    motion_score = float(cv2.mean(diff)[0])
                else:
                    motion_score = 10.0
                prev_gray = small_gray
            except Exception as ex:
                motion_score = 10.0

            # Dynamic AI Cadence: High FPS during active motion/work; Low-compute Sleep during static wait
            has_active_work = any(s.state in ("WORKING", "UNDER_VEHICLE") for s in snapshots)
            has_open_session = any(getattr(s, "session_open", False) for s in snapshots)
            has_detected_person = len(last_accepted) > 0
            if motion_score > 1.2 or has_active_work or has_open_session or has_detected_person:
                dynamic_interval = interval  # Full high-cadence AI while a person/worker is detected
            elif any(s.state in ("PARKED_WAITING", "ON_BREAK") for s in snapshots):
                dynamic_interval = 2.0  # Low-compute check (0.5 FPS during customer consultation wait)
            else:
                dynamic_interval = 3.0  # Idle Sleep Mode (wakes up instantly on motion)

            active_cam_cfg = next((c for c in (self.cfg.get("cameras") or []) if c.get("id") == active_cid), {})
            active_ml_enabled = bool(active_cam_cfg.get("ml_enabled", True))
            from graph.compile import pipeline_runtime_flags

            flags = pipeline_runtime_flags(self.cfg)

            if not flags["valid"]:
                self.status_text = "PIPELINE INVALID"
                last_accepted = []
                last_rejected = []
            elif not active_ml_enabled:
                self.status_text = "AI DISABLED"
                last_accepted = []
                last_rejected = []
            elif self.force_infer or now - last_infer >= dynamic_interval:
                last_infer = now
                self.force_infer = False
                try:
                    t_pred = time.perf_counter()
                    result = None
                    with self.infer_lock:
                        if flags["person_detect"]:
                            result = self.model.predict(
                                frame,
                                imgsz=imgsz,
                                conf=person_conf,
                                device=self.runtime_profile.yolo_device if self.runtime_profile else None,
                                verbose=False,
                            )[0]
                        vehicles = getattr(self, "cached_vehicles", [])
                        vehicle_interval = getattr(self, "vehicle_infer_interval", 1.0)
                        should_infer_vehicle = (
                            bool(flags["vehicle_detect"])
                            and getattr(self, "vehicle_model", None) is not None
                            and (self.force_infer or not vehicles or (now - getattr(self, "last_vehicle_infer", 0.0)) >= vehicle_interval)
                        )
                        if should_infer_vehicle:
                            try:
                                veh_res = self.vehicle_model.predict(
                                    frame,
                                    imgsz=512,
                                    classes=[2, 3, 5, 7],
                                    conf=0.18,
                                    device=None,
                                    verbose=False,
                                )[0]
                                vehicles = extract_vehicle_detections(veh_res, w, h, conf_min=0.18)
                                self.cached_vehicles = vehicles
                                self.last_vehicle_infer = now
                            except Exception as ex:
                                print(f"[VehicleInfer] Prediction error: {ex}")
                        elif not flags["vehicle_detect"]:
                            vehicles = []
                            self.cached_vehicles = []

                    auto_create = bool(self.cfg.get("auto_create_bays", False))
                    departed_bays = self.bay_manager.sync_auto_vehicles(
                        vehicles, w, h, now=now, auto_create=auto_create
                    )
                    if departed_bays and self.conn is not None:
                        for dep_id in departed_bays:
                            for bay_cfg in self.bay_manager.configs():
                                if bay_cfg.get("id") == dep_id and bay_cfg.get("job_id"):
                                    try:
                                        complete_vehicle_job(self.conn, bay_cfg["job_id"])
                                        eval_report = evaluate_completed_vehicle_job(self.conn, bay_cfg["job_id"])
                                        if eval_report:
                                            print(f"[Performance Evaluation] Job {eval_report.job_id}: Grade={eval_report.performance_grade}, Score={eval_report.performance_score}, Efficiency={eval_report.efficiency_pct:.1f}% by {eval_report.primary_technician}")
                                    except Exception as ex:
                                        print(f"[Vehicle Departure] Error completing/evaluating job {bay_cfg.get('job_id')}: {ex}")
                    track_low_thresh = float(self.cfg.get("track_low_thresh", 0.10))
                    if flags["person_detect"] and result is not None:
                        last_accepted, last_rejected, low_dets = person_detections_split(
                            result,
                            h,
                            conf_min=person_conf,
                            track_low_thresh=track_low_thresh,
                            min_height_frac=min_person_height,
                            min_aspect=min_aspect,
                            min_keypoints=min_keypoints,
                            kpt_conf=kpt_conf,
                        )
                        zoom_bays = self.bay_manager.configs() or (cfg.get("bays") or [])
                        with self.infer_lock:
                            last_accepted, last_rejected = self._apply_bay_zoom(
                                frame,
                                zoom_bays,
                                last_accepted,
                                last_rejected,
                                imgsz=imgsz,
                                person_conf=person_conf,
                                min_person_height=min_person_height,
                                min_aspect=min_aspect,
                                min_keypoints=min_keypoints,
                                kpt_conf=kpt_conf,
                                occupancy_by_id=occupancy_hints(snapshots),
                            )
                        last_accepted, vetoed = veto_vehicle_interior(
                            last_accepted, vehicles, kpt_conf=kpt_conf, protected_ids=protected_ids
                        )
                        last_rejected.extend(vetoed)
                    else:
                        last_accepted, last_rejected, low_dets = [], [], []
                    if self.tracker is not None:
                        last_accepted = run_identity_pipeline(
                            frame,
                            last_accepted,
                            self.tracker,
                            face_rec=self.face_rec if flags["face_id"] else None,
                            reid=self.reid if flags["reid"] else None,
                            probe=self.liveness_probe,
                            low_detections=low_dets,
                        )
                        self._bank_hard_negatives(frame, self.tracker.clutter_events)
                    elif flags["face_id"] and self.face_rec is not None:
                        self.face_rec.annotate_detections(frame, last_accepted)
                    if flags["customer_visits"]:
                        prev_conn = self.visit_monitor.conn
                        if not flags["persist"]:
                            self.visit_monitor.conn = None
                        try:
                            visit_snaps = self.visit_monitor.update(last_accepted, w, h, now)
                        finally:
                            self.visit_monitor.conn = prev_conn
                        self.person_count = len(last_accepted)
                        self.identities = [det.identity or "Visitor" for det in last_accepted]
                        self.staff_names = []
                        occupied_zones = [s for s in visit_snaps if s.occupied]
                        self.status_text = (
                            f"VISITORS [{len(occupied_zones)} zones]"
                            if occupied_zones
                            else "FLOOR EMPTY"
                        )
                        self.is_occupied = bool(occupied_zones)
                        self.empty_elapsed = 0.0 if occupied_zones else self.empty_elapsed
                        with self.lock:
                            self.bay_telemetry = [s.as_dict() for s in visit_snaps]
                        last_state = GhostState(self.is_occupied, self.empty_elapsed, False)
                    if flags["employee_labor"]:
                        snapshots = self.bay_manager.update(
                            last_accepted, w, h, now, kpt_conf=kpt_conf, frame=frame
                        )
                        if self.tracker is not None:
                            # Spare still workers in an active bay from the inanimate sweep.
                            self.tracker.protected_ids = self.bay_manager.protected_track_ids()
                        ghost.absent_seconds = float(cfg.get("absent_seconds") or 10)
                        ghost.cooldown_seconds = float(cfg.get("cooldown_seconds") or 30)
                        any_occupied = any(
                            getattr(s, "person_present", False)
                            or s.state in ("WORKING", "UNDER_VEHICLE", "IDLE", "NOT_WORKING")
                            for s in snapshots
                        )
                        last_state = ghost.update(any_occupied, now)
                        stamp = datetime.now()
                        if flags["persist"]:
                            self._record_garage_tick(last_accepted, snapshots, last_state, stamp, now, clock_out_grace)

                        self.is_occupied = last_state.occupied
                        self.empty_elapsed = last_state.empty_elapsed
                        self.person_count = len(last_accepted)
                        self.staff_names = [
                            det.identity for det in last_accepted if det.is_staff and det.identity
                        ]
                        self.identities = [det.identity or "person" for det in last_accepted]
                        working = [s for s in snapshots if s.state == "WORKING"]
                        if working:
                            names = ", ".join(s.mechanic_name or s.name for s in working)
                            self.status_text = f"WRENCH TIME [{names}]"
                        elif any_occupied:
                            self.status_text = till_status_label(
                                True,
                                last_accepted,
                                self.empty_elapsed,
                                absent,
                                face_id_enabled=flags["face_id"] and self.face_rec is not None,
                            )
                        else:
                            self.status_text = "SHOP FLOOR EMPTY"
                        with self.lock:
                            self.bay_telemetry = [s.as_dict() for s in snapshots]
                    elif not flags["customer_visits"]:
                        self.status_text = "PIPELINE IDLE"

                    # Multi-Camera Concurrent ROI Tracking
                    if self.active_roi_cameras:
                        bg_cids = list(self.active_roi_cameras.keys())
                        for bg_cid in bg_cids:
                            if bg_cid == active_cid:
                                continue
                            if (now - self._last_bg_roi_infer.get(bg_cid, 0.0)) < 1.0:
                                continue
                            self._last_bg_roi_infer[bg_cid] = now
                            bg_worker = self.camera_pool.get_worker(bg_cid)
                            if bg_worker is None or not bg_worker.cfg.get("ml_enabled", True):
                                self.active_roi_cameras.pop(bg_cid, None)
                                self._last_bg_roi_infer.pop(bg_cid, None)
                                continue
                            bg_pkt = bg_worker.grabber.peek_latest_frame()
                            if bg_pkt is None or bg_pkt.frame is None:
                                bg_pkt = bg_worker.latest_packet
                            if bg_pkt is None or bg_pkt.frame is None:
                                continue
                            bg_pkt_jpeg = getattr(bg_pkt, "jpeg", None)
                            if bg_pkt_jpeg and (bg_pkt.timestamp or now) > (bg_worker.latest_ts or 0.0):
                                with bg_worker._lock:
                                    bg_worker.latest_jpeg = bg_pkt_jpeg
                                    bg_worker.latest_ts = bg_pkt.timestamp or now
                                    bg_worker.latest_packet = bg_pkt
                            bg_frame = bg_pkt.frame
                            bg_h, bg_w = bg_frame.shape[:2]
                            bg_bays = bg_worker.cfg.get("bays") or []
                            if not bg_bays:
                                bg_roi = parse_roi(bg_worker.cfg.get("roi")) or [0.3, 0.2, 0.4, 0.6]
                                bg_bays = [{"id": f"{bg_cid}_bay", "name": "Bay 1", "roi": bg_roi}]
                            try:
                                with self.infer_lock:
                                    bg_res = self.model.predict(
                                        bg_frame,
                                        imgsz=imgsz,
                                        conf=person_conf,
                                        device=self.runtime_profile.yolo_device if self.runtime_profile else None,
                                        verbose=False,
                                    )[0]
                                bg_persons, bg_rejected = person_detections(
                                    bg_res,
                                    bg_h,
                                    conf_min=person_conf,
                                    min_height_frac=min_person_height,
                                    min_aspect=min_aspect,
                                    min_keypoints=min_keypoints,
                                    kpt_conf=kpt_conf,
                                )
                                with self.infer_lock:
                                    bg_persons, bg_rejected = self._apply_bay_zoom(
                                        bg_frame,
                                        bg_bays,
                                        bg_persons,
                                        bg_rejected,
                                        imgsz=imgsz,
                                        person_conf=person_conf,
                                        min_person_height=min_person_height,
                                        min_aspect=min_aspect,
                                        min_keypoints=min_keypoints,
                                        kpt_conf=kpt_conf,
                                    )
                                bg_in_roi = [
                                    det for det in bg_persons
                                    if any(detection_in_bay(det, b, bg_w, bg_h, kpt_conf=kpt_conf) for b in bg_bays)
                                ]
                                if bg_in_roi:
                                    self.active_roi_cameras[bg_cid] = now
                                else:
                                    last_seen = self.active_roi_cameras.get(bg_cid, 0.0)
                                    if (now - last_seen) >= 3.0:
                                        self.active_roi_cameras.pop(bg_cid, None)
                                        self._last_bg_roi_infer.pop(bg_cid, None)
                                        print(f"[MultiCamROI] Person exited ROI on {bg_cid}. Remaining active: {list(self.active_roi_cameras.keys())}", flush=True)
                            except Exception as ex:
                                print(f"[MultiCamROI] Worker infer error on {bg_cid}: {ex}", flush=True)

                    if self.conn is not None:
                        upsert_minute(
                            self.conn,
                            stamp.strftime("%Y-%m-%d %H:%M"),
                            len(last_accepted),
                            last_state.occupied,
                        )
                        if last_state.occupied and not has_opened_today(self.conn, stamp.date()):
                            insert_event(self.conn, "opened", stamp)

                    if last_state.should_alert:
                        primary = snapshots[0] if snapshots else None
                        roi_px = roi_to_pixels(w, h, primary.roi if primary else [0.3, 0.2, 0.4, 0.6])
                        proof_frame = frame
                        main_src = str(cfg.get("main_source") or "").strip()
                        if main_src:
                            still = request_still(
                                main_src,
                                gateway=self.media,
                                stream_id=self._gateway_main_stream_id,
                                timeout=1.5,
                            )
                            if still is not None:
                                sh, sw = still.shape[:2]
                                roi_px = scale_roi_px(roi_px, (w, h), (sw, sh))
                                proof_frame = still
                        path = save_proof(proof_frame, roi_px, stamp, proofs, kind="idle_bay")
                        if self.conn is not None:
                            insert_event(self.conn, "abandoned", stamp, str(path))
                        caption = (
                            f"{venue}: no active wrench time "
                            f"for {int(absent)}s.\n{stamp.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        print(f"[LiveStreamEngine Alert] Out of ROI alert: {path}")
                        try:
                            if flags["telegram"] and flags["alerts"]:
                                if not self.bot.enabled:
                                    print(f"[LiveStreamEngine Alert] Telegram bot not configured (token={bool(self.bot.token)}, chat_id={bool(self.bot.chat_id)})")
                                sent = self.bot.send_photo(path, caption)
                                if sent:
                                    print(f"[LiveStreamEngine Alert] Successfully sent photo to Telegram")
                                else:
                                    print(f"[LiveStreamEngine Alert] Failed to send photo to Telegram")
                        except Exception as bot_err:
                            print(f"[LiveStreamEngine Alert] Error sending Telegram photo: {bot_err}")
                except Exception as exc:
                    print(f"[LiveStreamEngine Infer Error] {exc}")

            annotated = frame.copy()
            bay_cfgs = self.bay_manager.configs()
            for det in last_rejected:
                det_in_roi = any(detection_in_bay(det, b, w, h, kpt_conf) for b in bay_cfgs)
                draw_detection(
                    annotated, det, in_roi=det_in_roi, kpt_conf=kpt_conf
                )
            for det in last_accepted:
                det_in_roi = any(detection_in_bay(det, b, w, h, kpt_conf) for b in bay_cfgs)
                draw_detection(
                    annotated,
                    det,
                    in_roi=det_in_roi,
                    kpt_conf=kpt_conf,
                )

            # ROI geometry is rendered as an interactive DOM overlay in hub.html.
            # Do not burn boxes, handles, or labels into the JPEG — that duplicates
            # the overlay and leaves a lagging ghost box when the user drags.

            success, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if success:
                jpeg_bytes = buffer.tobytes()
                with self.lock:
                    self.current_frame_jpeg = jpeg_bytes
                    self.current_frame_bgr = frame.copy()
                    self.frame_seq += 1
                self.new_frame_event.set()
                active_worker = self.camera_pool.get_worker(active_cid)
                if active_worker is not None:
                    with active_worker._lock:
                        active_worker.latest_jpeg = jpeg_bytes
                        active_worker.latest_ts = now
                        active_worker.latest_packet = packet
            self._drain_background_evals()

        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _record_garage_tick(
        self,
        detections: list[Detection],
        snapshots,
        last_state: GhostState,
        stamp: datetime,
        now: float,
        clock_out_grace: float,
    ) -> None:
        if self.conn is None:
            return
        for det in detections:
            if det.is_staff and det.identity:
                self.last_face_seen[det.identity] = now
                record_face_clock_in(self.conn, det.identity, stamp)

        occupied_ids: set[str] = set()
        for tick in self.bay_manager.activity_ticks():
            bay_id = tick[0]
            technician = tick[1]
            is_working = tick[2]
            dt = tick[3]
            state = tick[4] if len(tick) > 4 else ("WORKING" if is_working else "IDLE")
            job_id = tick[5] if len(tick) > 5 else None

            if dt <= 0:
                continue
            if state != "EMPTY":
                occupied_ids.add(bay_id)
            named = technician if technician and technician != UNKNOWN_WORKER else None
            if named:
                update_technician_activity(self.conn, named, bay_id, is_working, dt, stamp)

            if bay_id and named:
                active_job = job_id or get_or_create_vehicle_job(
                    self.conn, bay_id, primary_technician=named, timestamp=stamp
                )
                active_dt = dt if state in ("WORKING", "UNDER_VEHICLE") else 0.0
                break_dt = dt if state == "ON_BREAK" else 0.0
                update_vehicle_job_activity(
                    self.conn, active_job, active_dt, break_dt, named, stamp, status=state
                )
        close_empty_bays(self.conn, occupied_ids, stamp)

        wifi_dt = 0.0
        if self._wifi_last_tick is not None:
            wifi_dt = max(0.0, min(now - self._wifi_last_tick, 120.0))
        self._wifi_last_tick = now
        for row in self.wifi.snapshot():
            if row.get("connected") and wifi_dt > 0:
                add_wifi_minutes(self.conn, row["name"], wifi_dt, stamp)

        for departed in self.wifi.departures():
            last = self.last_face_seen.get(departed, 0.0)
            if now - last >= min(clock_out_grace, 90.0):
                record_face_clock_out(self.conn, departed, stamp)

        for name, last in list(self.last_face_seen.items()):
            if now - last < clock_out_grace:
                continue
            if self.wifi.is_connected(name):
                continue
            record_face_clock_out(self.conn, name, stamp)


GLOBAL_ENGINE: LiveStreamEngine | None = None


def init_global_engine() -> LiveStreamEngine:
    """Create the process-wide engine. Deferred on frozen Windows so import
    failures and config errors are logged from ``main()`` instead of crashing
    before the boot banner.
    """
    global GLOBAL_ENGINE
    if GLOBAL_ENGINE is None:
        GLOBAL_ENGINE = LiveStreamEngine()
    return GLOBAL_ENGINE


if not is_frozen():
    init_global_engine()


HUB_HTML_PATH = get_resource_path("hub.html")


def load_hub_html() -> bytes:
    return HUB_HTML_PATH.read_text(encoding="utf-8").encode("utf-8")


def encode_mjpeg_part(frame_bytes: bytes, timestamp: float | None = None) -> bytes:
    """One multipart MJPEG part with Content-Length so libsoup/WebKit can frame it."""
    ts = time.time() if timestamp is None else timestamp
    header = (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        + f"Content-Length: {len(frame_bytes)}\r\n".encode("ascii")
        + f"X-Timestamp: {ts:.3f}\r\n".encode("ascii")
        + b"\r\n"
    )
    return header + frame_bytes + b"\r\n"


class DashboardRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _cors(self) -> None:
        origin = self.headers.get("Origin") or "*"
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Vary", "Origin")

    def end_headers(self):
        self._cors()
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def _identity_cfg(self) -> dict[str, Any]:
        return GLOBAL_ENGINE.cfg or read_config()

    def _identity_photo_url(self, name: str, filename: str) -> str:
        return (
            "/api/identities/"
            + urllib.parse.quote(name)
            + "/photo/"
            + urllib.parse.quote(filename)
        )

    def _enrich_identity(self, row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        thumb = out.get("thumbnail")
        if thumb:
            out["thumbnail"] = self._identity_photo_url(out["name"], str(thumb))
        photos = []
        for photo in out.get("photos") or []:
            fname = photo.get("filename") if isinstance(photo, dict) else None
            if not fname:
                continue
            photos.append({"filename": fname, "url": self._identity_photo_url(out["name"], fname)})
        if "photos" in out:
            out["photos"] = photos
        return out

    def _handle_identities_get(self, parts: list[str]) -> bool:
        cfg = self._identity_cfg()
        try:
            if parts == ["api", "identities"]:
                rows = [self._enrich_identity(row) for row in list_identities(cfg)]
                self._send_json({"identities": rows})
                return True
            if len(parts) == 3 and parts[0] == "api" and parts[1] == "identities":
                row = self._enrich_identity(get_identity(parts[2], cfg))
                self._send_json(row)
                return True
            if (
                len(parts) == 5
                and parts[0] == "api"
                and parts[1] == "identities"
                and parts[3] == "photo"
            ):
                path = identity_photo_path(parts[2], parts[4], cfg)
                data = path.read_bytes()
                mime = "image/jpeg"
                suffix = path.suffix.lower()
                if suffix == ".png":
                    mime = "image/png"
                elif suffix == ".webp":
                    mime = "image/webp"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return True
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, 404)
            return True
        except ValueError as exc:
            self._send_json({"error": str(exc)}, 400)
            return True
        return False

    def _parse_multipart_files(self, content_type: str, body: bytes) -> list[tuple[str, bytes]]:
        from email import policy
        from email.parser import BytesParser

        header = f"MIME-Version: 1.0\r\nContent-Type: {content_type}\r\n\r\n".encode("utf-8")
        msg = BytesParser(policy=policy.default).parsebytes(header + body)
        files: list[tuple[str, bytes]] = []
        if not msg.is_multipart():
            return files
        for part in msg.iter_parts():
            filename = part.get_filename()
            payload = part.get_payload(decode=True)
            if filename and payload:
                files.append((filename, payload))
        return files

    def _handle_identities_write(self, parts: list[str], method: str, payload: dict, files: list[tuple[str, bytes]]) -> bool:
        cfg = self._identity_cfg()
        try:
            if method == "POST" and parts == ["api", "identities"]:
                row = self._enrich_identity(create_identity(str(payload.get("name") or ""), cfg))
                GLOBAL_ENGINE.reload_face_id()
                self._send_json(row, 201)
                return True
            if (
                method == "POST"
                and len(parts) == 4
                and parts[0] == "api"
                and parts[1] == "identities"
                and parts[3] == "photos"
            ):
                saved = []
                errors = []
                for filename, data in files:
                    try:
                        stored = save_identity_photo(parts[2], data, filename, cfg)
                        saved.append(
                            {
                                "filename": stored,
                                "url": self._identity_photo_url(parts[2], stored),
                            }
                        )
                    except Exception as exc:
                        errors.append({"filename": filename, "error": str(exc)})
                GLOBAL_ENGINE.reload_face_id()
                status = 200 if saved else 400
                self._send_json({"saved": saved, "errors": errors}, status)
                return True
            if method == "DELETE" and len(parts) == 3 and parts[0] == "api" and parts[1] == "identities":
                delete_identity(parts[2], cfg)
                GLOBAL_ENGINE.reload_face_id()
                self._send_json({"ok": True})
                return True
            if (
                method == "DELETE"
                and len(parts) == 5
                and parts[0] == "api"
                and parts[1] == "identities"
                and parts[3] == "photo"
            ):
                delete_identity_photo(parts[2], parts[4], cfg)
                GLOBAL_ENGINE.reload_face_id()
                self._send_json({"ok": True})
                return True
        except FileExistsError as exc:
            self._send_json({"error": str(exc)}, 409)
            return True
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, 404)
            return True
        except ValueError as exc:
            self._send_json({"error": str(exc)}, 400)
            return True
        return False

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(load_hub_html())

        elif parsed.path in (
            "/favicon.ico",
            "/favicon.png",
            "/favicon.svg",
            "/inb_surveillance.png",
            "/inb_surveillance.jpg",
            "/INB Surveillance.jpg",
            "/inb_surveillance-removebg-preview.png",
            "/logo.png",
        ):
            filename = parsed.path.lstrip("/")
            names_to_try = [filename]
            if "inb_surveillance" in filename or filename == "logo.png":
                names_to_try = [
                    filename,
                    "inb_surveillance-removebg-preview.png",
                    "inb_surveillance.png",
                    "inb_surveillance.jpg",
                    "INB Surveillance.jpg",
                ]

            search_dirs = [
                ROOT,
                ROOT / "public",
                DATA_DIR,
                Path.cwd(),
                Path.cwd() / "public",
            ]
            if not is_frozen():
                search_dirs.extend(
                    [
                        ROOT.parent,
                        ROOT.parent / "public",
                    ]
                )
            content = None
            resolved_ext = Path(filename).suffix.lower()
            for name in names_to_try:
                for sdir in search_dirs:
                    cand = sdir / name
                    if cand.exists() and cand.is_file():
                        try:
                            content = cand.read_bytes()
                            resolved_ext = cand.suffix.lower()
                            break
                        except Exception:
                            pass
                if content:
                    break

            if content:
                content_type = "image/png"
                if resolved_ext == ".ico":
                    content_type = "image/x-icon"
                elif resolved_ext == ".svg":
                    content_type = "image/svg+xml"
                elif resolved_ext in (".jpg", ".jpeg"):
                    content_type = "image/jpeg"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(404)
                self.end_headers()

        elif parsed.path.startswith("/static/"):
            asset = resolve_static_asset(parsed.path)
            content = None
            if asset is not None:
                try:
                    content = asset.read_bytes()
                except OSError:
                    content = None
            if content is not None and asset is not None:
                suffix = asset.suffix.lower()
                content_type = "application/javascript" if suffix == ".js" else "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(404)
                self.end_headers()

        elif parsed.path == "/api/public-config":
            self._send_json(public_supabase_config())

        elif parsed.path == "/api/auth-session":
            session_file = DATA_DIR / "session.json"
            if session_file.exists():
                try:
                    with session_file.open("r", encoding="utf-8") as f:
                        saved = json.load(f)
                    sess = saved.get("session") if (isinstance(saved, dict) and "session" in saved and isinstance(saved.get("session"), dict)) else saved
                    self._send_json({"ok": True, "session": sess})
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=500)
            else:
                self._send_json({"ok": False, "session": None})

        elif parsed.path == "/api/profile/avatar":
            avatars_dir = DATA_DIR / "avatars"
            avatar_file = avatars_dir / "avatar.jpg"
            if not avatar_file.exists():
                avatar_file = avatars_dir / "avatar.png"
            if avatar_file.exists():
                try:
                    content = avatar_file.read_bytes()
                    ct = "image/png" if avatar_file.suffix.lower() == ".png" else "image/jpeg"
                    self.send_response(200)
                    self.send_header("Content-Type", ct)
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(content)
                    return
                except Exception:
                    pass
            self.send_response(404)
            self.end_headers()

        elif parsed.path == "/api/profile":
            cfg = GLOBAL_ENGINE.cfg or read_config()
            avatar_exists = (DATA_DIR / "avatars" / "avatar.jpg").exists() or (DATA_DIR / "avatars" / "avatar.png").exists()
            avatar_path = f"/api/profile/avatar?t={int(time.time())}" if avatar_exists else None
            session_file = DATA_DIR / "session.json"
            saved_profile = {}
            if session_file.exists():
                try:
                    with session_file.open("r", encoding="utf-8") as f:
                        sdata = json.load(f)
                    saved_profile = sdata.get("profile") or {}
                except Exception:
                    pass
            display_name = saved_profile.get("display_name") or cfg.get("garage_name") or cfg.get("venue") or "Operator"
            venue_name = saved_profile.get("venue_name") or cfg.get("venue") or cfg.get("garage_name") or "Private operator"
            self._send_json({
                "ok": True,
                "profile": {
                    "display_name": display_name,
                    "venue_name": venue_name,
                    "avatar_url": avatar_path,
                }
            })

        elif parsed.path == "/api/config":
            self._send_json(GLOBAL_ENGINE.redacted_config())

        elif parsed.path == "/api/pipeline":
            graph = (GLOBAL_ENGINE.cfg or {}).get("pipeline_graph")
            self._send_json({"ok": True, "graph": graph})

        elif parsed.path == "/api/telemetry":
            grabber = GLOBAL_ENGINE.grabber
            connection = grabber.connection_state
            if connection == "STANDBY":
                connection = GLOBAL_ENGINE.connection_state
            sid = GLOBAL_ENGINE._gateway_stream_id
            garage = GLOBAL_ENGINE.workplace_telemetry()
            protocol = str(
                GLOBAL_ENGINE.cfg.get("protocol")
                or protocol_from_source(GLOBAL_ENGINE.cfg.get("source"))
            )
            data = {
                "occupied": GLOBAL_ENGINE.is_occupied,
                "empty_elapsed": GLOBAL_ENGINE.empty_elapsed,
                "person_count": GLOBAL_ENGINE.person_count,
                "staff_names": GLOBAL_ENGINE.staff_names,
                "identities": GLOBAL_ENGINE.identities,
                "fps": GLOBAL_ENGINE.fps,
                "ingest_fps": grabber.ingest_fps,
                "infer_ms": round(float(GLOBAL_ENGINE.infer_ms or 0.0), 1),
                "protocol": protocol,
                "main_stream": bool(str(GLOBAL_ENGINE.cfg.get("main_source") or "").strip()),
                "status": GLOBAL_ENGINE.status_text,
                "connection": connection,
                "resolution": GLOBAL_ENGINE.stream_resolution,
                "error": grabber.error or GLOBAL_ENGINE.error_message,
                "roi": list(GLOBAL_ENGINE.cfg.get("roi") or [0.30, 0.20, 0.40, 0.60]),
                "bays": garage.get("bays"),
                "garage": {
                    "name": garage.get("garage_name"),
                    "shop_open": garage.get("shop_open"),
                    "active_bay_count": garage.get("active_bay_count"),
                    "total_bays": garage.get("total_bays"),
                },
                "stream_id": sid,
                "media": GLOBAL_ENGINE.media.status(sid),
                "active_camera_id": str(GLOBAL_ENGINE.cfg.get("active_camera_id") or ""),
                "active_roi_cameras": list(GLOBAL_ENGINE.active_roi_cameras.keys()),
                "latest_camera_event": GLOBAL_ENGINE.latest_camera_event,
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))

        elif parsed.path in ("/api/garage/telemetry", "/api/workplace/telemetry"):
            data = GLOBAL_ENGINE.workplace_telemetry()
            grabber = GLOBAL_ENGINE.grabber
            data.update(
                {
                    "fps": GLOBAL_ENGINE.fps,
                    "ingest_fps": grabber.ingest_fps,
                    "infer_ms": round(float(GLOBAL_ENGINE.infer_ms or 0.0), 1),
                    "protocol": str(
                        GLOBAL_ENGINE.cfg.get("protocol")
                        or protocol_from_source(GLOBAL_ENGINE.cfg.get("source"))
                    ),
                    "main_stream": bool(str(GLOBAL_ENGINE.cfg.get("main_source") or "").strip()),
                    "status": GLOBAL_ENGINE.status_text,
                    "connection": grabber.connection_state
                    if grabber.connection_state != "STANDBY"
                    else GLOBAL_ENGINE.connection_state,
                    "resolution": GLOBAL_ENGINE.stream_resolution,
                    "error": grabber.error or GLOBAL_ENGINE.error_message,
                    "person_count": GLOBAL_ENGINE.person_count,
                }
            )
            self._send_json(data)

        elif parsed.path == "/api/garage/scorecard":
            self._send_json(GLOBAL_ENGINE.garage_scorecard())

        elif parsed.path == "/api/workplace/visits":
            data = GLOBAL_ENGINE.workplace_telemetry().get("visits") or customer_visit_counts(
                GLOBAL_ENGINE.conn
            )
            self._send_json(data)

        elif parsed.path == "/api/telegram/status":
            qs = urllib.parse.parse_qs(parsed.query or "")
            user_id = str((qs.get("user_id") or [""])[0] or "").strip()
            self._send_json(GLOBAL_ENGINE.telegram_link_status(user_id))

        elif parsed.path == "/api/garage/evaluations":
            evals = []
            if GLOBAL_ENGINE.conn is not None:
                try:
                    rows = GLOBAL_ENGINE.conn.execute(
                        """
                        SELECT job_id, vehicle_type, primary_technician, technicians_json,
                               total_wrench_seconds, total_break_seconds, efficiency_pct,
                               performance_grade, performance_score, summary_notes,
                               started_at, completed_at
                        FROM vehicle_job_evaluations
                        ORDER BY id DESC LIMIT 50
                        """
                    ).fetchall()
                    for r in rows:
                        evals.append({
                            "job_id": r[0],
                            "vehicle_type": r[1],
                            "primary_technician": r[2],
                            "technicians": json.loads(r[3]) if r[3] else {},
                            "total_wrench_seconds": r[4],
                            "total_break_seconds": r[5],
                            "efficiency_pct": r[6],
                            "performance_grade": r[7],
                            "performance_score": r[8],
                            "summary_notes": r[9],
                            "started_at": r[10],
                            "completed_at": r[11],
                        })
                except Exception as ex:
                    print(f"[API] Error loading evaluations: {ex}")
            self._send_json({"evaluations": evals})

        elif parsed.path == "/api/garage/templates":
            self._send_json({"templates": KNOWLEDGE_BASE.service_templates})

        elif parsed.path == "/api/discovery/results":
            self._send_json(GLOBAL_ENGINE.discovery.results())

        elif parsed.path == "/api/uploaded-videos":
            init_videos_dir()
            videos = []
            allowed_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts"}
            if VIDEOS_DIR.is_dir():
                for f in sorted(VIDEOS_DIR.iterdir()):
                    if f.is_file() and f.suffix.lower() in allowed_exts:
                        videos.append({
                            "name": f.name,
                            "path": str(f.resolve()),
                            "size": f.stat().st_size,
                        })
            self._send_json({"videos": videos})

        elif parsed.path == "/api/frame.jpeg" or parsed.path.startswith("/api/camera/"):
            cam_id = None
            if parsed.path.startswith("/api/camera/"):
                parts = [p for p in parsed.path.strip("/").split("/") if p]
                if len(parts) >= 3 and parts[2] != "frame.jpeg":
                    cam_id = urllib.parse.unquote(parts[2])
                elif len(parts) >= 2 and parts[1] not in ("camera", "frame.jpeg"):
                    cam_id = urllib.parse.unquote(parts[1])
            if not cam_id and parsed.query:
                q = dict(urllib.parse.parse_qsl(parsed.query))
                cam_id = q.get("camera_id") or q.get("src") or q.get("id")

            frame_bytes, ctype = GLOBAL_ENGINE.get_camera_frame(cam_id)
            if not frame_bytes:
                self.send_response(204)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(frame_bytes)))
            self.send_header("Cache-Control", "no-cache, private, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            self.wfile.write(frame_bytes)

        elif parsed.path == "/api/stream":
            # MJPEG kept for non-WebKit clients. Desktop hub uses /api/frame.jpeg
            # instead — WebKitGTK segfaults on long-lived multipart/x-mixed-replace.
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, private, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.end_headers()

            try:
                self.connection.settimeout(2.0)
            except Exception:
                pass

            last_seq = -1
            stream_camera_id = str(GLOBAL_ENGINE.cfg.get("active_camera_id") or "")
            stream_generation = int(getattr(GLOBAL_ENGINE, "mjpeg_generation", 0) or 0)
            idle_ticks = 0
            try:
                while GLOBAL_ENGINE.running:
                    # If active camera switched or a same-camera reconnect started,
                    # terminate this old stream so the browser frees the connection slot.
                    cur_cam = str(GLOBAL_ENGINE.cfg.get("active_camera_id") or "")
                    cur_gen = int(getattr(GLOBAL_ENGINE, "mjpeg_generation", 0) or 0)
                    if cur_cam != stream_camera_id or cur_gen != stream_generation:
                        break

                    with GLOBAL_ENGINE.lock:
                        cur_seq = GLOBAL_ENGINE.frame_seq
                        frame_bytes = GLOBAL_ENGINE.current_frame_jpeg

                    if cur_seq != last_seq and frame_bytes is not None:
                        last_seq = cur_seq
                        idle_ticks = 0
                        self.wfile.write(encode_mjpeg_part(frame_bytes))
                        self.wfile.flush()
                    else:
                        idle_ticks += 1
                        # If frame is absent for > 8 seconds (400 * 0.02s), terminate to free socket
                        if idle_ticks > 400:
                            break
                    time.sleep(0.02)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, socket.error, OSError):
                pass

        elif parsed.path == "/api/garage/jobs":
            self._send_json(list_vehicle_jobs(GLOBAL_ENGINE.conn))

        elif parsed.path.startswith("/api/garage/jobs/") and parsed.path.endswith("/history"):
            parts = [urllib.parse.unquote(p) for p in parsed.path.split("/") if p]
            job_id = parts[3] if len(parts) >= 4 else ""
            res = get_vehicle_job_history(GLOBAL_ENGINE.conn, job_id)
            if res:
                self._send_json(res)
            else:
                self._send_json({"error": "Job not found"}, 404)

        elif parsed.path == "/api/ai-audits":
            limit = 20
            bay_id = None
            if parsed.query:
                q = dict(urllib.parse.parse_qsl(parsed.query))
                try:
                    limit = int(q.get("limit", 20))
                except Exception:
                    limit = 20
                bay_id = q.get("bay_id")
            audits = []
            try:
                conn = connect(DATA_DIR / "events.db", check_same_thread=False)
                try:
                    audits = get_recent_ai_audits(conn, bay_id=bay_id, limit=limit)
                finally:
                    conn.close()
            except Exception as e:
                print(f"[API /api/ai-audits error] {e}", flush=True)
            self._send_json({"audits": audits})

        elif parsed.path.startswith("/proofs/ai_audits/"):
            rel_name = parsed.path[len("/proofs/ai_audits/"):].lstrip("/")
            safe_name = Path(rel_name).name
            proof_file = DATA_DIR / "proofs" / "ai_audits" / safe_name
            if proof_file.exists() and proof_file.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(proof_file.read_bytes())
            else:
                self.send_response(404)
                self.end_headers()

        elif parsed.path == "/api/complaints":
            limit = 50
            if parsed.query:
                q = dict(urllib.parse.parse_qsl(parsed.query))
                try:
                    limit = int(q.get("limit", 50))
                except Exception:
                    limit = 50
            complaints = []
            try:
                conn = connect(DATA_DIR / "events.db", check_same_thread=False)
                try:
                    complaints = get_recent_complaints(conn, limit=limit)
                    for c in complaints:
                        if c.get("audio_path"):
                            c["audio_url"] = f"/api/complaints/audio/{Path(c['audio_path']).name}"
                        if c.get("screenshot_path"):
                            c["screenshot_url"] = f"/api/complaints/stills/{Path(c['screenshot_path']).name}"
                finally:
                    conn.close()
            except Exception as e:
                print(f"[API /api/complaints error] {e}", flush=True)
            self._send_json({"complaints": complaints})

        elif parsed.path.startswith("/api/complaints/audio/"):
            rel_name = parsed.path[len("/api/complaints/audio/"):].lstrip("/")
            safe_name = Path(rel_name).name
            audio_file = Path(__file__).parent / "proofs" / "complaints" / "audio" / safe_name
            if audio_file.exists() and audio_file.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(audio_file.read_bytes())
            else:
                self.send_response(404)
                self.end_headers()

        elif parsed.path.startswith("/api/complaints/stills/"):
            rel_name = parsed.path[len("/api/complaints/stills/"):].lstrip("/")
            safe_name = Path(rel_name).name
            still_file = Path(__file__).parent / "proofs" / "complaints" / "stills" / safe_name
            if still_file.exists() and still_file.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(still_file.read_bytes())
            else:
                self.send_response(404)
                self.end_headers()

        elif self._handle_identities_get(
            [urllib.parse.unquote(p) for p in parsed.path.split("/") if p]
        ):
            return

        else:
            self.send_response(404)
            self.end_headers()


    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        parts = [urllib.parse.unquote(p) for p in parsed.path.split("/") if p]
        length = int(self.headers.get("Content-Length", 0) or 0)
        max_upload_size = 1024 * 1024 * 1024 if parsed.path == "/api/upload-video" else 40 * 1024 * 1024
        if length > max_upload_size:
            self._send_json({"error": "Upload is too large."}, 413)
            return
        body = self.rfile.read(length) if length > 0 else b""
        content_type = self.headers.get("Content-Type") or ""
        files: list[tuple[str, bytes]] = []
        payload: dict[str, Any] = {}
        raw_json: Any = {}
        if "multipart/form-data" in content_type:
            files = self._parse_multipart_files(content_type, body)
        else:
            try:
                raw_json = json.loads(body.decode("utf-8") or "{}")
            except Exception:
                raw_json = {}
            payload = raw_json if isinstance(raw_json, dict) else {}

        if self._handle_identities_write(parts, "POST", payload, files):
            return

        if parsed.path == "/api/garage/jobs/complete":
            job_id = str(payload.get("job_id") or "").strip()
            ok = complete_vehicle_job(GLOBAL_ENGINE.conn, job_id)
            self._send_json({"ok": ok, "job_id": job_id})
            return

        if parsed.path == "/api/ai-audit/trigger":
            bay_id = str(payload.get("bay_id") or "bay_1").strip()
            technician_id = str(payload.get("technician_id") or "manual_trigger").strip()
            with GLOBAL_ENGINE.lock:
                frame = GLOBAL_ENGINE.current_frame_bgr
            if frame is not None and GLOBAL_ENGINE.ai_auditor is not None:
                GLOBAL_ENGINE.ai_auditor.force_audit(
                    bay_id=bay_id,
                    technician_id=technician_id,
                    frame=frame,
                )
                self._send_json({"success": True, "message": f"Audit triggered for {bay_id}"})
            else:
                self._send_json({"success": False, "message": "Frame not available or AI auditor not active"}, 400)
            return

        if parsed.path == "/api/complaints/test-trigger":
            khmer_text = str(payload.get("khmer_text") or "ខ្ញុំមិនពេញចិត្តនឹងការជួសជុលឡាននេះទេ សំឡេងហ្វ្រាំងនៅតែលាន់ដដែល").strip()
            customer_id = str(payload.get("customer_id") or "Customer").strip()
            camera_id = str(payload.get("camera_id") or "cam-1").strip()
            
            if GLOBAL_ENGINE.complaint_service is not None:
                english_text = GLOBAL_ENGINE.complaint_service.translator.translate_km_to_en(khmer_text)
                analysis = GLOBAL_ENGINE.complaint_service.auditor.analyze(khmer_text, english_text)
                
                now = datetime.now()
                stamp_str = now.strftime("%Y-%m-%d_%H%M%S")
                uid = f"CMP-{int(time.time()*1000)%1000000:06d}"
                audio_filename = f"complaint_{stamp_str}_{uid}.wav"
                audio_path = Path(__file__).parent / "proofs" / "complaints" / "audio" / audio_filename
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write sample audio WAV
                sample_rate = 16000
                fake_audio = np.zeros(int(sample_rate * 2.0), dtype=np.float32)
                import soundfile as sf
                sf.write(str(audio_path), fake_audio, sample_rate, subtype="PCM_16")
                
                screenshot_filename = f"complaint_{stamp_str}_{uid}.jpg"
                screenshot_dest = Path(__file__).parent / "proofs" / "complaints" / "stills" / screenshot_filename
                screenshot_dest.parent.mkdir(parents=True, exist_ok=True)
                
                with GLOBAL_ENGINE.lock:
                    frame = GLOBAL_ENGINE.current_frame_bgr
                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                annotated = frame.copy()
                h, w = annotated.shape[:2]
                banner_text = f"CUSTOMER COMPLAINT [{uid}] - {analysis.category.upper()} ({analysis.severity.upper()})"
                time_text = now.strftime("%Y-%m-%d %H:%M:%S")
                cv2.rectangle(annotated, (0, 0), (w, 54), (0, 0, 180), -1)
                cv2.putText(annotated, banner_text, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated, time_text, (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.imwrite(str(screenshot_dest), annotated)
                screenshot_path = str(screenshot_dest)
                
                from complaint_service import ComplaintRecord
                record = ComplaintRecord(
                    complaint_id=uid,
                    customer_id=customer_id,
                    camera_id=camera_id,
                    audio_source="test_trigger",
                    timestamp=now.isoformat(),
                    audio_path=str(audio_path),
                    screenshot_path=screenshot_path,
                    khmer_transcript=khmer_text,
                    english_transcript=english_text,
                    is_complaint=analysis.is_complaint,
                    category=analysis.category,
                    severity=analysis.severity,
                    summary=analysis.summary,
                    telegram_sent=False,
                )
                
                conn = connect(DATA_DIR / "events.db", check_same_thread=False)
                try:
                    insert_customer_complaint(
                        conn=conn,
                        complaint_id=record.complaint_id,
                        audio_path=record.audio_path,
                        screenshot_path=record.screenshot_path,
                        khmer_transcript=record.khmer_transcript,
                        english_transcript=record.english_transcript,
                        category=record.category,
                        severity=record.severity,
                        summary=record.summary,
                        customer_id=record.customer_id,
                        camera_id=record.camera_id,
                        audio_source=record.audio_source,
                        timestamp=record.timestamp,
                        is_complaint=record.is_complaint,
                        telegram_sent=False,
                    )
                finally:
                    conn.close()
                    
                if record.is_complaint and GLOBAL_ENGINE.bot.enabled:
                    tg_ok = GLOBAL_ENGINE.bot.send_complaint_alert(
                        complaint=record.as_dict(),
                        audio_path=audio_path,
                        photo_path=screenshot_dest,
                    )
                    record.telegram_sent = tg_ok
                    conn = connect(DATA_DIR / "events.db", check_same_thread=False)
                    try:
                        update_complaint_telegram_status(conn, record.complaint_id, sent=tg_ok)
                    finally:
                        conn.close()
                        
                self._send_json({"ok": True, "complaint": record.as_dict()})
                return
            else:
                self._send_json({"ok": False, "error": "Complaint service not initialized"}, 500)
                return


        if parsed.path in ("/api/connect-stream", "/api/save"):
            result = GLOBAL_ENGINE.connect_camera(payload)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

        elif parsed.path == "/api/disconnect":
            GLOBAL_ENGINE.disconnect()
            self._send_json({"success": True, "status": "STANDBY"})
            return

        elif parsed.path == "/api/discovery/scan":
            self._send_json(GLOBAL_ENGINE.discovery.start_scan())

        elif parsed.path in ("/api/settings", "/api/config"):
            result = GLOBAL_ENGINE.apply_hub_settings(payload)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

        elif parsed.path == "/api/cameras":
            result = GLOBAL_ENGINE.save_camera(payload)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

        elif parsed.path == "/api/cameras/delete":
            result = GLOBAL_ENGINE.delete_camera(payload.get("id"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

        elif parsed.path in ("/api/cameras/toggle-port", "/api/cameras/toggle"):
            result = GLOBAL_ENGINE.toggle_camera_port(payload.get("id"), payload.get("enabled"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

        elif parsed.path in ("/api/cameras/toggle-ml", "/api/cameras/ml-toggle"):
            cid = payload.get("id") or payload.get("camera_id")
            result = GLOBAL_ENGINE.toggle_camera_ml(cid, payload.get("ml_enabled"))
            self._send_json(result)

        elif parsed.path in ("/api/cameras/acknowledge-event", "/api/cameras/ack-event"):
            GLOBAL_ENGINE.acknowledge_event(payload.get("event_id"))
            self._send_json({"success": True})

        elif parsed.path == "/api/pipeline":
            graph = raw_json if isinstance(raw_json, dict) and "nodes" in raw_json else payload.get("graph") or payload
            result = GLOBAL_ENGINE.apply_pipeline(graph)
            self._send_json(result, status=200 if result.get("ok") else 400)

        elif parsed.path == "/api/orient":
            result = GLOBAL_ENGINE.set_orient(
                payload.get("rotate"),
                payload.get("flip"),
                payload.get("bays"),
                payload.get("roi"),
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

        elif parsed.path == "/api/roi":
            result = GLOBAL_ENGINE.set_roi(payload.get("roi"))
            self._send_json(result)

        elif parsed.path in ("/api/garage/bays", "/api/workplace/zones"):
            bays_payload = raw_json if isinstance(raw_json, list) else (
                payload.get("bays") if "bays" in payload else payload
            )
            result = GLOBAL_ENGINE.set_bays(bays_payload)
            self._send_json(result)

        elif parsed.path == "/api/cameras/import-bays":
            src_id = payload.get("source_camera_id") or payload.get("source_id")
            tgt_id = payload.get("target_camera_id") or payload.get("target_id")
            result = GLOBAL_ENGINE.import_bays(src_id, tgt_id)
            self._send_json(result)

        elif parsed.path == "/api/upload-video":
            init_videos_dir()
            filename = ""
            file_content = b""

            params = urllib.parse.parse_qs(parsed.query)
            if "filename" in params:
                filename = params["filename"][0]

            if files:
                filename = filename or files[0][0]
                file_content = files[0][1]
            elif "multipart/form-data" in content_type and b"filename=" in body:
                match = re.search(rb'filename="([^"]+)"', body)
                if match:
                    filename = match.group(1).decode("utf-8", errors="replace")
                parts = body.split(b"\r\n\r\n", 1)
                if len(parts) == 2:
                    raw_data = parts[1]
                    boundary_end = raw_data.rfind(b"\r\n--")
                    file_content = raw_data[:boundary_end] if boundary_end != -1 else raw_data
                else:
                    file_content = body
            else:
                file_content = body

            if not filename:
                filename = f"upload_{int(time.time())}.mp4"

            safe_name = Path(filename).name
            dest = VIDEOS_DIR / safe_name
            dest.write_bytes(file_content)

            self._send_json({
                "success": True,
                "filename": safe_name,
                "path": str(dest.resolve()),
                "size": len(file_content),
            })

        elif parsed.path == "/api/garage/wifi":
            devices = raw_json if isinstance(raw_json, list) else payload.get("wifi_devices")
            if devices is None:
                devices = payload.get("devices")
            if devices is None and payload.get("name"):
                existing = list(GLOBAL_ENGINE.cfg.get("wifi_devices") or [])
                existing.append(payload)
                devices = existing
            result = GLOBAL_ENGINE.set_wifi_devices(devices)
            self._send_json(result)

        elif parsed.path == "/api/test-telegram":
            token = str(GLOBAL_ENGINE.cfg.get("telegram_bot_token") or "").strip()
            chat = str(
                payload.get("chat_id")
                or payload.get("telegram_chat_id")
                or GLOBAL_ENGINE.cfg.get("telegram_chat_id")
                or ""
            ).strip()
            venue = str(payload.get("venue") or payload.get("garage_name") or "Demo Garage").strip()

            if not token:
                res = {"success": False, "error": "TELEGRAM_BOT_TOKEN is not configured on the engine."}
            elif not chat:
                res = {"success": False, "error": "Connect via /start first, or paste a chat id."}
            else:
                GLOBAL_ENGINE.apply_hub_settings({
                    "telegram_chat_id": chat,
                })
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                proofs_dir = DATA_DIR / "proofs"
                proofs_dir.mkdir(parents=True, exist_ok=True)
                proof_path = proofs_dir / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

                frame_bytes = None
                with GLOBAL_ENGINE.lock:
                    if GLOBAL_ENGINE.current_frame_jpeg is not None:
                        frame_bytes = GLOBAL_ENGINE.current_frame_jpeg

                if frame_bytes is None and GLOBAL_ENGINE.is_streaming:
                    GLOBAL_ENGINE.new_frame_event.wait(timeout=2.0)
                    with GLOBAL_ENGINE.lock:
                        frame_bytes = GLOBAL_ENGINE.current_frame_jpeg

                if frame_bytes is None:
                    res = {"success": False, "error": "Could not capture camera frame. Please connect to a camera stream first."}
                else:
                    with proof_path.open("wb") as pf:
                        pf.write(frame_bytes)

                    status_str = GLOBAL_ENGINE.status_text or (
                        "STAFF IN ROI" if GLOBAL_ENGINE.is_occupied else "EMPTY"
                    )
                    identified = ", ".join(GLOBAL_ENGINE.identities) or "none"
                    caption = (
                        f"🔧 *Inbound Garage Snapshot*\n\n"
                        f"🏢 *Shop:* {venue}\n"
                        f"⏰ *Timestamp:* {now_str}\n"
                        f"🛠️ *Floor Status:* {status_str}\n"
                        f"👥 *Detections:* {GLOBAL_ENGINE.person_count} Person(s)\n"
                        f"🪪 *Identified:* {identified}\n"
                    )

                    url = f"https://api.telegram.org/bot{token}/sendPhoto"
                    try:
                        with proof_path.open("rb") as handle:
                            r = requests.post(
                                url,
                                data={"chat_id": chat, "caption": caption, "parse_mode": "Markdown"},
                                files={"photo": handle},
                                timeout=30,
                            )
                        data = r.json()
                        if r.status_code == 200 and data.get("ok"):
                            res = {
                                "success": True,
                                "message": f"Snapshot captured at {now_str} and sent to Telegram!",
                            }
                        else:
                            res = {"success": False, "error": data.get("description", f"HTTP {r.status_code}")}
                    except Exception as e:
                        res = {"success": False, "error": str(e)}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

        elif parsed.path == "/api/telegram/begin-link":
            user_id = str(payload.get("user_id") or "").strip()
            display_name = str(payload.get("display_name") or "").strip()
            open_client = bool(payload.get("open_client"))
            self._send_json(
                GLOBAL_ENGINE.begin_telegram_link(
                    user_id, display_name, open_client=open_client
                )
            )

        elif parsed.path == "/api/telegram/open":
            app_link = str(payload.get("app_link") or "").strip()
            deep_link = str(payload.get("deep_link") or "").strip()
            opened = open_host_url(app_link) if app_link else False
            if not opened and deep_link:
                opened = open_host_url(deep_link)
            self._send_json({"ok": opened, "opened_on_host": opened})

        elif parsed.path == "/api/telegram/status":
            user_id = str(payload.get("user_id") or "").strip()
            self._send_json(GLOBAL_ENGINE.telegram_link_status(user_id))

        elif parsed.path == "/api/auth-session":
            session_file = DATA_DIR / "session.json"
            try:
                sess_data = payload.get("session") if (isinstance(payload, dict) and "session" in payload and isinstance(payload.get("session"), dict)) else payload
                with session_file.open("w", encoding="utf-8") as f:
                    json.dump(sess_data, f, indent=2)
                self._send_json({"ok": True})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)

        elif parsed.path == "/api/profile":
            updates = {}
            if "display_name" in payload:
                updates["display_name"] = str(payload.get("display_name") or "").strip()
            if "venue_name" in payload:
                updates["venue_name"] = str(payload.get("venue_name") or "").strip()
                updates["venue"] = updates["venue_name"]
                updates["garage_name"] = updates["venue_name"]
            
            if "venue" in updates:
                GLOBAL_ENGINE.apply_hub_settings({"venue": updates["venue"]})
            
            session_file = DATA_DIR / "session.json"
            if session_file.exists():
                try:
                    with session_file.open("r", encoding="utf-8") as f:
                        sdata = json.load(f)
                    prof = sdata.get("profile") or {}
                    prof.update(updates)
                    sdata["profile"] = prof
                    with session_file.open("w", encoding="utf-8") as f:
                        json.dump(sdata, f, indent=2)
                except Exception:
                    pass
            self._send_json({"ok": True, "profile": updates})

        elif parsed.path == "/api/profile/avatar":
            import base64
            image_data = payload.get("avatar") or payload.get("image") or payload.get("data")
            if not image_data:
                self._send_json({"ok": False, "error": "No image data provided"}, status=400)
            else:
                try:
                    if "," in image_data:
                        image_data = image_data.split(",", 1)[1]
                    raw_bytes = base64.b64decode(image_data)
                    avatars_dir = DATA_DIR / "avatars"
                    avatars_dir.mkdir(parents=True, exist_ok=True)
                    avatar_file = avatars_dir / "avatar.jpg"
                    with avatar_file.open("wb") as f:
                        f.write(raw_bytes)
                    self._send_json({"ok": True, "url": f"/api/profile/avatar?t={int(time.time())}"})
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=500)

        else:
            self.send_response(404)
            self.end_headers()

    def do_DELETE(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/auth-session":
            session_file = DATA_DIR / "session.json"
            if session_file.exists():
                try:
                    session_file.unlink()
                except Exception:
                    pass
            self._send_json({"ok": True})
            return
        elif parsed.path == "/api/profile/avatar":
            avatars_dir = DATA_DIR / "avatars"
            for ext in (".jpg", ".jpeg", ".png"):
                f = avatars_dir / f"avatar{ext}"
                if f.exists():
                    try:
                        f.unlink()
                    except Exception:
                        pass
            self._send_json({"ok": True})
            return
        parts = [urllib.parse.unquote(p) for p in parsed.path.split("/") if p]
        if self._handle_identities_write(parts, "DELETE", {}, []):
            return
        self.send_response(404)
        self.end_headers()


def start_unified_server(port: int = 8765, open_browser: bool = True) -> None:
    engine = init_global_engine()
    actual_port = find_free_port(port)
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("127.0.0.1", actual_port), DashboardRequestHandler)
    server.daemon_threads = True

    engine.start()

    url = f"http://127.0.0.1:{actual_port}"
    print("Inbound Garage", flush=True)
    print(f"Dashboard: {url}", flush=True)
    print(
        f"[INBOUND_SERVER_READY] port={actual_port} build={INBOUND_BUILD_ID}",
        flush=True,
    )

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    shutting_down = {"done": False}

    def shutdown(*_args: object) -> None:
        if shutting_down["done"]:
            return
        shutting_down["done"] = True
        print("\nStopping Inbound Garage...", flush=True)
        try:
            engine.stop()
        except Exception:
            pass
        try:
            server.shutdown()
        except Exception:
            pass

    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown)
    if sys.platform == "win32" and hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, shutdown)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        shutdown()
    finally:
        shutdown()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inbound Garage camera hub")
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Preferred HTTP port (falls back to a free port if busy).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a system browser.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        args = parse_args()
        start_unified_server(port=args.port, open_browser=not args.no_browser)
    except Exception as _boot_err:
        fatal_boot(_boot_err)
