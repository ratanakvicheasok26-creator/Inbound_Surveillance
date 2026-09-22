"""FastAPI application for Virtual Camera Streamer."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from generate_test_clip import generate_garage_sample
from stream_manager import StreamManager

logger = logging.getLogger("virtual_camera_app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = Path(os.environ.get("VIDEOS_DIR", BASE_DIR / "videos")).resolve()
MEDIAMTX_CONFIG = str(BASE_DIR / "mediamtx.yml")
MEDIAMTX_PATH = os.environ.get("MEDIAMTX_PATH", "mediamtx")

RTSP_INTERNAL_PORT = int(os.environ.get("RTSP_INTERNAL_PORT", 8554))
EXTERNAL_RTSP_PORT = int(os.environ.get("EXTERNAL_RTSP_PORT", 8556))
EXTERNAL_WEBRTC_PORT = int(os.environ.get("EXTERNAL_WEBRTC_PORT", 8889))
EXTERNAL_HOST = os.environ.get("EXTERNAL_HOST", "localhost")

manager = StreamManager(
    mediamtx_path=MEDIAMTX_PATH,
    mediamtx_config=MEDIAMTX_CONFIG,
    rtsp_internal_port=RTSP_INTERNAL_PORT,
    external_rtsp_port=EXTERNAL_RTSP_PORT,
    external_webrtc_port=EXTERNAL_WEBRTC_PORT,
    external_host=EXTERNAL_HOST,
)

# In-memory download task tracking
downloads: Dict[str, dict[str, Any]] = {}
downloads_lock = threading.Lock()


def run_yt_dlp_download(task_id: str, url: str, target_dir: Path) -> None:
    """Download video using yt-dlp to target directory in standard MP4 format."""
    with downloads_lock:
        downloads[task_id] = {
            "id": task_id,
            "url": url,
            "status": "downloading",
            "progress": 0,
            "filename": None,
            "error": None,
            "started_at": time.time(),
        }

    try:
        import yt_dlp

        downloaded_file: Optional[str] = None

        def progress_hook(d: dict[str, Any]):
            if d.get("status") == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                downloaded = d.get("downloaded_bytes") or 0
                pct = int((downloaded / total * 100)) if total > 0 else 50
                with downloads_lock:
                    if task_id in downloads:
                        downloads[task_id]["progress"] = min(99, pct)
            elif d.get("status") == "finished":
                nonlocal downloaded_file
                downloaded_file = d.get("filename")
                with downloads_lock:
                    if task_id in downloads:
                        downloads[task_id]["progress"] = 100

        ydl_opts = {
            "outtmpl": str(target_dir / "%(title).50s-%(id)s.%(ext)s"),
            # Merge to MP4 with H.264 video for maximum compatibility
            "format": "bestvideo[vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
            "progress_hooks": [progress_hook],
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "downloaded_video")
            final_filename = ydl.prepare_filename(info)
            # When merged, extension is mp4
            p = Path(final_filename)
            if not p.exists() and p.with_suffix(".mp4").exists():
                p = p.with_suffix(".mp4")

        with downloads_lock:
            downloads[task_id]["status"] = "completed"
            downloads[task_id]["progress"] = 100
            downloads[task_id]["filename"] = p.name

        logger.info(f"Successfully downloaded online video: {p.name}")

    except Exception as ex:
        logger.error(f"Failed to download online video from {url}: {ex}")
        with downloads_lock:
            downloads[task_id]["status"] = "error"
            downloads[task_id]["error"] = str(ex)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    manager.start_mediamtx()

    # If no videos exist, generate sample clip
    sample_path = VIDEOS_DIR / "sample_garage_demo.mp4"
    if not sample_path.exists() and len(list(VIDEOS_DIR.glob("*.mp4"))) == 0:
        logger.info("Generating initial sample garage video for instant demo...")
        try:
            generate_garage_sample(str(sample_path), duration_sec=15)
        except Exception as ex:
            logger.warning(f"Could not generate initial sample video: {ex}")

    # Auto-start default 'garage' stream if a video is present
    existing_videos = sorted(VIDEOS_DIR.glob("*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True)
    if existing_videos:
        default_video = existing_videos[0]
        try:
            manager.start_stream("garage", str(default_video))
            logger.info(f"Auto-started default stream 'garage' with {default_video.name}")
        except Exception as ex:
            logger.warning(f"Could not auto-start default stream: {ex}")

    yield

    # Shutdown
    manager.shutdown()


app = FastAPI(
    title="Virtual Camera Streamer",
    description="Stream uploaded or online videos as continuous RTSP IP cameras for Inbound Surveillance ML",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartStreamRequest(BaseModel):
    video: str
    channel: str = "garage"


class StopStreamRequest(BaseModel):
    channel: str = "garage"


class DownloadUrlRequest(BaseModel):
    url: str


@app.get("/api/videos")
def list_videos():
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    videos = []
    for ext in ("*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm"):
        for f in VIDEOS_DIR.glob(ext):
            stat = f.stat()
            videos.append({
                "name": f.name,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
            })
    videos.sort(key=lambda x: x["modified_at"], reverse=True)
    return {"videos": videos}


@app.delete("/api/videos/{filename}")
def delete_video(filename: str):
    file_path = VIDEOS_DIR / filename
    if not file_path.resolve().is_relative_to(VIDEOS_DIR) or not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    # Stop any stream currently using this video
    for stream in manager.list_streams():
        if stream["video_file"] == filename:
            manager.stop_stream(stream["channel"])

    try:
        file_path.unlink()
        return {"success": True, "message": f"Deleted {filename}"}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    clean_name = Path(file.filename or "uploaded_video.mp4").name
    # Ensure safe filename
    clean_name = clean_name.replace(" ", "_").replace("..", "")
    target = VIDEOS_DIR / clean_name

    logger.info(f"Receiving file upload: {clean_name}")
    try:
        async with aiofiles.open(target, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                await out_file.write(chunk)
    except Exception as ex:
        logger.error(f"Upload failed: {ex}")
        raise HTTPException(status_code=500, detail=f"Failed to write file: {ex}")

    return {
        "success": True,
        "filename": clean_name,
        "size_mb": round(target.stat().st_size / (1024 * 1024), 2),
    }


@app.post("/api/download-url")
def download_online_video(req: DownloadUrlRequest, background_tasks: BackgroundTasks):
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL. Must begin with http:// or https://")

    task_id = f"dl-{int(time.time() * 1000)}"
    background_tasks.add_task(run_yt_dlp_download, task_id, url, VIDEOS_DIR)
    return {"success": True, "task_id": task_id, "message": "Download started in background"}


@app.get("/api/downloads")
def list_downloads():
    with downloads_lock:
        return {"downloads": list(downloads.values())}


@app.post("/api/generate-sample")
def generate_sample_clip():
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = VIDEOS_DIR / f"sample_garage_demo_{int(time.time())}.mp4"
    try:
        created = generate_garage_sample(str(sample_path), duration_sec=15)
        p = Path(created)
        return {
            "success": True,
            "filename": p.name,
            "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
        }
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


@app.get("/api/streams")
def get_streams():
    return {
        "streams": manager.list_streams(),
        "external_host": EXTERNAL_HOST,
        "rtsp_port": EXTERNAL_RTSP_PORT,
    }


@app.post("/api/streams/start")
def start_stream(req: StartStreamRequest):
    video_path = VIDEOS_DIR / req.video
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file '{req.video}' not found in videos library")

    channel = req.channel.strip().lower() or "garage"
    stream = manager.start_stream(channel, str(video_path))
    if stream.status == "error":
        raise HTTPException(status_code=500, detail=stream.error or "Failed to start stream")

    return {
        "success": True,
        "stream": stream.to_dict(EXTERNAL_HOST, EXTERNAL_RTSP_PORT, EXTERNAL_WEBRTC_PORT),
    }


@app.post("/api/streams/stop")
def stop_stream(req: StopStreamRequest):
    channel = req.channel.strip().lower() or "garage"
    ok = manager.stop_stream(channel)
    return {"success": ok, "channel": channel}


# Mount static directory for UI
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
