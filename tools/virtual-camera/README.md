# Inbound Virtual Camera Streamer

A dedicated containerized RTSP IP camera simulator for **Inbound Surveillance**. It allows uploading recorded video files (MP4, MKV, MOV, AVI) or ingesting online videos (YouTube or direct links) and streams them continuously in real-time over RTSP in an infinite loop, behaving identically to physical auto-shop IP cameras.

---

## Features

- **Real-Time Looping RTSP Server**: Uses MediaMTX & FFmpeg with `-re -stream_loop -1` to stream video at normal 1.0x camera speed with zero frame lag or accumulation.
- **Local File Upload**: Drag-and-drop or select any video file via the Web UI.
- **Online Video Downloader**: Enter any YouTube link or direct video URL to download and convert on-the-fly via `yt-dlp`.
- **Multi-Camera Channel Support**: Stream to multiple channels simultaneously (e.g. `garage`, `bay1`, `bay2`).
- **Zero Host Port Conflict**: Mapped to RTSP port `8556` by default to avoid conflicting with Inbound Surveillance's embedded `go2rtc` process on `8554`.
- **Built-in Quick Demo Generator**: 1-click synthetic garage video generator for instant offline testing.
- **Persistent Video Library**: Videos are saved in `./videos` on the host, persisting across container reboots.

---

## Quick Start

### 1. Start the Container

```bash
cd tools/virtual-camera
./run.sh
# Or using Docker Compose directly:
# docker compose up --build -d
```

### 2. Open the Web Dashboard

Open your browser to:
👉 **[http://localhost:8090](http://localhost:8090)**

From the dashboard you can:
- Drag and drop any garage or car video.
- Or paste a YouTube video URL and click **Download & Add to Library**.
- Or click **Quick Demo Clip** for an instant synthetic test video.
- Click **Copy** to copy the RTSP stream URL (`rtsp://127.0.0.1:8556/garage`).

---

## Connecting to Inbound Surveillance

1. Start Inbound Surveillance:
   ```bash
   npm run engine
   # Or open the desktop app
   ```
2. Open the Hub interface (`http://localhost:8765` or `edge/hub.html`).
3. In the **Camera Config** section:
   - **Camera location**: `Virtual Garage Camera`
   - **Protocol**: `RTSP`
   - **Stream URL**: `rtsp://127.0.0.1:8556/garage`
4. Click **Connect Stream**.
5. The machine learning pipeline (YOLO pose estimation, vehicle detection, and lift bay analytics) will immediately process the virtual camera feed as if it were a real physical camera!

> **Note on Multi-Camera Concurrency**: Secondary camera streams (virtual RTSP, IP phone cameras, video files) stream smoothly in the multi-camera grid without freezing, even when another camera is actively tracking technicians in its ROI.

---

## Testing / Verification

To verify that the RTSP feed is active and readable using OpenCV:

```bash
# Using the edge virtual environment:
edge/.venv/bin/python tools/virtual-camera/verify_stream.py --url rtsp://127.0.0.1:8556/garage
```

---

## Stopping the Container

```bash
cd tools/virtual-camera
./stop.sh
# Or: docker compose down
```
