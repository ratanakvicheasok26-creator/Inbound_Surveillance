# Task: Fix Phone (IP Webcam) Camera Display, Auto-Rotation & Sizing Mismatch

## Problem Summary
When connecting a mobile device (such as an Android phone running IP Webcam) to the Inbound Garage Surveillance architecture (`edge/hub.html` / `http://127.0.0.1:8765/`):
1. **Initial Tiny Box Display:** The live camera stream displays as a tiny, shrunken box in the middle of the screen instead of properly filling the video container.
2. **0° Angle Button Side Effect:** Clicking the `0°` orientation button expands the video box to full width, but the stream displays sideways/horizontal instead of upright.
3. **Repeated Reset Glitch:** Switching cameras, changing protocols, or reloading the dashboard forcibly resets the orientation back to `auto`, repeating the bug every time.
4. **Coordinate & Display Desync:** Bounding boxes, bay overlays, and video rendering become desynchronized between go2rtc WebRTC and Python OpenCV.

---

## Root Causes Identified

### 1. Hardcoded Heuristic in Backend `suggest_rotate()` (`edge/launcher.py` & `edge/main.py`)
In `edge/launcher.py`:
```python
def suggest_rotate(source: Any, frame) -> int:
    if protocol_from_source(source) != "phone":
        return 0
    if frame is None:
        return 90
    h, w = frame.shape[:2]
    if w >= h:
        return 90
    return 0
```
- IP Webcam streams video in standard landscape sensor coordinates (e.g. 1920×1080 or 1280×720).
- When `rotate` is set to `auto` (the default), `suggest_rotate()` unconditionally returns `90` (90° CW rotation) whenever `w >= h`.
- Python rotates the frame in memory using `cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)`, creating a 1080×1920 (portrait) frame for YOLO and reporting `resolution: "1080x1920"` via `/api/telemetry`.

### 2. Dimension Mismatch Between WebRTC and Server Telemetry (`edge/hub.html`)
- **Go2rtc WebRTC (`#live-webrtc`):** Delivers the raw, unrotated 16:9 stream (1920×1080) directly from the camera to the browser.
- **Python Telemetry (`/api/telemetry`):** Reports the rotated 9:16 portrait dimensions (`1080x1920`) because OpenCV rotated the frame in memory.
- In `hub.html`:
  ```javascript
  const nw = streamW || (usingRtc ? video.videoWidth : ...);
  const nh = streamH || (usingRtc ? video.videoHeight : ...);
  ```
  `streamW` (`1080`) and `streamH` (`1920`) from telemetry override `video.videoWidth` (`1920`) and `video.videoHeight` (`1080`).
- `layoutVideoStage()` sizes `#video-stage` to portrait (`ar = 1080 / 1920 = 0.5625`).
  With `maxH = 400px`, stage width is set to `400 * 0.5625 = 225px`.
- Inside this 225px wide portrait container, the `<video>` element (playing 16:9 horizontal WebRTC video with `object-fit: contain`) is forced to shrink to **225px × 126px** — a tiny postage stamp in the middle of a widescreen monitor.

### 3. Why Clicking "0°" Expands the View but Leaves it Sideways
- When the user clicks `0°` (`setRotate(0)`):
  - Backend receives `rotate: 0`. OpenCV stops rotating the frame.
  - Telemetry resolution updates to `1920x1080` (`ar = 1.777`).
  - `layoutVideoStage()` resizes `#video-stage` to 16:9 (`w = vw = 1000px, h = 562px`), expanding to full width.
  - However, because the phone camera's physical orientation was vertical but raw stream is landscape, the picture is rendered sideways on its side (horizontal instead of vertical).

### 4. Frontend Forcibly Resets User Rotation on Phone Protocol (`edge/hub.html`)
In `edge/hub.html` lines 1465–1468 (`fillCameraForm`) and lines 1245–1249 (`onProtocolChange`):
```javascript
if (proto === 'phone' && (rot === undefined || rot === '' || rot === 'auto' || String(rot) === '0')) {
  rotateDeg = 'auto';
  flipMode = 'none';
}
```
- Whenever a phone camera is selected or the protocol is set to `phone`, any user-configured `0` degree or manual orientation is wiped out and overwritten to `auto`.
- This causes the glitch to re-occur every time the camera reconnects or is edited.

### 5. WebRTC vs OpenCV Orientation Desync
- Go2rtc streams raw frames without CSS rotation.
- When rotation is applied in OpenCV (e.g. 90° or 270°), the WebRTC video element in the frontend is not rotated via CSS, causing:
  - Video displayed in orientation A.
  - Bounding boxes and ROI bay polygons displayed in orientation B.

---

## Step-by-Step Implementation Instructions

### Step 1: Fix Forced Auto-Rotate Overwrite in `edge/hub.html`
In [`edge/hub.html`](file:///home/george/Documents/Inbound-Surveillance/edge/hub.html), ensure that explicit user rotation settings (including `0`) are respected and not overwritten when loading or switching to a phone camera.

#### In `fillCameraForm(cam)` (around lines 1463–1478):
Replace:
```javascript
      const proto = cam.protocol || document.getElementById('protocol-input').value;
      const rot = cam.rotate;
      if (proto === 'phone' && (rot === undefined || rot === '' || rot === 'auto' || String(rot) === '0')) {
        rotateDeg = 'auto';
        flipMode = 'none';
      } else if (rot !== undefined && rot !== 'auto') {
        const n = Number(rot);
        if ([0, 90, 180, 270].indexOf(n) !== -1) rotateDeg = n;
      } else {
        rotateDeg = 'auto';
      }
```
With:
```javascript
      const rot = cam.rotate;
      if (rot !== undefined && rot !== '' && rot !== 'auto') {
        const n = Number(rot);
        if ([0, 90, 180, 270].indexOf(n) !== -1) {
          rotateDeg = n;
        } else {
          rotateDeg = 'auto';
        }
      } else {
        rotateDeg = 'auto';
      }
```

#### In `onProtocolChange()` (around lines 1241–1250):
Remove the block that forcibly overrides `rotateDeg` to `'auto'` when selecting `phone`:
```diff
       } else if (tab === 'phone') {
         if (sourceInput.value === '0' || sourceInput.value.startsWith('rtsp') || sourceInput.value.startsWith('tapo') || sourceInput.value.startsWith('onvif')) {
           sourceInput.value = 'http://192.168.1.50:8080/video';
         }
-        if (rotateDeg === 0) {
-          rotateDeg = 'auto';
-          flipMode = 'none';
-          paintOrient();
-        }
       }
```

---

### Step 2: Make `suggest_rotate()` Sensible in `edge/launcher.py` and `edge/main.py`
In [`edge/launcher.py`](file:///home/george/Documents/Inbound-Surveillance/edge/launcher.py), allow `suggest_rotate` to default to `0` (natural stream orientation) instead of forcing `90` on all landscape HTTP phone streams. `auto` should respect the natural stream resolution unless the user explicitly chooses `90° (CW)`, `270° (CCW)`, or `180°`.

In `edge/launcher.py` (around lines 309–325):
```python
def suggest_rotate(source: Any, frame) -> int:
    """Return default rotation for auto-orient.
    Default to 0 (native camera orientation) so streams are not unexpectedly rotated sideways.
    Users can select CW (90°), CCW (270°), or 180° for mounted phone orientations.
    """
    return 0
```

In `edge/main.py` (around lines 87–95):
```python
def default_rotate(source: int | str) -> int:
    """Default to native orientation (0) across all camera sources."""
    return 0
```

---

### Step 3: Synchronize Video Player Aspect Ratio in `edge/hub.html`
In [`edge/hub.html`](file:///home/george/Documents/Inbound-Surveillance/edge/hub.html), inside `layoutVideoStage()` (around lines 2480–2508):
When WebRTC is active (`usingRtc === true`), use the actual WebRTC video element's natural aspect ratio (`video.videoWidth / video.videoHeight`), or when using MJPEG/telemetry, use `streamW / streamH`.

```javascript
    function layoutVideoStage() {
      const img = document.getElementById('live-camera-feed');
      const video = document.getElementById('live-webrtc');
      const viewport = document.getElementById('video-viewport');
      const stage = document.getElementById('video-stage');
      const placeholder = document.getElementById('stream-placeholder');
      const workspace = document.getElementById('workspace');
      if (!viewport || !stage || !isStreamActive) return;

      stage.style.display = 'block';
      if (placeholder) placeholder.style.display = 'none';

      const usingRtc = video && video.style.display !== 'none' && video.videoWidth > 1 && video.videoHeight > 1;
      const nw = usingRtc ? video.videoWidth : (streamW || (img && img.naturalWidth) || 640);
      const nh = usingRtc ? video.videoHeight : (streamH || (img && img.naturalHeight) || 480);

      if (workspace && !workspace.classList.contains('is-native-feed')) {
        workspace.classList.add('is-native-feed');
        requestAnimationFrame(layoutVideoStage);
        return;
      }
      const vw = viewport.clientWidth || 640;
      if (vw < 2) return;
      const maxH = visibleFeedMaxHeight(workspace);
      const ar = (nw > 0 && nh > 0) ? (nw / nh) : (16 / 9);
      let w = vw;
      let h = Math.max(1, w / ar);
      if (h > maxH) {
        h = maxH;
        w = Math.max(1, h * ar);
      }
      if (w > vw) {
        w = vw;
        h = Math.max(1, w / ar);
      }
      viewport.style.height = Math.round(h) + 'px';
      stage.style.width = Math.round(w) + 'px';
      stage.style.height = Math.round(h) + 'px';
      layoutRoiOverlay();
    }
```

---

### Step 4: Add WebRTC Stream Fallback & MJPEG Stream Support for Rotated Phones
When a user explicitly selects a non-zero rotation (e.g. `90° CW` or `270° CCW` for a vertical phone):
- In `startLivePlayer(streamId, media)` in `edge/hub.html`:
  If `rotateDeg !== 0 && rotateDeg !== 'auto'`, prefer the annotated/rotated MJPEG stream (`/api/stream`) so that video orientation, YOLO detections, and ROI bay overlays remain 100% pixel-aligned.
- When `rotateDeg === 0 || rotateDeg === 'auto'`, use WebRTC via go2rtc for ultra-low latency streaming.

In `edge/hub.html` (around lines 993–1005):
```javascript
    function startLivePlayer(streamId, media) {
      const img = document.getElementById('live-camera-feed');
      // If the stream is rotated (non-zero), use server-side oriented MJPEG to keep YOLO boxes in sync
      if (rotateDeg !== 0 && rotateDeg !== 'auto') {
        showAnnotatedMjpeg();
        return;
      }
      const ready = media && media.ready && (media.ws || streamId);
      if (!ready) {
        showAnnotatedMjpeg();
        return;
      }
      const wsUrl = media.ws || ('ws://127.0.0.1:1984/api/ws?src=' + encodeURIComponent(streamId));
      const started = startGo2rtcPlayer(wsUrl, media);
      if (!started) {
        if (media.mjpeg) showGo2rtcMjpeg(media.mjpeg);
        else showAnnotatedMjpeg();
      }
      ...
    }
```

---

## Verification & Testing

1. **Verify JavaScript Syntax:**
   ```bash
   node -e '
   const fs = require("fs");
   const vm = require("node:vm");
   const html = fs.readFileSync("edge/hub.html", "utf8");
   const scriptStart = html.indexOf("<script>") + 8;
   const scriptEnd = html.lastIndexOf("</script>");
   const code = html.substring(scriptStart, scriptEnd);
   new vm.Script(code, { filename: "edge/hub.html" });
   console.log("hub.html syntax OK");
   '
   ```

2. **Run Edge Suite Unit Tests:**
   ```bash
   edge/.venv/bin/python edge/test_capture.py && \
   edge/.venv/bin/python edge/test_discovery.py && \
   edge/.venv/bin/python edge/test_garage.py && \
   edge/.venv/bin/python edge/test_go2rtc.py && \
   edge/.venv/bin/python edge/test_onvif.py
   ```

3. **End-to-End Functional Test:**
   - Launch server: `edge/.venv/bin/python edge/launcher.py --port 8765`
   - Open `http://127.0.0.1:8765/`.
   - Add/Select a phone IP Webcam stream (`http://PHONE_IP:8080/video`).
   - Verify stream opens at full width without shrinking into a tiny box.
   - Verify selecting `0°`, `90° (CW)`, `CCW`, or `Auto` preserves user selection and updates orientation and bounding boxes cleanly.
