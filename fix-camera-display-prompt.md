# Task: Fix Camera Not Displaying on Inbound Garage Dashboard

## Problem Summary
When loading the Inbound Garage dashboard (`edge/hub.html` / `http://127.0.0.1:8765/`), the live camera video does not display (shows a black screen, placeholder "No camera connected", or does not stream).

---

## Root Causes Identified

### 1. [Critical] Fatal JavaScript Syntax Error in `edge/hub.html`
Inside the `<script>` tag of [`edge/hub.html`](file:///home/george/Documents/Inbound-Surveillance/edge/hub.html#L1876-L1879), there is an orphaned `catch` block (lines 1876–1879) after `hitBay()`:
```javascript
    function hitBay(nx, ny) {
      for (let i = liveBays.length - 1; i >= 0; i--) {
        const [x, y, w, h] = liveBays[i].roi;
        if (nx >= x && nx <= x + w && ny >= y && ny <= y + h) return liveBays[i];
      }
      return null;
    }
      } catch (err) {
        /* ignore poll glitches */
      }
    }

    function clamp01(v) { return Math.max(0, Math.min(1, v)); }
```
**Impact:** Browser throws `Uncaught SyntaxError: Unexpected token '}'`. The entire script fails to execute. No stream connection functions, initial config loaders, or telemetry loops ever run.

---

### 2. [Architecture] No Auto-Connect on Initial Page Load
- In `edge/launcher.py`, `LiveStreamEngine.__init__` starts with `self.is_streaming = False` and grabber in `STANDBY`.
- In `edge/hub.html`, `loadInitialConfig()` loads the configuration from `/api/config` into the form inputs, but **never triggers `connectAndStreamCamera()`** on page load.
- In HTML, `#video-stage` starts with `style="display: none;"` and `#stream-placeholder` is visible.
- **Impact:** Opening the dashboard leaves the video area on "No camera connected" until a user manually clicks "Connect Stream" (or selects a camera from the list).

---

### 3. [Configuration] Unreachable Camera Source in `edge/config.yaml`
In `edge/config.yaml`:
```yaml
source: http://hello:admin@192.168.9.177:8080/video
active_camera_id: cam-1788163535721
```
- `active_camera_id` is set to an external phone IP (`192.168.9.177:8080`). If the phone app is not currently running or the IP changed, the connection will time out / fail.
- To use the local webcam instead, `source` must be `0` (or the `pc` camera `cam-1788164516880` must be active).

---

### 4. [Rendering] Sizing & Unhide Block in `layoutVideoStage()`
In `edge/hub.html`:
```javascript
const nw = streamW || (usingRtc ? video.videoWidth : (img && img.naturalWidth) || 0) || 0;
const nh = streamH || (usingRtc ? video.videoHeight : (img && img.naturalHeight) || 0) || 0;
if (nw < 2 || nh < 2) return;
```
- When using MJPEG (`<img>`), `naturalWidth`/`naturalHeight` is initially `0` in many browsers until the first multipart image frame is decoded.
- If `streamW` and `streamH` are `0` before telemetry returns, `layoutVideoStage()` returns early and `#video-stage` remains with `display: none`.

---

## Step-by-Step Implementation Instructions

### Step 1: Remove Syntax Error in `edge/hub.html`
Delete lines 1876–1879 in `edge/hub.html`:
```diff
    function hitBay(nx, ny) {
      for (let i = liveBays.length - 1; i >= 0; i--) {
        const [x, y, w, h] = liveBays[i].roi;
        if (nx >= x && nx <= x + w && ny >= y && ny <= y + h) return liveBays[i];
      }
      return null;
    }
-      } catch (err) {
-        /* ignore poll glitches */
-      }
-    }

    function clamp01(v) { return Math.max(0, Math.min(1, v)); }
```

### Step 2: Add Auto-Connect on Dashboard Load in `edge/hub.html`
In `loadInitialConfig()` in `edge/hub.html`, after setting up the active camera, automatically trigger stream connection if an active camera or source is configured:
```javascript
// At the end of loadInitialConfig()
if (activeCameraId || currentConfig.source !== undefined) {
  connectAndStreamCamera();
}
```

### Step 3: Ensure Video Stage Unhides on Stream Start
In `layoutVideoStage()` in `edge/hub.html`, ensure default dimensions fallback (e.g. 640x480 or container aspect ratio) if `nw < 2` so `#video-stage` is unhidden and `#stream-placeholder` is hidden when `isStreamActive` is true:
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

  const usingRtc = video && video.style.display !== 'none' && video.videoWidth > 1;
  const nw = streamW || (usingRtc ? video.videoWidth : (img && img.naturalWidth) || 0) || 640;
  const nh = streamH || (usingRtc ? video.videoHeight : (img && img.naturalHeight) || 0) || 480;

  const vw = viewport.clientWidth || 640;
  const maxH = visibleFeedMaxHeight(workspace);
  const ar = nw / nh;
  let w = vw;
  let h = Math.max(1, w / ar);
  if (h > maxH) {
    h = maxH;
    w = Math.max(1, h * ar);
  }
  viewport.style.height = Math.round(h) + 'px';
  stage.style.width = Math.round(w) + 'px';
  stage.style.height = Math.round(h) + 'px';
  layoutRoiOverlay();
}
```

### Step 4: Verify Default Source Configuration in `edge/config.yaml`
Check `edge/config.yaml`. If testing with a local USB/laptop webcam, ensure `active_camera_id: cam-1788164516880` or `source: 0`. If using a phone camera, ensure the phone IP and port match the IP Webcam app (`http://IP:8080/video`).

---

## Verification
1. Verify JS syntax:
   ```bash
   node -e '
   const fs = require("fs");
   const vm = require("node:vm");
   const html = fs.readFileSync("edge/hub.html", "utf8");
   const code = html.substring(html.indexOf("<script>") + 8, html.lastIndexOf("</script>"));
   new vm.Script(code, { filename: "edge/hub.html" });
   console.log("hub.html syntax OK");
   '
   ```
2. Start the server:
   ```bash
   edge/.venv/bin/python edge/launcher.py --port 8765
   ```
3. Open `http://127.0.0.1:8765/` in the browser:
   - Live stream loads and displays in `#video-stage`.
   - YOLO bounding boxes, bay overlays, and FPS telemetry render smoothly.
