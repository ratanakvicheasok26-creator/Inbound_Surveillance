# Inbound Surveillance — Customer Quick-Start Guide

> **System Version:** 2.4  
> **Platform:** Inbound Surveillance & Garage Operations Console  
> **Document Type:** Concise Customer Operations & Quick-Start Manual  
> **PDF Manual:** [Inbound_Surveillance_User_Manual.pdf](file:///home/george/Documents/Inbound-Surveillance/Inbound_Surveillance_User_Manual.pdf)

---

## 1. System Overview & Core Capabilities

**Inbound Surveillance** transforms your standard security cameras and webcams into proactive AI edge agents. Unlike passive CCTV that only records video, Inbound actively tracks vehicle bay occupancy, calculates technician wrench-time, records automated Face ID attendance, and sends instant Telegram photo proofs during safety breaches.

- **Local Edge Processing:** All video analysis runs 100% locally on your venue station. Video feeds never upload to external public clouds.
- **Two Interfaces Available:** Use the **Web Operations Console** in any browser (`http://localhost:5173/dashboard.html`) or launch the standalone **Native Desktop App**.

---

## 2. Navigating the Live View Console

The **Live View** is your day-to-day command center. It shows real-time camera video streams, active telemetry, and intelligent bay overlays:
- **Multi-Camera Grid:** Displays all active cameras. Click any tile to inspect it in full screen.
- **Top Telemetry Banner:** Displays *Ingest FPS* (camera delivery speed), *Infer Latency* (AI processing speed, typically 20–45ms), and resolution.
- **Audio Chimes (SOUND ON / MUTE):** Click the sound pill in the header to enable audio chimes whenever a car enters a service bay.
- **Shop Operating Hours:** Automatically marks the venue as **SHOP OPEN** during business hours (e.g. 08:00–18:00) and switches to **AFTER HOURS** mode (which triggers high-priority trespass alerts).

---

## 3. Understanding Service Bay Overlays & Colors

| Border Color | Status Name | Detection Meaning | Operational Action |
| :--- | :--- | :--- | :--- |
| **Vibrant Green** | **Active Labor** | Vehicle present and technician actively detected working (wrench pose). | Accumulates productive wrench-time on technician scorecards. |
| **Warm Amber** | **Bay Idle** | Vehicle present, but no active tool movement detected for >120 seconds. | Increments idle timer; alerts manager to parts waiting or downtime. |
| **Subtle Gray** | **Available / Empty** | No vehicle or technician detected inside the service bay boundary. | Flags bay as open for immediate customer intake. |

---

## 4. Setting Up Service Bays & ROIs (Drag-and-Drop)

To position service bays over your vehicle lifts or alignment racks:
1. Open **Live View** and locate your target camera feed.
2. Click the **Settings Cog** or right-click any existing bay to open the **Bay Context Menu**.
3. Select **Edit ROI** to activate interactive bounding handles. Drag the corners to frame your vehicle bay.
4. Assign a descriptive name (e.g. *Lift Bay 1*, *Tire Station*) and click **Save**.

---

## 5. Connecting Cameras & Video Sources

Inbound Surveillance supports real-time IP cameras, mobile phone cameras, USB webcams, and pre-recorded video files. Click **Add Camera** in the sidebar or setup modal to connect a video source:

| Source Type | Protocol | Connection URL Example / Steps |
| :--- | :--- | :--- |
| **IP Camera (RTSP)**<br/>Hikvision, Dahua, Axis | `rtsp` | `rtsp://username:password@192.168.1.50:554/h264Preview_01_main`<br/>*Use camera's local IP address, RTSP port 554, and ONVIF/RTSP credentials.* |
| **Pre-Recorded Video**<br/>Uploaded MP4/AVI file | `video` | Select **Local Video File**, click **Upload Video**, and choose your file.<br/>*The engine loops the video locally for offline testing, demos, and forensics.* |
| **Android Phone**<br/>Using *IP Webcam* app | `phone` | `http://192.168.1.85:8080/video`<br/>*Ensure phone is on the same Wi-Fi. Start server in the app and enter the video URL.* |
| **USB Webcam**<br/>Logitech / PC Camera | `webcam` | Enter device index `0` (or `1` for secondary webcam, `/dev/video0` on Linux).<br/>*Great for front-desk enrollment and test setups.* |
| **One-Click Auto Scan**<br/>ONVIF & Tapo Discovery | `onvif` / `tapo` | Click **Scan Network**. The system automatically detects compatible ONVIF and Tapo cameras on your subnet. Click **Connect** to apply. |

### How to Upload & Analyze Pre-Recorded Video Files
If you want to test the system or analyze pre-recorded incident footage without a live camera:
1. Click **Add Camera** in the sidebar or open **Settings → Source**.
2. Set **Protocol** to **Local Video File (MP4/AVI)**.
3. Click **Upload Video** and select your file (e.g., `garage_shift_morning.mp4`).
4. The system automatically uploads the file to the local edge directory (`edge/videos/`) and loops playback seamlessly through the AI pose and bay tracking pipeline.
5. Enter a camera label (e.g., *Shift Review - Bay 1*) and click **Connect & Save**.

### Dual-Stream Best Practice (Pro Tip for Commercial Garages)
To get 25+ FPS AI detection without overloading your CPU, configure dual streams in `config.yaml`:
- **Substream (`source:`):** Set to 640x360 or 1280x720 resolution @ 15 FPS. Used for continuous, high-speed AI inference.
- **Mainstream (`main_source:`):** Set to 1080p or 4K resolution. Queried only when an alert fires to capture crisp photographic evidence.

---

## 6. Connecting Your Telegram Alerts (1-Click Pairing)

Inbound Surveillance comes with an official pre-configured Telegram Bot. You do **not** need to create a bot or paste API tokens. The software automatically links your Telegram account and Chat ID directly to your signed-in profile with one click:

| Step | Action | Exact Instructions |
| :--- | :--- | :--- |
| **1** | **Sign In** | Log in to your venue account on the Inbound Surveillance web console or desktop application so the bot knows which account to pair. |
| **2** | **Click Connect** | Navigate to **Settings → Integrations** (or open the **Telegram Panel** on the right sidebar) and click the green **Connect Telegram** button. |
| **3** | **Tap Start** | The official Inbound Surveillance bot opens automatically in your Telegram app. Tap **Start** (or send `/start`). Your unique pairing token is verified instantly. |
| **4** | **Auto-Linked** | Your Chat ID is automatically saved to your software account. The badge switches to **Linked**, activating instant incident photo dispatches and daily scorecards. |
| **5** | **Send Test** | Click **Capture & Send Test** (or **Submit Ticket**) in the panel to receive an immediate live photo proof on your smartphone and confirm delivery. |

---

## 7. Quick Troubleshooting Matrix

| Issue / Symptom | Probable Cause | Quick Fix (1 Minute) |
| :--- | :--- | :--- |
| **Black screen or 'Connecting…'** | Camera IP changed or wrong password. | Verify camera IP in router DHCP table. Test stream URL in VLC player. |
| **Telegram alerts not sending** | Telegram account not yet linked. | Open **Settings → Integrations**, click **Connect Telegram**, and tap **Start** in the Telegram app. |
| **Frame rate drops below 10 FPS** | Stream resolution too high for CPU. | Switch camera to 720p or 360p substream. In `config.yaml` set `detect_fps: 15`. |
| **Technician marked 'Idle' under car** | Worker occluded longer than grace limit. | Increase `under_car_grace_seconds: 45` in `config.yaml` to maintain working state. |
| **Face ID identifies staff as 'Customer'** | Enrolled photo has poor lighting/angle. | Upload 2 fresh, front-facing photos in good lighting via **Identity → Enroll Staff**. |

---

## 8. Desktop Application Downloads (Windows & Linux Ubuntu)

Inbound Surveillance can be deployed across Windows and Linux workstations:

| Platform | Format | Download & Setup Instructions |
| :--- | :--- | :--- |
| **Windows 10 / 11** (64-bit)<br/>*`[Experimental]`* | GitHub Actions Artifact (`.zip`) | **[Download Windows Build (GitHub Artifacts)](https://github.com/sothunly-alt/Ibound-survillence-V2/actions/runs/34489388811/artifacts/10157584518)**<br/>*Direct Link:* `https://github.com/sothunly-alt/Ibound-survillence-V2/actions/runs/34489388811/artifacts/10157584518`<br/>Extract the downloaded ZIP package and launch the setup executable.<br/>**[Windows Platform Disclaimer]:** Inbound Surveillance was developed natively on Linux. The Linux Ubuntu version is thoroughly tested and runs error-free. Because our primary environment is Linux, the Windows build has received minimal testing and users may experience crashing issues or failure to run. For production stability, Linux Ubuntu is strongly recommended. |
| **Linux Ubuntu** (20.04+)<br/>*`[Verified & Stable]`* | Debian Package (`.deb`) | For Ubuntu Linux, the native `.deb` package is provided directly in the application folder.<br/>Install via terminal: `sudo dpkg -i Inbound_Surveillance_amd64.deb` (or double-click the `.deb` file to install through the Ubuntu Software Center). Fully tested and verified. |

> [!WARNING]
> **Windows Platform Disclaimer & Stability Notice:**  
> Inbound Surveillance is natively built and verified on Linux Ubuntu. The Windows release is an experimental build compiled via CI with minimal hardware testing. Windows users may encounter startup crashes or instability. If you experience issues running on Windows, we strongly recommend deploying on Linux Ubuntu.

> [!TIP]
> **Pre-Configured Login Credentials:**  
> Once installed, launch the application and log in using:  
> - **Email:** `testaccount@gmail.com`  
> - **Password:** `abcd1234`  
> *(Grants full access to live camera feeds, service bay overlays, and analytics.)*

---


## 9. Customer Support

For system onboarding, camera configuration assistance, or technical inquiries, reach out directly to the Inbound Crew:

- **Support Email:** [inboundcrew82@gmail.com](mailto:inboundcrew82@gmail.com)
- **Service Hours:** Dedicated onboarding assistance and priority support for commercial garage operators.


