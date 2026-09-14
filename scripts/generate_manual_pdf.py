#!/usr/bin/env python3
"""Generate a concise, 3-page Customer Quick-Start & Operations Manual for Inbound Surveillance.

Covers:
- Page 1: System Overview, Navigation, Bay Overlays & ROI Setup
- Page 2: Connecting Streams & Uploading Video Files (RTSP, Phone, USB, Upload MP4)
- Page 3: Telegram Bot Setup (Step-by-Step), Troubleshooting Matrix & Quick Commands
"""

import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

# --- Theme Color Tokens ---
COLOR_PRIMARY = colors.HexColor("#008F39")       # Surveillance Green (Print contrast)
COLOR_SECONDARY = colors.HexColor("#0B833A")     # Stealth Green
COLOR_DARK = colors.HexColor("#141414")          # Deep Charcoal/Black
COLOR_TEXT = colors.HexColor("#222222")          # Body text
COLOR_MUTED = colors.HexColor("#555555")         # Muted gray text
COLOR_BG_LIGHT = colors.HexColor("#F8F9FA")      # Table alternate row
COLOR_BORDER = colors.HexColor("#D8DCE0")        # Subtle border
COLOR_CALLOUT_BG = colors.HexColor("#F0FDF4")    # Light green tint for tips/notes
COLOR_WARN_BG = colors.HexColor("#FFF7ED")       # Light orange tint for warnings
COLOR_WARN_BORDER = colors.HexColor("#EA580C")   # Orange border


class NumberedCanvas(canvas.Canvas):
    """Canvas that adds clean running headers and footers with total page count."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        page_width, page_height = letter
        margin = 36  # 0.5 inch

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(COLOR_MUTED)

        # Header
        header_text = "INBOUND SURVEILLANCE — CUSTOMER QUICK-START MANUAL"
        self.drawString(margin, page_height - 25, header_text)
        self.setStrokeColor(COLOR_BORDER)
        self.setLineWidth(0.5)
        self.line(margin, page_height - 28, page_width - margin, page_height - 28)

        # Footer
        self.line(margin, 28, page_width - margin, 28)
        self.drawString(margin, 18, "Confidential — For Authorized Venue Operators & Customers")
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(page_width - margin, 18, page_str)
        self.restoreState()


def build_callout(text: str, title: str = "NOTE", kind: str = "note", styles=None):
    """Build a compact styled callout box."""
    bg_color = COLOR_CALLOUT_BG if kind != "warning" else COLOR_WARN_BG
    border_color = COLOR_PRIMARY if kind != "warning" else COLOR_WARN_BORDER

    t_style = styles["CalloutTitle"]
    b_style = styles["CalloutBody"]

    title_p = Paragraph(f"<b>{title.upper()}</b>", t_style)
    body_p = Paragraph(text, b_style)

    box_data = [[title_p], [body_p]]
    box_table = Table(box_data, colWidths=[540])
    box_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_color),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('LINELEFT', (0, 0), (0, -1), 3.5, border_color),
        ('TOPPADDING', (0, 0), (-1, 0), 4),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 1),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 4),
    ]))
    return box_table


def generate_pdf(output_path: str):
    pdf_path = Path(output_path).resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Tight, clean 0.5 inch margins (36pt) for maximum content density and readability
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    sample_styles = getSampleStyleSheet()

    styles = {
        "MainTitle": ParagraphStyle(
            "MainTitle",
            parent=sample_styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=COLOR_DARK,
            alignment=0,
            spaceAfter=2,
        ),
        "MainSubtitle": ParagraphStyle(
            "MainSubtitle",
            parent=sample_styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=COLOR_MUTED,
            spaceAfter=8,
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=sample_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14.5,
            textColor=COLOR_PRIMARY,
            spaceBefore=5,
            spaceAfter=2.5,
            keepWithNext=True,
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=sample_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=COLOR_DARK,
            spaceBefore=6,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=sample_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.5,
            textColor=COLOR_TEXT,
            spaceAfter=4,
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=sample_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.5,
            textColor=COLOR_TEXT,
            leftIndent=12,
            firstLineIndent=-8,
            spaceAfter=2,
        ),
        "CalloutTitle": ParagraphStyle(
            "CalloutTitle",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=COLOR_DARK,
        ),
        "CalloutBody": ParagraphStyle(
            "CalloutBody",
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=COLOR_TEXT,
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            fontName="Helvetica",
            fontSize=8,
            leading=10.5,
            textColor=COLOR_TEXT,
        ),
        "TableCellBold": ParagraphStyle(
            "TableCellBold",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10.5,
            textColor=COLOR_DARK,
        ),
    }

    story = []

    repo_dir = Path(__file__).resolve().parent.parent
    logo_path = repo_dir / "dist" / "email" / "logo-black.png"
    if not logo_path.exists():
        logo_path = repo_dir / "dist" / "inb_surveillance.png"

    # =========================================================================
    # PAGE 1: SYSTEM OVERVIEW, DAILY USAGE & BAY SETUP
    # =========================================================================
    # Header Banner with Logo + Title
    header_data = [
        [
            Image(str(logo_path), width=48, height=50) if logo_path.exists() else Paragraph("", styles["Body"]),
            [
                Paragraph("<b>INBOUND SURVEILLANCE</b> — Customer Quick-Start Guide", styles["MainTitle"]),
                Paragraph("AI Bay Occupancy Monitoring · Technician Wrench-Time Analytics · Face ID · Telegram Dispatch", styles["MainSubtitle"]),
            ]
        ]
    ]
    header_table = Table(header_data, colWidths=[55, 485])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=COLOR_PRIMARY, spaceBefore=4, spaceAfter=8))

    story.append(Paragraph("1. System Overview & Core Capabilities", styles["H1"]))
    story.append(Paragraph(
        "<b>Inbound Surveillance</b> transforms your security cameras and webcams into proactive AI edge agents. "
        "Unlike passive CCTV that only records video, Inbound actively tracks vehicle bay occupancy, calculates technician wrench-time, "
        "records automated Face ID attendance, and sends instant Telegram photo proofs during safety breaches.",
        styles["Body"]
    ))
    story.append(Paragraph("• <b>Local Edge Processing:</b> All video analysis runs 100% locally on your venue station. Video feeds never upload to external public clouds.", styles["Bullet"]))
    story.append(Paragraph("• <b>Two Interfaces Available:</b> Use the <b>Web Operations Console</b> in any browser (<code>http://localhost:5173/dashboard.html</code>) or launch the standalone <b>Native Desktop App</b>.", styles["Bullet"]))

    story.append(Spacer(1, 4))
    story.append(Paragraph("2. Navigating the Live View Console", styles["H1"]))
    story.append(Paragraph(
        "The <b>Live View</b> is your day-to-day command center. It shows real-time camera video streams, active telemetry, and intelligent bay overlays:",
        styles["Body"]
    ))
    story.append(Paragraph("• <b>Multi-Camera Grid:</b> Displays all active cameras. Click any tile to inspect it in full screen.", styles["Bullet"]))
    story.append(Paragraph("• <b>Top Telemetry Banner:</b> Displays <i>Ingest FPS</i> (camera delivery speed), <i>Infer Latency</i> (AI processing speed, typically 20-45ms), and resolution.", styles["Bullet"]))
    story.append(Paragraph("• <b>Audio Chimes (SOUND ON / MUTE):</b> Click the sound pill in the header to enable pleasant audio chimes whenever a car enters a service bay.", styles["Bullet"]))
    story.append(Paragraph("• <b>Shop Operating Hours:</b> Automatically marks the venue as <b>SHOP OPEN</b> during business hours (e.g. 08:00–18:00) and switches to <b>AFTER HOURS</b> mode (which triggers high-priority trespass alerts).", styles["Bullet"]))

    story.append(Spacer(1, 4))
    story.append(Paragraph("3. Understanding Service Bay Overlays & Colors", styles["H1"]))
    bay_table_data = [
        [Paragraph("Border Color", styles["TableHeader"]), Paragraph("Status Name", styles["TableHeader"]), Paragraph("Detection Meaning", styles["TableHeader"]), Paragraph("Operational Action", styles["TableHeader"])],
        [
            Paragraph("<b>Vibrant Green</b>", styles["TableCellBold"]),
            Paragraph("<b>Active Labor</b>", styles["TableCellBold"]),
            Paragraph("Vehicle present and technician actively detected working (wrench pose).", styles["TableCell"]),
            Paragraph("Accumulates productive wrench-time on technician scorecards.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Warm Amber</b>", styles["TableCellBold"]),
            Paragraph("<b>Bay Idle</b>", styles["TableCellBold"]),
            Paragraph("Vehicle present, but no active tool movement detected for >120 seconds.", styles["TableCell"]),
            Paragraph("Increments idle timer; alerts manager to parts waiting or downtime.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Subtle Gray</b>", styles["TableCellBold"]),
            Paragraph("<b>Available / Empty</b>", styles["TableCellBold"]),
            Paragraph("No vehicle or technician detected inside the service bay boundary.", styles["TableCell"]),
            Paragraph("Flags bay as open for immediate customer intake.", styles["TableCell"])
        ],
    ]
    bay_table = Table(bay_table_data, colWidths=[75, 95, 230, 140])
    bay_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_DARK),
        ('GRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_BG_LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(bay_table)

    story.append(Spacer(1, 4))
    story.append(Paragraph("4. Setting Up Service Bays & ROIs (Drag-and-Drop)", styles["H1"]))
    story.append(Paragraph(
        "To position service bays over your vehicle lifts or alignment racks:<br/>"
        "1. Open <b>Live View</b> and locate your target camera feed.<br/>"
        "2. Click the <b>Settings Cog</b> or right-click any existing bay to open the <b>Bay Context Menu</b>.<br/>"
        "3. Select <b>Edit ROI</b> to activate interactive bounding handles. Drag the corners to frame your vehicle bay.<br/>"
        "4. Assign a descriptive name (e.g. <i>Lift Bay 1</i>, <i>Tire Station</i>) and click <b>Save</b>.",
        styles["Body"]
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: CONNECTING STREAMS & UPLOADING VIDEO FILES
    # =========================================================================
    story.append(Paragraph("5. Connecting Cameras & Video Sources", styles["H1"]))
    story.append(Paragraph(
        "Inbound Surveillance supports real-time IP cameras, mobile phone cameras, USB webcams, and pre-recorded video files. "
        "Click <b>Add Camera</b> in the sidebar or setup modal to connect a video source.",
        styles["Body"]
    ))

    source_table_data = [
        [Paragraph("Source Type", styles["TableHeader"]), Paragraph("Protocol", styles["TableHeader"]), Paragraph("Connection URL Example / Steps", styles["TableHeader"])],
        [
            Paragraph("<b>IP Camera (RTSP)</b><br/>Hikvision, Dahua, Axis", styles["TableCellBold"]),
            Paragraph("<code>rtsp</code>", styles["TableCell"]),
            Paragraph("<code>rtsp://username:password@192.168.1.50:554/h264Preview_01_main</code><br/><i>Use camera's local IP address, RTSP port 554, and ONVIF/RTSP credentials.</i>", styles["TableCell"])
        ],
        [
            Paragraph("<b>Pre-Recorded Video</b><br/>Uploaded MP4/AVI file", styles["TableCellBold"]),
            Paragraph("<code>video</code>", styles["TableCell"]),
            Paragraph("Select <b>Local Video File</b>, click <b>Upload Video</b>, and choose your file.<br/><i>The engine loops the video locally for offline testing, demos, and forensics.</i>", styles["TableCell"])
        ],
        [
            Paragraph("<b>Android Phone</b><br/>Using <i>IP Webcam</i> app", styles["TableCellBold"]),
            Paragraph("<code>phone</code>", styles["TableCell"]),
            Paragraph("<code>http://192.168.1.85:8080/video</code><br/><i>Ensure phone is on the same Wi-Fi. Start server in the app and enter the video URL.</i>", styles["TableCell"])
        ],
        [
            Paragraph("<b>USB Webcam</b><br/>Logitech / PC Camera", styles["TableCellBold"]),
            Paragraph("<code>webcam</code>", styles["TableCell"]),
            Paragraph("Enter device index <code>0</code> (or <code>1</code> for secondary webcam, <code>/dev/video0</code> on Linux).<br/><i>Great for front-desk enrollment and test setups.</i>", styles["TableCell"])
        ],
        [
            Paragraph("<b>One-Click Auto Scan</b><br/>ONVIF & Tapo Discovery", styles["TableCellBold"]),
            Paragraph("<code>onvif</code> / <code>tapo</code>", styles["TableCell"]),
            Paragraph("Click <b>Scan Network</b>. The system automatically detects compatible ONVIF and Tapo cameras on your subnet. Click <b>Connect</b> to apply.", styles["TableCell"])
        ],
    ]
    source_table = Table(source_table_data, colWidths=[110, 65, 365])
    source_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_DARK),
        ('GRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_BG_LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(source_table)

    story.append(Spacer(1, 4))
    story.append(Paragraph("How to Upload & Analyze Pre-Recorded Video Files", styles["H2"]))
    story.append(Paragraph(
        "If you want to test the system or analyze pre-recorded incident footage without a live camera:<br/>"
        "1. Click <b>Add Camera</b> in the sidebar or open <b>Settings → Source</b>.<br/>"
        "2. Set <b>Protocol</b> to <b>Local Video File (MP4/AVI)</b>.<br/>"
        "3. Click <b>Upload Video</b> and select your file (e.g., <code>garage_shift_morning.mp4</code>).<br/>"
        "4. The system automatically uploads the file to the local edge directory (<code>edge/videos/</code>) and loops playback seamlessly through the AI pose and bay tracking pipeline.<br/>"
        "5. Enter a camera label (e.g., <i>Shift Review - Bay 1</i>) and click <b>Connect & Save</b>.",
        styles["Body"]
    ))

    story.append(Spacer(1, 4))
    story.append(Paragraph("Dual-Stream Best Practice (Pro Tip for Commercial Garages)", styles["H2"]))
    story.append(Paragraph(
        "To get 25+ FPS AI detection without overloading your CPU, configure dual streams in <code>config.yaml</code>:<br/>"
        "• <b>Substream (<code>source:</code>):</b> Set to 640x360 or 1280x720 resolution @ 15 FPS. Used for continuous, high-speed AI inference.<br/>"
        "• <b>Mainstream (<code>main_source:</code>):</b> Set to 1080p or 4K resolution. Queried only when an alert fires to capture crisp photographic evidence.",
        styles["Body"]
    ))

    story.append(Spacer(1, 4))
    story.append(build_callout(
        "<b>Phone Camera Tip:</b> When using an Android phone with the free <i>IP Webcam</i> app, make sure to append <code>/video</code> to the end of the URL (e.g. <code>http://192.168.1.85:8080/video</code>). Avoid using the web interface root URL.",
        title="PHONE STREAM SETUP",
        kind="note",
        styles=styles
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: TELEGRAM BOT SETUP, TROUBLESHOOTING & SUPPORT
    # =========================================================================
    story.append(Paragraph("6. Connecting Your Telegram Alerts (1-Click Pairing)", styles["H1"]))
    story.append(Paragraph(
        "Inbound Surveillance comes with an official pre-configured Telegram Bot. You do <b>not</b> need to create a bot or paste API tokens. "
        "The software automatically links your Telegram account and Chat ID directly to your signed-in profile with one click:",
        styles["Body"]
    ))

    tg_setup_steps = [
        [Paragraph("Step", styles["TableHeader"]), Paragraph("Action", styles["TableHeader"]), Paragraph("Exact Instructions", styles["TableHeader"])],
        [
            Paragraph("<b>1</b>", styles["TableCellBold"]),
            Paragraph("<b>Sign In</b>", styles["TableCellBold"]),
            Paragraph("Log in to your venue account on the Inbound Surveillance web console or desktop application so the bot knows which account to pair.", styles["TableCell"])
        ],
        [
            Paragraph("<b>2</b>", styles["TableCellBold"]),
            Paragraph("<b>Click Connect</b>", styles["TableCellBold"]),
            Paragraph("Navigate to <b>Settings → Integrations</b> (or open the <b>Telegram Panel</b> on the right sidebar) and click the green <b>Connect Telegram</b> button.", styles["TableCell"])
        ],
        [
            Paragraph("<b>3</b>", styles["TableCellBold"]),
            Paragraph("<b>Tap Start</b>", styles["TableCellBold"]),
            Paragraph("The official Inbound Surveillance bot opens automatically in your Telegram app. Tap <b>Start</b> (or send <code>/start</code>). Your unique pairing token is verified instantly.", styles["TableCell"])
        ],
        [
            Paragraph("<b>4</b>", styles["TableCellBold"]),
            Paragraph("<b>Auto-Linked</b>", styles["TableCellBold"]),
            Paragraph("Your Chat ID is automatically saved to your software account. The badge switches to <b>Linked</b>, activating instant incident photo dispatches and daily scorecards.", styles["TableCell"])
        ],
        [
            Paragraph("<b>5</b>", styles["TableCellBold"]),
            Paragraph("<b>Send Test</b>", styles["TableCellBold"]),
            Paragraph("Click <b>Capture & Send Test</b> (or <b>Submit Ticket</b>) in the panel to receive an immediate live photo proof on your smartphone and confirm delivery.", styles["TableCell"])
        ],
    ]
    tg_table = Table(tg_setup_steps, colWidths=[30, 90, 420])
    tg_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_DARK),
        ('GRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_BG_LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(tg_table)

    story.append(Spacer(1, 3))
    story.append(Paragraph("7. Quick Troubleshooting Matrix", styles["H1"]))
    trouble_data = [
        [Paragraph("Issue / Symptom", styles["TableHeader"]), Paragraph("Probable Cause", styles["TableHeader"]), Paragraph("Quick Fix (1 Minute)", styles["TableHeader"])],
        [
            Paragraph("<b>Black screen or 'Connecting…'</b>", styles["TableCellBold"]),
            Paragraph("Camera IP changed or wrong password.", styles["TableCell"]),
            Paragraph("Verify camera IP in router DHCP table. Test stream URL in VLC player.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Telegram alerts not sending</b>", styles["TableCellBold"]),
            Paragraph("Telegram account not yet linked.", styles["TableCell"]),
            Paragraph("Open <b>Settings → Integrations</b>, click <b>Connect Telegram</b>, and tap <b>Start</b> in the Telegram app.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Frame rate drops below 10 FPS</b>", styles["TableCellBold"]),
            Paragraph("Stream resolution too high for CPU.", styles["TableCell"]),
            Paragraph("Switch camera to 720p or 360p substream. In <code>config.yaml</code> set <code>detect_fps: 15</code>.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Technician marked 'Idle' under car</b>", styles["TableCellBold"]),
            Paragraph("Worker occluded longer than grace limit.", styles["TableCell"]),
            Paragraph("Increase <code>under_car_grace_seconds: 45</code> in <code>config.yaml</code> to maintain working state.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Face ID identifies staff as 'Customer'</b>", styles["TableCellBold"]),
            Paragraph("Enrolled photo has poor lighting/angle.", styles["TableCell"]),
            Paragraph("Upload 2 fresh, front-facing photos in good lighting via <b>Identity → Enroll Staff</b>.", styles["TableCell"])
        ],
    ]
    trouble_table = Table(trouble_data, colWidths=[125, 140, 275])
    trouble_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_DARK),
        ('GRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_BG_LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(trouble_table)

    story.append(Spacer(1, 3))
    story.append(Paragraph("8. Desktop Application Downloads (Windows & Linux Ubuntu)", styles["H1"]))
    download_data = [
        [
            Paragraph("Operating System", styles["TableHeader"]),
            Paragraph("Distribution Format", styles["TableHeader"]),
            Paragraph("Download & Setup Instructions", styles["TableHeader"])
        ],
        [
            Paragraph("<b>Windows 10 / 11</b><br/>64-bit Desktop Client<br/><font color=\"#EA580C\"><b>[Experimental]</b></font>", styles["TableCellBold"]),
            Paragraph("GitHub Actions Artifact<br/>(ZIP Package)", styles["TableCell"]),
            Paragraph(
                "Download official Windows client installer build from GitHub:<br/>"
                '<font color="#008F39"><u><a href="https://github.com/sothunly-alt/Ibound-survillence-V2/actions/runs/34489388811/artifacts/10157584518"><b>Click Here: Download Windows Build (GitHub Artifacts)</b></a></u></font><br/>'
                "<font size=\"6.5\" color=\"#555555\">URL: https://github.com/sothunly-alt/Ibound-survillence-V2/actions/runs/34489388811/artifacts/10157584518</font><br/>"
                "<i>Extract ZIP package and launch installer to complete setup.</i><br/>"
                "<font color=\"#C2410C\"><b>[Windows Platform Disclaimer]:</b> Inbound Surveillance was developed natively on Linux. "
                "The Linux Ubuntu version is thoroughly tested and runs error-free. "
                "Because our primary environment is Linux, the Windows build has received minimal testing and users may experience crashing issues or failure to run. "
                "For production stability, Linux Ubuntu is strongly recommended.</font>",
                styles["TableCell"]
            )
        ],
        [
            Paragraph("<b>Linux Ubuntu</b><br/>20.04 / 22.04 / 24.04<br/><font color=\"#008F39\"><b>[Verified & Stable]</b></font>", styles["TableCellBold"]),
            Paragraph("Debian Package<br/>(<code>.deb</code> installer)", styles["TableCell"]),
            Paragraph(
                "For Ubuntu Linux, the native <code>.deb</code> installer file is provided directly in the application folder.<br/>"
                "Install via terminal: <code>sudo dpkg -i Inbound_Surveillance_amd64.deb</code> (or double-click the <code>.deb</code> file to install via Ubuntu Software). Fully tested and error-free.",
                styles["TableCell"]
            )
        ],
    ]
    download_table = Table(download_data, colWidths=[115, 110, 315])
    download_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_DARK),
        ('GRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_BG_LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(download_table)

    story.append(Spacer(1, 3))
    creds_content = (
        "Once installed, launch the application and log in using the pre-configured credentials:<br/>"
        "• <b>Email:</b> <code>testaccount@gmail.com</code> &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp; "
        "• <b>Password:</b> <code>abcd1234</code><br/>"
        "<i>These credentials grant full access to live camera feeds, service bay overlays, and analytics.</i>"
    )
    story.append(build_callout(creds_content, title="PRE-CONFIGURED LOGIN CREDENTIALS", kind="note", styles=styles))

    story.append(Spacer(1, 3))
    story.append(Paragraph("9. Customer Support", styles["H1"]))
    support_content = (
        "For system onboarding, camera configuration assistance, or technical inquiries, contact the Inbound Crew:<br/>"
        '• <b>Support Email:</b> <font color="#008F39"><u><a href="mailto:inboundcrew82@gmail.com"><b>inboundcrew82@gmail.com</b></a></u></font><br/>'
        "• <b>Service Hours:</b> Dedicated assistance and priority support for commercial garage operators."
    )
    story.append(build_callout(support_content, title="INBOUND CREW SUPPORT", kind="note", styles=styles))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Concise PDF Generated successfully: {pdf_path}")
    print(f"File Size: {pdf_path.stat().st_size:,} bytes")

    # Sync to root and artifacts
    import shutil
    root_pdf = repo_dir / "Inbound_Surveillance_User_Manual.pdf"
    if pdf_path != root_pdf:
        shutil.copy2(pdf_path, root_pdf)
        print(f"[SYNC] Copied to root: {root_pdf}")

    artifact_dir = Path("/home/george/.gemini/antigravity/brain/ab42098d-c830-4ae4-845c-8313b5465973")
    if artifact_dir.exists():
        shutil.copy2(pdf_path, artifact_dir / "Inbound_Surveillance_User_Manual.pdf")
        print(f"[SYNC] Copied to artifact directory")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/Inbound_Surveillance_User_Manual.pdf"
    generate_pdf(out)

