import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable, PageBreak
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#475569"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "INBOUND SURVEILLANCE — SYSTEM ARCHITECTURE & OPERATIONS BRIEF")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)

        # Footer (all pages)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(54, 36, "Confidential — Inbound Surveillance & Edge AI Systems")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_text)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        
        self.restoreState()

def build_pdf(filename="Inbound_Surveillance_System_Overview.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#0F172A")    # Slate 900
    accent_blue = colors.HexColor("#2563EB")      # Blue 600
    text_dark = colors.HexColor("#1E293B")        # Slate 800
    text_muted = colors.HexColor("#64748B")       # Slate 500
    bg_light = colors.HexColor("#F8FAFC")         # Slate 50
    border_color = colors.HexColor("#E2E8F0")     # Slate 200

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=primary_color,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=accent_blue,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=accent_blue,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=text_dark,
        spaceAfter=6
    )

    body_bold = ParagraphStyle(
        'BodyDarkBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13.5,
        textColor=text_dark
    )

    bullet_style = ParagraphStyle(
        'BulletItem',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.2,
        leading=13,
        textColor=text_dark,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E3A8A")
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=text_dark
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=text_dark
    )

    story = []

    # Title Block
    story.append(Paragraph("INBOUND SURVEILLANCE", title_style))
    story.append(Paragraph("Edge-Native AI Computer Vision & Autonomous Operations Platform", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent_blue, spaceBefore=0, spaceAfter=10))

    # Meta banner table
    meta_data = [
        [
            Paragraph("<b>Platform Version:</b> v2.4 (Production Edge Release)", table_cell_style),
            Paragraph("<b>Architecture:</b> Edge AI + Rust/Tauri Desktop + React 19", table_cell_style),
            Paragraph("<b>Document:</b> System Architecture Brief", table_cell_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[170, 200, 134])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('BOX', (0,0), (-1,-1), 0.5, border_color),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary & Core Value Proposition", h1_style))
    exec_summary_text = (
        "<b>Inbound Surveillance</b> is an intelligent, edge-native computer vision and physical operations platform. "
        "Unlike conventional CCTV systems that passively record video for retrospective manual reviews, Inbound actively "
        "transforms standard IP cameras (RTSP), USB webcams, and mobile video feeds into proactive real-time edge agents. "
        "The system continuously monitors commercial venue floor space, evaluates service bay occupancy, measures active "
        "technician wrench-time vs. idle downtime, clocks staff attendance via Face ID biometrics, and dispatches automated "
        "photographic proof alerts to management via Telegram in under 2 seconds."
    )
    story.append(Paragraph(exec_summary_text, body_style))

    # Privacy Box
    privacy_callout = [
        [
            Paragraph("<b>100% Local On-Premise Privacy Guarantee:</b> All neural network inferences (YOLO11, YuNet, SFace, OSNet) "
                      "execute locally on the venue workstation. Raw video streams are processed entirely on-device and never "
                      "streamed to external public cloud servers, guaranteeing data sovereignty and zero cloud bandwidth costs.", callout_style)
        ]
    ]
    priv_table = Table(privacy_callout, colWidths=[504])
    priv_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#93C5FD")),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(priv_table)
    story.append(Spacer(1, 10))

    # 2. System Architecture & Tiered Layers
    story.append(Paragraph("2. System Architecture & Tiered Components", h1_style))
    arch_intro = (
        "The software is structured into four highly decoupled, resilient architectural layers designed for sub-second "
        "latency and fault-tolerant continuous uptime:"
    )
    story.append(Paragraph(arch_intro, body_style))

    layers_data = [
        [
            Paragraph("Layer", table_header_style),
            Paragraph("Core Components", table_header_style),
            Paragraph("Key Responsibilities & Technologies", table_header_style)
        ],
        [
            Paragraph("<b>1. Ingest Layer</b>", table_cell_bold),
            Paragraph("• go2rtc Gateway<br/>• AsyncFrameGrabber<br/>• Protocol Adapters", table_cell_style),
            Paragraph("Multiplexes RTSP IP cameras, Android IP Webcams, USB webcams, and local video loops. Non-blocking frame buffers eliminate queue backpressure and stream stuttering.", table_cell_style)
        ],
        [
            Paragraph("<b>2. Edge AI Engine</b>", table_cell_bold),
            Paragraph("• YOLO11 Nano Pose<br/>• YuNet + SFace Biometrics<br/>• Bay Zone Manager<br/>• Wi-Fi Tracker", table_cell_style),
            Paragraph("Multi-threaded Python 3.12 AI pipeline performing 17-keypoint worker pose estimation, facial recognition against staff galleries, bay ROI intersection math, and Wi-Fi ARP presence correlation.", table_cell_style)
        ],
        [
            Paragraph("<b>3. Persistence & Evidence</b>", table_cell_bold),
            Paragraph("• SQLite (events.db)<br/>• Proof Snapshot Engine<br/>• YAML Configuration", table_cell_style),
            Paragraph("Stores immutable timestamped incident logs, minute-by-minute technician productivity records, and high-resolution annotated evidence snapshots (stored in <code>edge/proofs/</code>).", table_cell_style)
        ],
        [
            Paragraph("<b>4. Presentation & Alerts</b>", table_cell_bold),
            Paragraph("• React 19 Operations SPA<br/>• Tauri v2 Rust Desktop App<br/>• Telegram Dispatcher<br/>• Embedded HTTP API", table_cell_style),
            Paragraph("Low-latency Web Operations Console, cross-platform native desktop application, embedded MJPEG streaming hub (:8765), and 1-click paired Telegram bot alert delivery.", table_cell_style)
        ]
    ]

    layers_table = Table(layers_data, colWidths=[90, 140, 274])
    layers_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_light]),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(layers_table)
    story.append(Spacer(1, 12))

    # Page Break for clean reading
    story.append(PageBreak())

    # 3. How the System Works: End-to-End Pipeline
    story.append(Paragraph("3. How the System Works (End-to-End Pipeline)", h1_style))
    story.append(Paragraph("From video capture to real-time Telegram dispatch, the data pipeline follows a strict, deterministic sequence:", body_style))

    pipeline_steps = [
        ("Step 1: Ingestion & Frame Normalization",
         "The camera adapter captures incoming frames at native resolution. The <code>AsyncFrameGrabber</code> worker normalizes orientation, flips/rotates feeds if specified, and serves low-latency frames to the AI inference thread."),
        
        ("Step 2: Pose & Human Activity Estimation",
         "The <b>YOLO11 Nano Pose</b> neural model (<code>yolo11n-pose.pt</code>) detects persons and extracts 17 skeletal keypoints (wrists, elbows, shoulders, knees). This allows the system to distinguish between an actively working technician (arms bent, active tool wrenching) versus an idle individual standing stationary."),
        
        ("Step 3: Multi-Person Biometrics & Physical Exclusivity",
         "When faces are visible, <b>OpenCV YuNet</b> detects facial landmarks and <b>SFace ONNX</b> generates 128-dimensional facial embeddings to match against enrolled staff photos. The system enforces the <b>Single-Camera Physical Exclusivity Law</b> (one physical human can never occupy two IDs in the same frame simultaneously) and uses anti-poisoning gallery thresholds to prevent identity drift."),
        
        ("Step 4: Bay Occupancy & Ghost-Filtering Hysteresis",
         "The <b>Bay Zone Manager</b> computes normalized polygon intersections between detected vehicles/workers and user-defined service bays. To prevent flickering during brief occlusions (e.g., when a mechanic walks behind a vehicle), a <code>GhostCounter</code> hysteresis buffer maintains state stability for up to 45 seconds."),
        
        ("Step 5: Wi-Fi Presence Correlation",
         "Auxiliary venue sensors scan local network ARP tables and ICMP pings. If an enrolled technician's phone is detected on the shop Wi-Fi, their presence is corroborated even if they step into camera blind spots (e.g., parts room or office)."),
        
        ("Step 6: Incident Detection & Telegram Dispatch",
         "When an operational rule triggers (such as after-hours motion, a vehicle left idle for >120s, or safety boundary breaches), the engine annotates an HD snapshot proof with bounding boxes and dispatches a rich photo alert directly to the shop manager's Telegram account within 1.5 seconds."),
        
        ("Step 7: Automated End-of-Day Operations Scorecards",
         "At the close of business, the reporting engine aggregates SQLite records and automatically generates daily scorecards detailing total vehicle throughput, bay utilization percentages, technician wrench-time minutes, and staff attendance logs.")
    ]

    for title, desc in pipeline_steps:
        story.append(Paragraph(f"<b>• {title}:</b> {desc}", bullet_style))

    story.append(Spacer(1, 10))

    # 4. Service Bay States & Operational Matrix
    story.append(Paragraph("4. Real-Time Service Bay Status Matrix", h1_style))
    story.append(Paragraph("The operations console color-codes service bays dynamically based on real-time computer vision inference:", body_style))

    bay_status_data = [
        [
            Paragraph("Status / Color", table_header_style),
            Paragraph("Visual Meaning", table_header_style),
            Paragraph("AI Detection Criteria", table_header_style),
            Paragraph("Business & Operational Impact", table_header_style)
        ],
        [
            Paragraph("<b>ACTIVE LABOR</b><br/>(Vibrant Green)", table_cell_bold),
            Paragraph("Vehicle being actively serviced", table_cell_style),
            Paragraph("Vehicle present inside ROI + Technician detected in active wrenching posture.", table_cell_style),
            Paragraph("Accumulates billable wrench-time on technician productivity scorecards.", table_cell_style)
        ],
        [
            Paragraph("<b>BAY IDLE</b><br/>(Warm Amber)", table_cell_bold),
            Paragraph("Vehicle stalled / Inactive", table_cell_style),
            Paragraph("Vehicle present inside ROI, but no tool movement detected for >120 seconds.", table_cell_style),
            Paragraph("Triggers manager alert to diagnose bottlenecks, missing parts, or staff distraction.", table_cell_style)
        ],
        [
            Paragraph("<b>AVAILABLE</b><br/>(Subtle Gray)", table_cell_bold),
            Paragraph("Empty service bay", table_cell_style),
            Paragraph("No vehicle or worker detected inside the bay boundary.", table_cell_style),
            Paragraph("Signals intake service advisor that the bay is ready for the next incoming vehicle.", table_cell_style)
        ]
    ]

    bay_table = Table(bay_status_data, colWidths=[95, 115, 140, 154])
    bay_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_light]),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(bay_table)
    story.append(Spacer(1, 12))

    # Page Break for clean reading
    story.append(PageBreak())

    # 5. Core Operational Modules & User Interfaces
    story.append(Paragraph("5. Core Operational Modules & User Interfaces", h1_style))
    story.append(Paragraph("The software offers two synchronized client interfaces: a web console and a native desktop app.", body_style))

    modules_data = [
        [
            Paragraph("Module", table_header_style),
            Paragraph("Interface / Path", table_header_style),
            Paragraph("Capabilities & Functionality", table_header_style)
        ],
        [
            Paragraph("<b>Live View Grid</b>", table_cell_bold),
            Paragraph("<code>/dashboard.html</code> (Tab 1)", table_cell_style),
            Paragraph("Multi-camera grid with real-time stream overlays, live FPS & inference latency counters (20–45ms), sound chimes, and interactive drag-and-drop ROI boundary configuration.", table_cell_style)
        ],
        [
            Paragraph("<b>Rules & Triggers</b>", table_cell_bold),
            Paragraph("<code>/dashboard.html</code> (Tab 2)", table_cell_style),
            Paragraph("Visual rule engine for customizing after-hours trespass alarms, maximum allowable bay dwell times, technician inactivity cooldowns, and notification priority levels.", table_cell_style)
        ],
        [
            Paragraph("<b>Incident Cases</b>", table_cell_bold),
            Paragraph("<code>/dashboard.html</code> (Tab 3)", table_cell_style),
            Paragraph("Full audit trail of logged security events and operational anomalies, complete with searchable filters, timestamped photo proofs, and resolution statuses.", table_cell_style)
        ],
        [
            Paragraph("<b>Face ID Attendance</b>", table_cell_bold),
            Paragraph("<code>edge/face_id.py</code> / Modal", table_cell_style),
            Paragraph("Automated facial enrollment and biometric recognition for staff. Clock-in and clock-out timestamps are registered automatically when workers enter camera view.", table_cell_style)
        ],
        [
            Paragraph("<b>Telegram Dispatch</b>", table_cell_bold),
            Paragraph("<code>TelegramPanel.tsx</code>", table_cell_style),
            Paragraph("1-click account pairing (no manual API tokens required). Instantly transmits high-resolution photo alerts, event descriptions, and daily KPI summaries to authorized smartphones.", table_cell_style)
        ]
    ]

    mod_table = Table(modules_data, colWidths=[110, 110, 284])
    mod_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_light]),
        ('PADDING', (0,0), (-1,-1), 5.5),
    ]))
    story.append(mod_table)
    story.append(Spacer(1, 12))

    # 6. Technical Specifications & Deployment Options
    story.append(Paragraph("6. Deployment & Hardware Specifications", h1_style))
    
    specs_data = [
        [
            Paragraph("Specification", table_header_style),
            Paragraph("Recommended Configuration", table_header_style)
        ],
        [
            Paragraph("<b>Operating System</b>", table_cell_bold),
            Paragraph("<b>Linux Ubuntu 20.04 / 22.04 / 24.04 LTS</b> (Verified & Stable production target)<br/><b>Windows 10 / 11 64-bit</b> (Standalone 1-click installation bundle available)", table_cell_style)
        ],
        [
            Paragraph("<b>Processor (CPU)</b>", table_cell_bold),
            Paragraph("Intel Core i5 / i7 (8th Gen+) or AMD Ryzen 5 / 7 (4+ cores for CPU-only inference @ 15–25 FPS)", table_cell_style)
        ],
        [
            Paragraph("<b>Memory (RAM)</b>", table_cell_bold),
            Paragraph("8 GB RAM minimum (16 GB recommended for 4+ concurrent HD video streams)", table_cell_style)
        ],
        [
            Paragraph("<b>Supported Video Sources</b>", table_cell_bold),
            Paragraph("• RTSP IP Security Cameras (Hikvision, Dahua, Tapo, Axis, UniFi)<br/>• Android Smartphones (via IP Webcam app over local Wi-Fi)<br/>• USB / V4L2 Webcams (Logitech, PC integrated)<br/>• Pre-Recorded Video Files (MP4, AVI) for forensic reviews and simulations", table_cell_style)
        ],
        [
            Paragraph("<b>AI Models & Frameworks</b>", table_cell_bold),
            Paragraph("• YOLO11 Nano Pose (PyTorch / ONNX) for human pose & labor estimation<br/>• OpenCV YuNet + SFace (ONNX) for face detection & 128-d biometric embeddings<br/>• ByteTrack + OSNet-x0.25 for multi-person tracking & re-identification", table_cell_style)
        ],
        [
            Paragraph("<b>Dual-Stream Optimization</b>", table_cell_bold),
            Paragraph("<b>Substream (360p/720p @ 15 FPS):</b> Continuous lightweight AI inference.<br/><b>Mainstream (1080p/4K):</b> Queried on-demand to capture high-definition proof snapshots.", table_cell_style)
        ]
    ]

    spec_table = Table(specs_data, colWidths=[140, 364])
    spec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_light]),
        ('PADDING', (0,0), (-1,-1), 5.5),
    ]))
    story.append(spec_table)
    story.append(Spacer(1, 10))

    # Summary Callout
    contact_box = [
        [
            Paragraph("<b>Technical Support & Inquiries:</b> For system onboarding, camera integration, or custom feature requests, contact the Inbound Engineering Crew at <b>inboundcrew82@gmail.com</b>.", callout_style)
        ]
    ]
    contact_table = Table(contact_box, colWidths=[504])
    contact_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(contact_table)

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    build_pdf()
