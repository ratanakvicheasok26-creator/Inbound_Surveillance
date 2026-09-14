#!/usr/bin/env python3
"""Generate a publication-grade PDF for Inbound Crew's Team & Organizational Structure.

Follows the exact visual formula and styling of the Inbound Surveillance Customer Manual:
- Goggled monkey logo banner
- Surveillance Green (#008F39) / Stealth Green (#0B833A) accents
- NumberedCanvas with running header and 'Page X of Y' footer
- Structured tables for team roles and commercial pricing models
- Callout cards for customer acquisition and competitive differentiation
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

# --- Theme Color Tokens (Identical to Customer Manual) ---
COLOR_PRIMARY = colors.HexColor("#008F39")       # Surveillance Green
COLOR_SECONDARY = colors.HexColor("#0B833A")     # Stealth Green
COLOR_DARK = colors.HexColor("#141414")          # Deep Charcoal/Black
COLOR_TEXT = colors.HexColor("#222222")          # Off-black body text
COLOR_MUTED = colors.HexColor("#555555")         # Muted gray text
COLOR_BG_LIGHT = colors.HexColor("#F8F9FA")      # Table alternate row
COLOR_BORDER = colors.HexColor("#D8DCE0")        # Subtle border
COLOR_CALLOUT_BG = colors.HexColor("#F0FDF4")    # Light green tint
COLOR_ACCENT_BG = colors.HexColor("#F0F9FF")     # Light blue tint for strategy


class NumberedCanvas(canvas.Canvas):
    """Canvas that adds running headers and footers with total page count."""

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
        header_text = "INBOUND SURVEILLANCE — TEAM & ORGANIZATIONAL STRUCTURE"
        self.drawString(margin, page_height - 25, header_text)
        self.setStrokeColor(COLOR_BORDER)
        self.setLineWidth(0.5)
        self.line(margin, page_height - 28, page_width - margin, page_height - 28)

        # Footer
        self.line(margin, 28, page_width - margin, 28)
        self.drawString(margin, 18, "Confidential — For FirstWave Pitch & Organizational Strategy Review")
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(page_width - margin, 18, page_str)
        self.restoreState()


def build_callout(text: str, title: str = "NOTE", kind: str = "green", styles=None):
    """Build a compact styled callout box matching the manual."""
    bg_color = COLOR_CALLOUT_BG if kind == "green" else COLOR_ACCENT_BG
    border_color = COLOR_PRIMARY if kind == "green" else colors.HexColor("#0284C7")

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
            fontSize=19,
            leading=23,
            textColor=COLOR_DARK,
            alignment=0,
            spaceAfter=2,
        ),
        "MainSubtitle": ParagraphStyle(
            "MainSubtitle",
            parent=sample_styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=12.5,
            textColor=COLOR_MUTED,
            spaceAfter=6,
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=sample_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14.5,
            textColor=COLOR_PRIMARY,
            spaceBefore=7,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=sample_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=12.5,
            textColor=COLOR_DARK,
            spaceBefore=5,
            spaceAfter=2,
            keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=sample_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.3,
            leading=11.3,
            textColor=COLOR_TEXT,
            spaceAfter=4,
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=sample_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.3,
            leading=11.3,
            textColor=COLOR_TEXT,
            leftIndent=12,
            firstLineIndent=-8,
            spaceAfter=2,
        ),
        "CalloutTitle": ParagraphStyle(
            "CalloutTitle",
            fontName="Helvetica-Bold",
            fontSize=7.8,
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
    # PAGE 1: EXECUTIVE SUMMARY, CUSTOMER ACQUISITION & BUSINESS MODEL
    # =========================================================================
    header_data = [
        [
            Image(str(logo_path), width=46, height=48) if logo_path.exists() else Paragraph("", styles["Body"]),
            [
                Paragraph("<b>INBOUND CREW</b> — Team & Organizational Structure", styles["MainTitle"]),
                Paragraph("Cross-Functional Startup Architecture · Sales Strategy · Engineering Engine · FirstWave", styles["MainSubtitle"]),
            ]
        ]
    ]
    header_table = Table(header_data, colWidths=[52, 488])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=COLOR_PRIMARY, spaceBefore=3, spaceAfter=6))

    # Section 1: Executive Summary
    story.append(Paragraph("1. Executive Summary: Breaking Down Startup Silos", styles["H1"]))
    story.append(Paragraph(
        "The Inbound Crew's organizational structure is explicitly designed to eliminate the separated, disconnected departments "
        "that routinely slow down early-stage technology startups. Instead of maintaining rigid boundaries between business development "
        "and technical engineering, our team operates in continuous cross-functional synchronization. Every team member understands "
        "both the codebase and the unit economics of the automotive repair shops we serve.",
        styles["Body"]
    ))
    story.append(Paragraph(
        "This structural blueprint details our division of labor, customer acquisition methodology, core technological ownership, "
        "and business strategy prepared for the <b>FirstWave Competition</b>.",
        styles["Body"]
    ))

    # Section 2: Customer Acquisition
    story.append(Paragraph("2. Customer Acquisition Strategy: Grassroots Collaboration", styles["H1"]))
    story.append(Paragraph(
        "Customer acquisition is not relegated to a detached sales silo—it is a shared team discipline. Every member of the Inbound Crew "
        "maintains active, authentic connections within local automotive, technical, and commercial business communities.",
        styles["Body"]
    ))
    story.append(Paragraph("• <b>Bypassing Cold Outreach:</b> Rather than burning capital on cold calling or digital ads that fail with local trades, we leverage personal introductions to get past the front office directly to shop owners.", styles["Bullet"]))
    story.append(Paragraph("• <b>Live Shop-Floor Validation:</b> We test and demonstrate our camera system directly on active shop floors. Experiencing real-world mechanic workflows firsthand ensures our software solves genuine garage bottlenecks.", styles["Bullet"]))

    # Section 3: Commercial Model & Sales Strategy
    story.append(Paragraph("3. Business, Vision & Revenue Strategy", styles["H1"]))
    story.append(Paragraph(
        "Our sales engine follows a low-friction <b>'Land-and-Expand'</b> trajectory designed for rapid adoption by independent garages:",
        styles["Body"]
    ))

    commercial_data = [
        [Paragraph("Commercial Stage", styles["TableHeader"]), Paragraph("Pricing & Model", styles["TableHeader"]), Paragraph("Customer Value & Expansion Mechanism", styles["TableHeader"])],
        [
            Paragraph("<b>Stage 1: Low-Friction Entry</b><br/>Single Bay Pilot", styles["TableCellBold"]),
            Paragraph("<b>$295 Flat Setup Fee</b><br/>(Single Bay Camera)", styles["TableCell"]),
            Paragraph("We install a single camera in one active service bay. The low upfront cost removes purchasing friction and makes it an effortless 'yes' for independent shop owners.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Stage 2: ROI Proof</b><br/>Daily Telegram Scorecard", styles["TableCellBold"]),
            Paragraph("<b>Instant Value Proof</b><br/>(Wrench-Time Recovery)", styles["TableCell"]),
            Paragraph("Owners receive automated end-of-day Telegram scorecards detailing recovered unbilled labor, idle bay minutes, and turnaround velocity. The system proves its ROI in week one.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Stage 3: SaaS Recurring</b><br/>Full Facility Expansion", styles["TableCellBold"]),
            Paragraph("<b>$99 / Month</b><br/>(Recurring Subscription)", styles["TableCell"]),
            Paragraph("Once initial value is established, we lock in the $99/month software subscription and expand camera coverage across all remaining service bays, scaling monthly recurring revenue (MRR).", styles["TableCell"])
        ],
    ]
    commercial_table = Table(commercial_data, colWidths=[120, 115, 305])
    commercial_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_DARK),
        ('GRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_BG_LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(commercial_table)

    story.append(Spacer(1, 4))
    story.append(build_callout(
        "<b>Core Strategic Wedge:</b> Independent auto shops operate on tight labor margins. Proving recovered wrench-time "
        "via automated daily Telegram messages transforms Inbound Surveillance from a security cost center into an indispensable profit driver.",
        title="COMMERCIAL ADVANTAGE",
        kind="green",
        styles=styles
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: LEADERSHIP ROLES, CORE ENGINEERING & CONCLUSION
    # =========================================================================
    story.append(Paragraph("4. Inbound Crew: Roles & Operational Responsibilities", styles["H1"]))
    story.append(Paragraph(
        "Each team member oversees specialized operational domains while maintaining complete visibility across the full system lifecycle:",
        styles["Body"]
    ))

    roster_data = [
        [Paragraph("Team Member", styles["TableHeader"]), Paragraph("Role & Title", styles["TableHeader"]), Paragraph("Core Operational & Technical Responsibilities", styles["TableHeader"])],
        [
            Paragraph("<b>Chanratharo Rath</b>", styles["TableCellBold"]),
            Paragraph("<b>Founder &<br/>Lead Architect</b>", styles["TableCellBold"]),
            Paragraph(
                "• Bridges cutting-edge edge AI architecture with bottom-line garage profitability.<br/>"
                "• Formulates system messaging focused on recovering lost revenue from unbilled wrench-time.<br/>"
                "• Directs brand identity: authored the gritty, high-contrast industrial 'terminal' UI aesthetic.<br/>"
                "• FirstWave Pitch Lead: creates live demo collateral, delivers stage presentations, and co-authors business plans.",
                styles["TableCell"]
            )
        ],
        [
            Paragraph("<b>Ly Sothun</b>", styles["TableCellBold"]),
            Paragraph("<b>Business Planning &<br/>Web Presence</b>", styles["TableCellBold"]),
            Paragraph(
                "• Manages operational business planning in close partnership with executive leadership.<br/>"
                "• Owns the <b>Lean Canvas</b> financial model ($295 setup fee + $99/month SaaS subscription).<br/>"
                "• Leads web engineering: builds and maintains the high-converting commercial landing page.<br/>"
                "• Governs milestone roadmaps, project timelines, and early customer acquisition funnels.",
                styles["TableCell"]
            )
        ],
        [
            Paragraph("<b>Eang HourMeng</b>", styles["TableCellBold"]),
            Paragraph("<b>Core Systems<br/>Engineer</b>", styles["TableCellBold"]),
            Paragraph(
                "• Co-leads end-to-end edge hardware and software system engineering.<br/>"
                "• Architects zero-cloud local edge pipeline (YOLO11 pose estimation & kinematic tracking).<br/>"
                "• Implements virtual safety/bay ROI zones and hysteresis filtering (GhostCounter).<br/>"
                "• Collaborates on integrating real-time telemetry graphics into the web operations console.",
                styles["TableCell"]
            )
        ],
        [
            Paragraph("<b>Sok Ratanak Vichea</b>", styles["TableCellBold"]),
            Paragraph("<b>Core Systems<br/>Engineer</b>", styles["TableCellBold"]),
            Paragraph(
                "• Co-leads computer vision algorithms, camera adapters (RTSP/ONVIF/Phone), and edge inference.<br/>"
                "• Engineers automated Telegram dispatch engine: real-time photo proofing and daily scorecards.<br/>"
                "• Optimizes local ONNX runtimes (YuNet + SFace) for high FPS without cloud dependency.<br/>"
                "• Ensures live production builds reflect the industrial FirstWave competition brand standards.",
                styles["TableCell"]
            )
        ],
    ]
    roster_table = Table(roster_data, colWidths=[95, 95, 350])
    roster_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_DARK),
        ('GRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_BG_LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(roster_table)

    story.append(Spacer(1, 4))
    # Section 5: Core Engineering Engine
    story.append(Paragraph("5. Core Engineering Engine: Autonomous Edge Intelligence", styles["H1"]))
    story.append(Paragraph(
        "HourMeng and Vichea represent the technical engine of Inbound Crew. Their mission is building an <b>'invisible'</b> tracking system "
        "that monitors productivity, safety, and bay occupancy without interrupting mechanics while they work. By engineering the system "
        "entirely from scratch to execute locally, they eliminate expensive cloud GPU bandwidth costs and latency delays.",
        styles["Body"]
    ))

    # Section 6: Conclusion
    story.append(Paragraph("6. Conclusion: Built for the Garage Floor", styles["H1"]))
    story.append(Paragraph(
        "Ultimately, the defining strength of the Inbound Crew is that we keep our technology grounded in shop-floor reality. "
        "In the fast-paced, high-pressure world of commercial auto repair, technology is only valuable if it actively protects the garage owner's revenue.",
        styles["Body"]
    ))
    story.append(Paragraph(
        "By tearing down the walls between coding and selling, we have built an agile team that is equally at home optimizing neural network "
        "runtimes as we are standing on a greasy garage floor pitching directly to an owner. We are not just developers releasing another generic "
        "software package; we are a united crew deploying invisible tools tailored specifically for trade businesses.",
        styles["Body"]
    ))

    story.append(Spacer(1, 4))
    story.append(build_callout(
        "<b>Summary for FirstWave Pitch:</b> Inbound Crew blends deep edge AI engineering with authentic trades empathy. "
        "We have the technical skill to build Inbound Surveillance from scratch, the commercial acumen to monetize it, and the relentless drive to scale.",
        title="TEAM READINESS & VISION",
        kind="green",
        styles=styles
    ))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Team Structure PDF Generated successfully: {pdf_path}")
    print(f"File Size: {pdf_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/Inbound_Crew_Team_Structure.pdf"
    generate_pdf(out)
