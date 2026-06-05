#!/usr/bin/env python3
"""Generate hackathon submission deck: AgenticInsights_Deck.pdf (max 10 slides)."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "AgenticInsights_Deck.pdf"
ASSETS = Path("/Users/shiraptinath/.cursor/projects/Users-shiraptinath-Desktop-Build/assets")
ARTIFACTS = PROJECT_ROOT / "data/artifacts/hr_fix_demo2"

# User-provided screenshots (Jun 5, 8:01 AM)
SCREEN_TITLE_REF = ASSETS / "Screenshot_2026-06-05_at_8.01.15_AM-95562583-4c1d-4d4c-a6f6-d6991a4e87b8.png"
# Earlier anomalies demo (before 8.01 overwrite)
SCREEN_ANOMALIES = ASSETS / "Screenshot_2026-06-05_at_7.39.11_AM-d363b773-75eb-4241-872b-873d7af126e5.png"
SCREEN_PIPELINE = ASSETS / "Screenshot_2026-06-05_at_7.03.46_AM-eab87632-0ff2-4440-8b28-3b667aa2c045.png"

# 16:9 slide size (points)
SLIDE_W, SLIDE_H = landscape((13.333 * inch, 7.5 * inch))

NAVY = colors.HexColor("#1e3a8a")
BLUE = colors.HexColor("#2563eb")
SLATE = colors.HexColor("#0f172a")
MUTED = colors.HexColor("#64748b")
LIGHT_BG = colors.HexColor("#f8fafc")


def _resize_image(path: Path, max_w: int = 1100, max_h: int = 620) -> io.BytesIO:
    """Resize screenshot for PDF embedding."""
    img = Image.open(path)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    buf.seek(0)
    return buf


def _img_flowable(path: Path, width: float, height: float | None = None) -> RLImage | None:
    if not path.exists():
        return None
    buf = _resize_image(path)
    pil = Image.open(buf)
    buf.seek(0)
    w_px, h_px = pil.size
    aspect = h_px / w_px
    h = height or (width * aspect)
    return RLImage(buf, width=width, height=min(h, 4.2 * inch))


def _slide_title(styles, title: str, subtitle: str = "") -> list:
    story: list = []
    story.append(Spacer(1, 0.35 * inch))
    story.append(
        Paragraph(
            f'<font color="#1e3a8a" size="22"><b>{title}</b></font>',
            ParagraphStyle("st", parent=styles["Heading1"], alignment=TA_LEFT),
        )
    )
    if subtitle:
        story.append(Spacer(1, 0.12 * inch))
        story.append(
            Paragraph(
                f'<font color="#64748b" size="12">{subtitle}</font>',
                ParagraphStyle("ss", parent=styles["Normal"], alignment=TA_LEFT),
            )
        )
    story.append(Spacer(1, 0.2 * inch))
    return story


def _bullets(styles, items: list[str]) -> list:
    story: list = []
    for item in items:
        story.append(
            Paragraph(
                f'<font color="#0f172a" size="13">• {item}</font>',
                ParagraphStyle("bu", parent=styles["Normal"], leftIndent=14, spaceAfter=8),
            )
        )
    return story


def _title_slide_block() -> Table:
    """Centered title slide — fixed line heights so text never overlaps."""
    title_style = ParagraphStyle(
        "DeckTitle",
        fontName="Helvetica-Bold",
        fontSize=32,
        leading=40,
        alignment=TA_CENTER,
        textColor=NAVY,
        spaceBefore=0,
        spaceAfter=0,
    )
    subtitle_style = ParagraphStyle(
        "DeckSubtitle",
        fontName="Helvetica",
        fontSize=17,
        leading=24,
        alignment=TA_CENTER,
        textColor=BLUE,
        spaceBefore=0,
        spaceAfter=0,
    )
    tagline_style = ParagraphStyle(
        "DeckTagline",
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        alignment=TA_CENTER,
        textColor=MUTED,
        spaceBefore=0,
        spaceAfter=0,
    )
    team_style = ParagraphStyle(
        "DeckTeam",
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=20,
        alignment=TA_CENTER,
        textColor=SLATE,
        spaceBefore=0,
        spaceAfter=0,
    )
    rows = [
        [Paragraph("Noise to Insight", title_style)],
        [Paragraph("AI Meets Data — Microsoft Build 2026", subtitle_style)],
        [Paragraph("From messy exports to executive-ready intelligence in one pipeline", tagline_style)],
        [Paragraph("Agentic Insights", team_style)],
    ]
    table = Table(rows, colWidths=[7.0 * inch])
    table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (0, 0), 0),
                ("BOTTOMPADDING", (0, 0), (0, 0), 4),
                ("TOPPADDING", (0, 1), (0, 1), 14),
                ("BOTTOMPADDING", (0, 1), (0, 1), 14),
                ("TOPPADDING", (0, 2), (0, 2), 20),
                ("BOTTOMPADDING", (0, 2), (0, 2), 10),
                ("TOPPADDING", (0, 3), (0, 3), 28),
                ("BOTTOMPADDING", (0, 3), (0, 3), 0),
            ]
        )
    )
    return table


def _architecture_table() -> Table:
    phases = [
        ("0 Ingest", "Profile schema & quality"),
        ("1 Clean", "LLM plan + Polars execution"),
        ("2 Patterns", "Correlations & segment lift"),
        ("3 Anomalies", "Isolation Forest + LLM explain"),
        ("4 Forecast", "Prophet / snapshot analytics"),
        ("5 Graph", "Knowledge graph (NetworkX + PyVis)"),
        ("6 Report", "Executive HTML / PDF"),
    ]
    data = [["Phase", "Capability"]] + list(phases)
    t = Table(data, colWidths=[1.6 * inch, 4.8 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("BACKGROUND", (0, 1), (-1, -1), LIGHT_BG),
                ("TEXTCOLOR", (0, 1), (-1, -1), SLATE),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return t


def build_deck() -> Path:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=(SLIDE_W, SLIDE_H),
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
    )
    story: list = []

    # Slide 1 — Title (table layout — no overlapping lines)
    story.append(Spacer(1, 1.35 * inch))
    story.append(_title_slide_block())
    story.append(PageBreak())

    # Slide 2 — Problem
    story.extend(_slide_title(styles, "Problem Statement", "Data is everywhere — insight is rare"))
    story.extend(
        _bullets(
            styles,
            [
                "Teams receive raw CSV/JSON exports (HR, orders, ops) with inconsistent schemas, "
                "missing dates, and hidden relationships.",
                "Manual cleaning and one-off dashboards take days; critical signals stay buried.",
                "Leaders need answers: <i>What patterns matter? Who is an outlier? What happens next?</i>",
                "Challenge: turn <b>noise</b> into <b>actionable insight</b> without a data-science team.",
            ],
        )
    )
    story.append(PageBreak())

    # Slide 3 — Solution
    story.extend(_slide_title(styles, "Solution Overview", "Six-phase agentic pipeline"))
    story.extend(
        _bullets(
            styles,
            [
                "<b>Noise to Insight</b> — upload messy files; LangGraph orchestrates six phases end-to-end.",
                "Outputs per run: cleaned data, ranked insights, explained anomalies, forecast/snapshot chart, "
                "knowledge graph, executive report.",
                "<b>Streamlit Mission Control</b> — phase-by-phase tabs, KPIs, charts, downloads.",
                "Works with or without Azure OpenAI (statistical core + LLM enrichment).",
            ],
        )
    )
    story.append(PageBreak())

    # Slide 4 — Architecture
    story.extend(_slide_title(styles, "Architecture", "Ingest → Clean → Discover → Detect → Predict → Graph → Report"))
    story.append(_architecture_table())
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            '<font color="#64748b" size="11">Stack: Polars, DuckDB, scikit-learn, Prophet, NetworkX, '
            "LangGraph, Azure OpenAI, Streamlit</font>",
            styles["Normal"],
        )
    )
    story.append(PageBreak())

    # Slide 5 — AI integration
    story.extend(_slide_title(styles, "AI Integration Details", "Azure OpenAI at every decision point"))
    ai_rows = [
        ["Phase", "AI role", "Fallback"],
        ["Cleaning", "LLM generates column rename/drop/impute plan", "Heuristic schema normalization"],
        ["Patterns", "Ranked insight cards from statistics", "Template insights from correlations"],
        ["Anomalies", "Hypothesis + recommended actions", "Isolation Forest + templates"],
        ["Forecast", "Narrative + prescriptive actions", "Prophet / sklearn + snapshot bars"],
        ["Graph", "Entity/relation extraction from samples", "Co-occurrence heuristic graph"],
        ["Report", "Executive summary synthesis", "Jinja HTML + optional WeasyPrint PDF"],
    ]
    t = Table(ai_rows, colWidths=[1.1 * inch, 2.9 * inch, 2.2 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t)
    story.append(PageBreak())

    # Slide 6 — Demo pipeline
    story.extend(_slide_title(styles, "Demo — Pipeline & Headline Insight", "HR Employee Attrition dataset (1,470 rows)"))
    shot1 = SCREEN_PIPELINE if SCREEN_PIPELINE.exists() else ASSETS / "Screenshot_2026-06-05_at_7.03.46_AM-eab87632-0ff2-4440-8b28-3b667aa2c045.png"
    img1 = _img_flowable(shot1, 6.8 * inch)
    if img1:
        story.append(img1)
    else:
        story.append(Paragraph("<i>Screenshot: Streamlit pipeline progress</i>", styles["Normal"]))
    story.append(PageBreak())

    # Slide 7 — Anomalies (earlier screenshot — 7.39.11 AM)
    story.extend(_slide_title(styles, "Demo — Anomaly Detection", "Multivariate outliers with employee IDs & actions"))
    shot2 = SCREEN_ANOMALIES
    img2 = _img_flowable(shot2, 6.5 * inch, 4.0 * inch)
    if img2:
        story.append(img2)
    else:
        story.append(Paragraph("<i>Anomaly detection screenshot not found</i>", styles["Normal"]))
    story.append(PageBreak())

    # Slide 8 — Forecast + Graph charts
    story.extend(_slide_title(styles, "Demo — Forecast & Knowledge Graph", "Snapshot analytics + entity network"))
    fc = ARTIFACTS / "forecast.png"
    gr = ARTIFACTS / "graph.png"
    row_imgs: list = []
    if fc.exists():
        row_imgs.append(_img_flowable(fc, 3.2 * inch, 2.4 * inch))
    if gr.exists():
        row_imgs.append(_img_flowable(gr, 3.2 * inch, 2.4 * inch))
    if len(row_imgs) == 2:
        t_img = Table([row_imgs], colWidths=[3.4 * inch, 3.4 * inch])
        t_img.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story.append(t_img)
    elif row_imgs:
        story.append(row_imgs[0])
    story.append(
        Paragraph(
            '<font color="#64748b" size="10">Wow moment: JobLevel ↔ MonthlyIncome r≈0.95; '
            "department income snapshot; connected HR entities</font>",
            styles["Normal"],
        )
    )
    story.append(PageBreak())

    # Slide 9 — Impact
    story.extend(_slide_title(styles, "Impact & Theme Alignment", "“I had no idea that was in our data”"))
    story.extend(
        _bullets(
            styles,
            [
                "<b>Intelligent cleaning</b> — normalize 35 HR columns automatically.",
                "<b>Pattern discovery</b> — surface strong correlations & segment lifts.",
                "<b>Actionable anomalies</b> — who to investigate and why (not just scores).",
                "<b>Analytics</b> — time-series forecast or snapshot bars when no dates exist.",
                "<b>Executive report</b> — one-click HTML/PDF for leadership.",
            ],
        )
    )
    story.append(PageBreak())

    # Slide 10 — Team
    story.append(Spacer(1, 0.9 * inch))
    story.append(
        Paragraph(
            '<font color="#1e3a8a" size="28"><b>Team</b></font>',
            ParagraphStyle("team", alignment=TA_CENTER),
        )
    )
    story.append(Spacer(1, 0.45 * inch))
    team_data = [
        ["Team name", "Agentic Insights"],
        ["Team contributor", "Shiraptinath C R"],
        ["Type", "Solo Product Builder"],
        ["Project", "Noise to Insight"],
        ["Repository", str(PROJECT_ROOT.name)],
    ]
    tt = Table(team_data, colWidths=[2.2 * inch, 3.8 * inch])
    tt.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 14),
                ("TEXTCOLOR", (0, 0), (-1, -1), SLATE),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
                ("BOX", (0, 0), (-1, -1), 1, BLUE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 14),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ]
        )
    )
    story.append(tt)
    story.append(Spacer(1, 0.4 * inch))
    story.append(
        Paragraph(
            '<font color="#64748b" size="12">Thank you — Microsoft Build 2026 Hackathon</font>',
            ParagraphStyle("thanks", alignment=TA_CENTER),
        )
    )

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return OUTPUT


def _footer(canv: canvas.Canvas, _doc) -> None:
    canv.saveState()
    canv.setFont("Helvetica", 8)
    canv.setFillColor(MUTED)
    canv.drawString(0.55 * inch, 0.3 * inch, "Agentic Insights — Noise to Insight — Microsoft Build 2026")
    page_num = canv.getPageNumber()
    canv.drawRightString(SLIDE_W - 0.55 * inch, 0.3 * inch, f"Slide {page_num} / 10")
    canv.restoreState()


if __name__ == "__main__":
    path = build_deck()
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Created: {path}")
    print(f"Size: {size_mb:.2f} MB ({path.stat().st_size} bytes)")
