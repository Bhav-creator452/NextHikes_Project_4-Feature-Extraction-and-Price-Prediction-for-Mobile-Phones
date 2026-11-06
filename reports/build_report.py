"""Render reports/project_report.md to reports/project_report.pdf.

    python reports/build_report.py

Supports the Markdown subset used in the report: headings, paragraphs, **bold**, *italic*, `code`,
bullet and numbered lists, pipe tables, images and <!-- pagebreak -->.
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
SRC, OUT = HERE / "project_report.md", HERE / "project_report.pdf"

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
try:  # DejaVu has the rupee sign; fall back to Helvetica if unavailable.
    pdfmetrics.registerFont(TTFont("Body", str(FONT_DIR / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Italic", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Mono", str(FONT_DIR / "DejaVuSansMono.ttf")))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")
    BODY, BOLD, MONO = "Body", "Body-Bold", "Body-Mono"
except Exception:  # pragma: no cover
    BODY, BOLD, MONO = "Helvetica", "Helvetica-Bold", "Courier"

INK, MUTED, ACCENT, RULE = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#2a78d6"), colors.HexColor("#d9d8d2")
base = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=base["Title"], fontName=BOLD, fontSize=24, leading=30, textColor=INK, spaceAfter=18, alignment=TA_CENTER),
    "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName=BOLD, fontSize=15, leading=19, textColor=INK, spaceBefore=14, spaceAfter=6),
    "h3": ParagraphStyle("h3", parent=base["Heading3"], fontName=BOLD, fontSize=12, leading=15, textColor=INK, spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("b", parent=base["BodyText"], fontName=BODY, fontSize=9.6, leading=13.6, textColor=INK, spaceAfter=6),
    "cell": ParagraphStyle("c", fontName=BODY, fontSize=8, leading=10, textColor=INK),
    "cellh": ParagraphStyle("ch", fontName=BOLD, fontSize=8, leading=10, textColor=colors.white),
    "cap": ParagraphStyle("cap", fontName=BODY, fontSize=8, leading=10, textColor=MUTED, alignment=TA_CENTER, spaceAfter=10),
    "center": ParagraphStyle("cen", fontName=BODY, fontSize=11, leading=16, textColor=MUTED, alignment=TA_CENTER),
}


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", rf'<font name="{MONO}">\1</font>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    return text


def table(lines: list[str], width: float):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    data = [[Paragraph(inline(c), S["cellh"] if i == 0 else S["cell"]) for c in r] for i, r in enumerate(rows)]
    ncol = len(rows[0])
    lens = [max(len(r[j]) if j < len(r) else 0 for r in rows) for j in range(ncol)]
    weights = [min(max(l, 12), 60) for l in lens]
    col_w = [width * w / sum(weights) for w in weights]
    t = Table(data, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f1")]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def image(path: Path, caption: str, width: float):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    draw_w = min(width, 16.5 * cm)
    draw_h = draw_w * h / w
    if draw_h > 11 * cm:
        draw_h = 11 * cm
        draw_w = draw_h * w / h
    return KeepTogether([Image(str(path), width=draw_w, height=draw_h), Paragraph(inline(caption), S["cap"])])


def build():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="Feature Extraction and Price Prediction for Mobile Phones",
                            author="NextHikes Project 4")
    width = doc.width
    story, i, first_page = [], 0, True
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1; continue
        if s == "<!-- pagebreak -->":
            story.append(PageBreak()); first_page = False; i += 1; continue
        if s.startswith("# "):
            story += [Spacer(1, 6 * cm), Paragraph(inline(s[2:]), S["title"])]; i += 1; continue
        if s.startswith("## "):
            story.append(Paragraph(inline(s[3:]), S["h2"])); i += 1; continue
        if s.startswith("### "):
            story.append(Paragraph(inline(s[4:]), S["h3"])); i += 1; continue
        m = re.match(r"!\[(.*?)\]\((.*?)\)", s)
        if m:
            story.append(image(HERE / m.group(2), m.group(1), width)); i += 1; continue
        if s.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            story += [table(block, width), Spacer(1, 8)]; continue
        if re.match(r"^(- |\d+\. )", s):
            items, numbered = [], bool(re.match(r"^\d+\. ", s))
            while i < len(lines) and re.match(r"^(\s*)(- |\d+\. )", lines[i]):
                text = re.sub(r"^\s*(- |\d+\. )", "", lines[i]); i += 1
                while (i < len(lines) and lines[i][:1] in (" ", "\t") and lines[i].strip()
                       and not re.match(r"^\s*(- |\d+\. )", lines[i])):
                    text += " " + lines[i].strip(); i += 1
                items.append(ListItem(Paragraph(inline(text), S["body"]), leftIndent=12))
            story.append(ListFlowable(items, bulletType="1" if numbered else "bullet", start="1" if numbered else None,
                                      leftIndent=14, bulletFontName=BODY, bulletFontSize=8.5))
            continue
        para = [s]; i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||!\[|- |\d+\. |<!--)", lines[i].strip()):
            para.append(lines[i].strip()); i += 1
        style = S["center"] if first_page else S["body"]
        story.append(Paragraph(inline(" ".join(para)), style))

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont(BODY, 7.5); canvas.setFillColor(MUTED)
        canvas.drawString(2 * cm, 1 * cm, "Feature Extraction and Price Prediction for Mobile Phones")
        canvas.drawRightString(A4[0] - 2 * cm, 1 * cm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
