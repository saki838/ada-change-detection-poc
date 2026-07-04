"""Enforcement Notice PDF Generator.

Creates digital evidence packs (PDF notices) for legal documentation,
matching the UPLC scope document's "digital evidence packs for legal
documentation" requirement.

Each notice includes:
  - Case reference number and date
  - Violation details (class, area, severity, zone)
  - Permit reconciliation results (if applicable)
  - Legal references based on zone type
  - Site imagery reference (overlay image path)
  - Enforcement officer assignment
  - QR-like case reference footer

Requires fpdf2:
    pip install fpdf2
"""
from __future__ import annotations

import pathlib
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from src.database.models import Case, EnforcementNotice, NoticeStatus
from src.services.zone_rules import ZONE_CONFIGS


# ── Unicode sanitizer for Latin-1 PDF compatibility ────────────────

def _sanitize(text: str) -> str:
    """Replace Unicode characters that can't be encoded in Latin-1 with
    ASCII equivalents. Core PDF fonts like Helvetica only support Latin-1."""
    replacements = {
        "\u2014": "-",   # em dash → hyphen
        "\u2013": "-",   # en dash → hyphen
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2022": "*",   # bullet
        "\u2026": "...", # ellipsis
        "\u00b2": "2",   # superscript 2
        "\u00b0": " deg", # degree symbol
        "\u20b9": "Rs.", # rupee sign
        "\u00a0": " ",   # non-breaking space
    }
    for char, ascii_repl in replacements.items():
        text = text.replace(char, ascii_repl)
    return text


# ── Legal reference templates per zone type ─────────────────────────

LEGAL_REFERENCES = {
    "heritage": (
        "Ancient Monuments and Archaeological Sites and Remains Act, 1958; "
        "Agra Heritage Building Regulations, 2017; "
        "AMASR (Regulation) Rules, 2011"
    ),
    "green_belt": (
        "Uttar Pradesh Urban Planning and Development Act, 1973; "
        "Master Plan of Agra 2031 - Green Belt Protection Clause"
    ),
    "riverfront": (
        "Yamuna Riverfront Development Regulations, 2019; "
        "National Green Tribunal guidelines for river buffer zones; "
        "Environment Protection Act, 1986"
    ),
    "residential": (
        "Agra Development Authority Building Bye-laws, 2019; "
        "Uttar Pradesh Urban Planning and Development Act, 1973"
    ),
    "commercial": (
        "Agra Development Authority Building Bye-laws, 2019 (Commercial); "
        "Uttar Pradesh Shops and Establishments Act"
    ),
    "industrial": (
        "Factories Act, 1948; "
        "Uttar Pradesh Industrial Area Development Act; "
        "Environment Protection Act, 1986"
    ),
    "other": (
        "Agra Development Authority Building Bye-laws, 2019; "
        "Uttar Pradesh Urban Planning and Development Act, 1973"
    ),
}

SEVERITY_LABELS = {
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
    "critical": "CRITICAL",
}

SEVERITY_COLORS = {
    "low": (76, 175, 80),      # green
    "medium": (255, 193, 7),   # amber
    "high": (255, 152, 0),     # orange
    "critical": (244, 67, 54), # red
}


def generate_notice(
    case: Case,
    output_dir: str,
    overlay_image_path: Optional[str] = None,
) -> str:
    """Generate a PDF enforcement notice for a case.

    Args:
        case: The Case ORM object containing violation details.
        output_dir: Directory to write the PDF to.
        overlay_image_path: Optional path to the pipeline overlay image
            to embed in the notice.

    Returns:
        Path to the generated PDF file.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        raise ImportError(
            "fpdf2 is required for PDF generation. "
            "Install it with: pip install fpdf2"
        )

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    _add_header(pdf, case)
    _add_case_info(pdf, case)
    _add_violation_details(pdf, case)
    _add_legal_references(pdf, case)
    if overlay_image_path and pathlib.Path(overlay_image_path).exists():
        _add_image(pdf, overlay_image_path)
    _add_footer(pdf, case)

    filename = f"notice_{case.case_number.lower().replace('-', '_')}.pdf"
    filepath = output_path / filename
    pdf.output(str(filepath))
    return str(filepath)


def save_notice_record(
    session: Session,
    case: Case,
    file_path: str,
    legal_reference: Optional[str] = None,
) -> EnforcementNotice:
    """Save an enforcement notice record to the database."""
    zone_config = ZONE_CONFIGS.get(case.zone_type or "other", ZONE_CONFIGS["other"])
    ref = legal_reference or LEGAL_REFERENCES.get(case.zone_type or "other", LEGAL_REFERENCES["other"])

    notice = EnforcementNotice(
        case_id=case.id,
        notice_number=f"ADA-NOTICE-{case.case_number}",
        status=NoticeStatus.DRAFT,
        file_path=file_path,
        legal_reference=ref,
    )
    session.add(notice)
    session.commit()
    session.refresh(notice)
    return notice


# ── PDF layout helpers ──────────────────────────────────────────────


def _add_header(pdf: FPDF, case: Case):
    """Add the official header with ADA branding."""
    # Top bar
    pdf.set_fill_color(31, 51, 85)  # official indigo
    pdf.rect(0, 0, 210, 30, "F")

    pdf.set_text_color(247, 243, 236)  # warm cream
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_xy(15, 8)
    pdf.cell(0, 10, "AGRA DEVELOPMENT AUTHORITY", align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(15, 20)
    pdf.cell(0, 6, _sanitize("Enforcement Notice - Building Violation Detection System"), align="C")

    pdf.ln(35)


def _add_case_info(pdf: FPDF, case: Case):
    """Add case reference information block."""
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(31, 51, 85)
    pdf.cell(0, 8, _sanitize(f"Case Number: {case.case_number}"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 6, _sanitize(f"Date: {case.created_at.strftime('%d %B %Y, %H:%M UTC')}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, _sanitize(f"Status: {case.status.replace('_', ' ').title()}"), new_x="LMARGIN", new_y="NEXT")

    if case.assigned_to:
        pdf.cell(0, 6, _sanitize(f"Assigned Officer: {case.assigned_to}"), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)


def _add_violation_details(pdf: FPDF, case: Case):
    """Add the core violation information table."""
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(31, 51, 85)
    pdf.cell(0, 8, "Violation Details", new_x="LMARGIN", new_y="NEXT")

    # Severity badge
    sev_label = SEVERITY_LABELS.get(case.severity, "UNKNOWN")
    sev_color = SEVERITY_COLORS.get(case.severity, (100, 100, 100))

    # Details table
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)

    details = [
        ("Classification", case.violation_class.replace("_", " ").title()),
        ("Severity", sev_label),
        ("Area", _sanitize(f"{case.area_m2:.1f} m2")),
        ("Zone Type", (case.zone_type or "Unclassified").replace("_", " ").title()),
        ("Confidence", _sanitize(f"{case.confidence:.0%}")),
    ]

    if case.description:
        details.append(("Description", _sanitize(case.description[:200])))

    for label, value in details:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(50, 7, _sanitize(f"  {label}:"), new_x="END")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, _sanitize(f" {value}"), new_x="LMARGIN", new_y="NEXT")

    # Severity color indicator bar
    r, g, b = sev_color
    pdf.set_fill_color(r, g, b)
    pdf.rect(15, pdf.get_y(), 180, 3, "F")
    pdf.ln(6)


def _add_legal_references(pdf: FPDF, case: Case):
    """Add legal references based on zone type."""
    zone_type = case.zone_type or "other"
    refs = LEGAL_REFERENCES.get(zone_type, LEGAL_REFERENCES["other"])

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(31, 51, 85)
    pdf.cell(0, 8, "Legal References", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(0, 5, _sanitize(refs))
    pdf.ln(4)


def _add_image(pdf: FPDF, image_path: str):
    """Embed the overlay image in the notice."""
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(31, 51, 85)
    pdf.cell(0, 8, "Site Imagery", new_x="LMARGIN", new_y="NEXT")

    try:
        pdf.image(image_path, x=25, w=160)
        pdf.ln(4)
    except Exception:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 6, _sanitize("(Image could not be embedded)"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)


def _add_footer(pdf: FPDF, case: Case):
    """Add footer with disclaimer and reference."""
    pdf.ln(10)
    pdf.set_draw_color(31, 51, 85)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4, _sanitize(
        "This notice is computer-generated by the ADA AI Change Detection System. "
        "It serves as a preliminary enforcement notification and requires "
        "field verification by an authorized enforcement officer before any "
        "legal action is taken."
    ))

    pdf.ln(2)
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(0, 4, _sanitize(f"Reference: {case.case_number} | Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"), new_x="LMARGIN", new_y="NEXT")
