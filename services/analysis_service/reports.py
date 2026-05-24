"""Forensic report PDF generation."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _ensure_pdf_from_html(html_path: Path) -> Path:
    """
    Convert report HTML into a real PDF file using xhtml2pdf.
    Returns the PDF path (same folder, same basename).
    """
    pdf_path = html_path.with_suffix(".pdf")
    if pdf_path.exists() and pdf_path.stat().st_mtime >= html_path.stat().st_mtime:
        return pdf_path

    try:
        from xhtml2pdf import pisa  # lazy import: optional runtime dependency
    except Exception as e:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=500,
            detail=f"PDF engine not installed. Run: pip install xhtml2pdf ({e})",
        ) from e

    html_text = html_path.read_text(encoding="utf-8", errors="replace")
    with pdf_path.open("wb") as pdf_file:
        result = pisa.CreatePDF(src=html_text, dest=pdf_file, encoding="utf-8")
    if result.err:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail="Failed to generate PDF from report.")
    return pdf_path
