"""Forensic report PDF generation."""

from __future__ import annotations

import logging
from pathlib import Path

from utils.report_pdf import html_to_pdf, report_prefers_html_delivery

logger = logging.getLogger(__name__)


def should_generate_pdf(html_path: Path, *, lang: str | None = None) -> bool:
    """False for Arabic reports — use HTML download instead."""
    if not html_path.is_file():
        return False
    html_text = html_path.read_text(encoding="utf-8", errors="replace")
    return not report_prefers_html_delivery(lang, html_text)


def resolve_report_download(
    html_path: Path,
    fmt: str,
    *,
    lang: str | None = None,
    force_download: bool = False,
) -> tuple[Path, str, str, bool]:
    """
    Resolve download target. Arabic + format=pdf → HTML attachment (correct glyphs).
    Returns (path, media_type, filename, content_disposition_attachment).
    """
    html_text = html_path.read_text(encoding="utf-8", errors="replace")
    prefer_html = report_prefers_html_delivery(lang, html_text)
    fmt_l = (fmt or "html").lower()

    if fmt_l == "pdf" and prefer_html:
        return html_path, "text/html; charset=utf-8", html_path.name, True

    if fmt_l == "pdf":
        pdf_path = _ensure_pdf_from_html(html_path, lang=lang)
        return pdf_path, "application/pdf", pdf_path.name, force_download

    return html_path, "text/html; charset=utf-8", html_path.name, force_download


def _ensure_pdf_from_html(html_path: Path, *, lang: str | None = None) -> Path:
    """
    Convert report HTML into a PDF file using xhtml2pdf.
    Returns the PDF path (same folder, same basename).
    """
    pdf_path = html_path.with_suffix(".pdf")
    html_text = html_path.read_text(encoding="utf-8", errors="replace")

    if report_prefers_html_delivery(lang, html_text):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail=(
                "التقرير العربي متاح بصيغة HTML فقط (عرض صحيح للغة). "
                "استخدم تحميل HTML أو أضف ?format=html"
            ),
        )

    if pdf_path.exists() and pdf_path.stat().st_mtime >= html_path.stat().st_mtime:
        return pdf_path

    try:
        from xhtml2pdf import pisa  # noqa: F401 — optional dependency check
    except Exception as e:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=500,
            detail=f"PDF engine not installed. Run: pip install xhtml2pdf ({e})",
        ) from e

    if html_to_pdf(html_text, pdf_path):
        return pdf_path

    logger.warning("PDF generation failed for %s — HTML report remains available", html_path)
    from fastapi import HTTPException

    raise HTTPException(
        status_code=500,
        detail="Failed to generate PDF from report. Open the HTML report in the browser.",
    )
