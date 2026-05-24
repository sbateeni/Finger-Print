"""HTML sanitization and PDF conversion for forensic reports (xhtml2pdf-safe)."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)

_FONTS_REGISTERED = False
_ARABIC_FONT_NAME = "FpArabic"
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BUNDLED_ARABIC_FONT = _REPO_ROOT / "static" / "fonts" / "report-arabic.ttf"


def _system_arabic_font_candidates() -> list[Path]:
    return [
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "tahoma.ttf",
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf",
        Path("/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"),
        _REPO_ROOT / "static" / "fonts" / "NotoSansArabic-Regular.ttf",
        _REPO_ROOT / "static" / "fonts" / "NotoNaskhArabic-Regular.ttf",
    ]


def ensure_report_arabic_font() -> Path | None:
    """
    Copy a system Arabic-capable TTF into static/fonts/ so xhtml2pdf can read it
    (reading directly from Windows\\Fonts often fails when xhtml2pdf copies to temp).
    """
    if _BUNDLED_ARABIC_FONT.is_file() and _BUNDLED_ARABIC_FONT.stat().st_size > 10000:
        return _BUNDLED_ARABIC_FONT
    for source in _system_arabic_font_candidates():
        if not source.is_file():
            continue
        try:
            _BUNDLED_ARABIC_FONT.parent.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy2(source, _BUNDLED_ARABIC_FONT)
            return _BUNDLED_ARABIC_FONT
        except OSError as e:
            logger.warning("Could not bundle Arabic PDF font from %s: %s", source, e)
    return None


def report_prefers_html_delivery(lang: str | None = None, html: str | None = None) -> bool:
    """Arabic forensic reports should be downloaded/viewed as HTML (PDF engine lacks proper RTL)."""
    lg = (lang or "").strip().lower()
    if lg == "ar" or lg.startswith("ar-"):
        return True
    return _html_is_arabic(html or "")


def _html_is_arabic(html: str) -> bool:
    if re.search(r'<html[^>]*\blang=["\']ar', html, re.I):
        return True
    if re.search(r'<html[^>]*\bdir=["\']rtl', html, re.I):
        return True
    if re.search(r"[\u0600-\u06FF]", html):
        return True
    return False


def resolve_arabic_font_path() -> Path | None:
    bundled = ensure_report_arabic_font()
    if bundled and bundled.is_file():
        return bundled
    for path in _system_arabic_font_candidates():
        if path.is_file():
            return path
    return None


def _register_pdf_font_file(reg_name: str, path: Path) -> None:
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        if reg_name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(reg_name, str(path.resolve())))
    except Exception as e:
        logger.debug("PDF font %s registration skipped: %s", reg_name, e)


def _register_pdf_fonts() -> None:
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    try:
        from reportlab.pdfbase import pdfmetrics

        fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        for reg_name, filename in (
            ("Tahoma", "tahoma.ttf"),
            ("Tahoma-Bold", "tahomabd.ttf"),
        ):
            path = fonts_dir / filename
            if path.is_file() and reg_name not in pdfmetrics.getRegisteredFontNames():
                _register_pdf_font_file(reg_name, path)
        arabic = resolve_arabic_font_path()
        if arabic:
            _register_pdf_font_file(_ARABIC_FONT_NAME, arabic)
        _FONTS_REGISTERED = True
    except Exception as e:
        logger.debug("PDF font registration skipped: %s", e)


def _arabic_pdf_css() -> str:
    family = f"'{_ARABIC_FONT_NAME}', Tahoma, Arial, sans-serif"
    return f"""
body, .container, .hero, .disclaimer, .audit, .result-box, p, h1, h2, h3, td, th, li, span, strong {{
  font-family: {family} !important;
  direction: rtl;
}}
table.params {{ direction: rtl; }}
"""


def _arabic_font_embed_tag(font_path: Path) -> str:
    """Embed font via PML tag (avoids @font-face temp-file issues on some Windows setups)."""
    src = _font_url_for_pdf(font_path)
    return f'<pdf:fontembed name="{_ARABIC_FONT_NAME}" src="{src}"/>'


def _font_url_for_pdf(font_path: Path) -> str:
    return str(font_path.resolve()).replace("\\", "/")


def sanitize_html_for_pdf(html: str) -> str:
    """
    Strip external resources and CSS that break xhtml2pdf.
    For Arabic: embed local TTF, enable RTL shaping via <pdf:language name="arabic"/>.
    """
    out = html
    out = re.sub(
        r'<link[^>]+href=["\']https?://[^"\']+["\'][^>]*/?>',
        "",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"@import\s+url\([^)]+\)\s*;", "", out, flags=re.IGNORECASE)
    out = re.sub(r"linear-gradient\([^)]+\)", "#fff9e6", out, flags=re.IGNORECASE)
    out = out.replace("'Noto Sans Arabic', ", "")
    out = out.replace("Noto Sans Arabic, ", "")
    # Remove any @font-face blocks (xhtml2pdf may copy fonts to a locked temp path on Windows)
    out = re.sub(r"@font-face\s*\{[^}]*\}", "", out, flags=re.IGNORECASE)

    if _html_is_arabic(out):
        font_path = ensure_report_arabic_font()
        if font_path:
            _register_pdf_font_file(_ARABIC_FONT_NAME, font_path)
        if re.search(r"<style[^>]*>", out, re.I):
            out = re.sub(
                r"(<style[^>]*>)",
                r"\1\n" + _arabic_pdf_css(),
                out,
                count=1,
                flags=re.IGNORECASE,
            )
        embed = _arabic_font_embed_tag(font_path) if font_path else ""
        lang = '<pdf:language name="arabic"/>'
        inject = "\n".join(x for x in (embed, lang) if x)
        if inject and not re.search(r"<pdf:language\b", out, re.I):
            out = re.sub(
                r"(<body[^>]*>)",
                rf"\1\n{inject}",
                out,
                count=1,
                flags=re.IGNORECASE,
            )
        out = re.sub(
            r"font-family:\s*[^;]+;",
            f"font-family: {_ARABIC_FONT_NAME}, Tahoma, Arial, sans-serif;",
            out,
            flags=re.IGNORECASE,
        )
    else:
        out = re.sub(
            r"font-family:\s*[^;]+;",
            "font-family: 'Segoe UI', Tahoma, Arial, sans-serif;",
            out,
            count=1,
            flags=re.IGNORECASE,
        )

    return out


def _pdf_link_callback(uri: str, rel: str) -> str:
    if not uri:
        return uri
    if uri.startswith("data:"):
        return uri
    if uri.startswith("file:"):
        parsed = urlparse(uri)
        path = unquote(parsed.path)
        if path.startswith("/") and len(path) > 2 and path[2] == ":":
            path = path[1:]
        if os.path.isfile(path):
            return str(Path(path).resolve())
    norm = uri.replace("\\", "/")
    if len(norm) > 2 and norm[1] == ":":
        p = Path(norm)
        if p.is_file():
            return str(p.resolve())
    path = Path(uri)
    if path.is_file():
        return str(path.resolve())
    repo_rel = _REPO_ROOT / uri.lstrip("/")
    if repo_rel.is_file():
        return str(repo_rel.resolve())
    return uri


def html_to_pdf(html_text: str, pdf_path: Path) -> bool:
    """Convert sanitized HTML to PDF. Returns True on success."""
    from xhtml2pdf import pisa

    if _html_is_arabic(html_text):
        ensure_report_arabic_font()
    _register_pdf_fonts()
    safe_html = sanitize_html_for_pdf(html_text)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with pdf_path.open("wb") as pdf_file:
        result = pisa.CreatePDF(
            src=safe_html,
            dest=pdf_file,
            encoding="utf-8",
            link_callback=_pdf_link_callback,
        )
    if result.err:
        logger.warning("xhtml2pdf reported errors for %s", pdf_path.name)
        return False
    return pdf_path.is_file() and pdf_path.stat().st_size > 0
