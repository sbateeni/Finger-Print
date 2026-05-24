from pathlib import Path

from utils.report_pdf import (
    _html_is_arabic,
    ensure_report_arabic_font,
    sanitize_html_for_pdf,
)


def test_html_is_arabic():
    assert _html_is_arabic('<html lang="ar">')
    assert not _html_is_arabic('<html lang="en">')


def test_sanitize_injects_arabic_pdf_tags():
    html = '<html lang="ar" dir="rtl"><head><style>body{}</style></head><body><p>مرحبا</p></body></html>'
    out = sanitize_html_for_pdf(html)
    assert "pdf:language" in out
    assert "pdf:fontembed" in out
    assert "@font-face" not in out.lower()
    assert "FpArabic" in out


def test_ensure_report_arabic_font_windows():
    path = ensure_report_arabic_font()
    if path is None:
        return
    assert path.is_file()
    assert path.stat().st_size > 10000
