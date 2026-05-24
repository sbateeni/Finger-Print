from pathlib import Path

from services.analysis_service.reports import resolve_report_download, should_generate_pdf
from utils.report_pdf import report_prefers_html_delivery


def test_arabic_prefers_html_delivery():
    html = '<html lang="ar" dir="rtl"><body>اختبار</body></html>'
    assert report_prefers_html_delivery("ar", html)
    assert not should_generate_pdf(Path("nope.html"), lang="ar")  # missing file


def test_resolve_pdf_request_returns_html_for_ar(tmp_path):
    p = tmp_path / "forensic_report.html"
    p.write_text('<html lang="ar" dir="rtl"><body>تقرير</body></html>', encoding="utf-8")
    path, media, name, attach = resolve_report_download(p, "pdf", lang="ar", force_download=True)
    assert path == p
    assert "html" in media
    assert attach is True
    assert name.endswith(".html")


def test_resolve_pdf_for_english(tmp_path):
    p = tmp_path / "report.html"
    p.write_text('<html lang="en"><body>Report</body></html>', encoding="utf-8")
    assert not report_prefers_html_delivery("en", p.read_text(encoding="utf-8"))
    assert should_generate_pdf(p, lang="en")
