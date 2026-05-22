from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from config import OUTPUT_DIR

router = APIRouter()

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
        raise HTTPException(
            status_code=500,
            detail=f"PDF engine not installed. Run: pip install xhtml2pdf ({e})",
        ) from e

    html_text = html_path.read_text(encoding="utf-8", errors="replace")
    with pdf_path.open("wb") as pdf_file:
        result = pisa.CreatePDF(src=html_text, dest=pdf_file, encoding="utf-8")
    if result.err:
        raise HTTPException(status_code=500, detail="Failed to generate PDF from report.")
    return pdf_path

@router.get("/download-report/{filepath:path}")
async def download_report(filepath: str, request: Request):
    base = Path(OUTPUT_DIR).resolve()
    # تأمين المسار لمنع التسلل خارج مجلد المخرجات (Path Traversal Protection)
    path = (base / filepath).resolve()
    
    if not str(path).startswith(str(base)) or not path.is_file():
        raise HTTPException(404, "التقرير غير موجود")
        
    fmt = (request.query_params.get("format") or "html").lower()
    force_download = request.query_params.get("download") == "1"
    target_path = path
    media_type = "text/html; charset=utf-8"
    filename = path.name
    if fmt == "pdf":
        target_path = _ensure_pdf_from_html(path)
        media_type = "application/pdf"
        filename = target_path.name

    return FileResponse(
        target_path,
        media_type=media_type,
        filename=filename,
        content_disposition_type="attachment" if force_download else "inline",
    )
