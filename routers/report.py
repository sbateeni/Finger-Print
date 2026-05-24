from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from config import OUTPUT_DIR
from services.analysis_service.reports import resolve_report_download

router = APIRouter()


@router.get("/download-report/{filepath:path}")
async def download_report(filepath: str, request: Request):
    base = Path(OUTPUT_DIR).resolve()
    path = (base / filepath).resolve()

    if not str(path).startswith(str(base)) or not path.is_file():
        raise HTTPException(404, "التقرير غير موجود")

    fmt = (request.query_params.get("format") or "html").lower()
    force_download = request.query_params.get("download") == "1"
    lang = (request.query_params.get("lang") or "").strip().lower()

    target_path, media_type, filename, as_attachment = resolve_report_download(
        path,
        fmt,
        lang=lang or None,
        force_download=force_download,
    )

    return FileResponse(
        target_path,
        media_type=media_type,
        filename=filename,
        content_disposition_type="attachment" if as_attachment else "inline",
    )
