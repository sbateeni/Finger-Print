from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, Response

from config import (
    APP_VERSION,
    MATCH_SCORE_THRESHOLDS,
    MIN_MINUTIAE_RECOMMENDED,
    SOFTWARE_NAME,
    DEFAULT_BORDER_MARGIN,
    DEFAULT_MIN_DISTANCE,
    DEFAULT_MIN_CONTRAST,
    DEFAULT_MIN_ANGLE_DIFF,
)
from utils.web_utils import _render, LIVE_RELOAD_ENABLED, BUILD_STAMP

router = APIRouter()
BASE_DIR = Path(__file__).resolve().parent.parent

@router.get("/health")
async def health():
    return {"status": "ok"}

if LIVE_RELOAD_ENABLED:
    @router.get("/__dev/build-id")
    async def dev_build_stamp():
        """للمطابقة مع الصفحة المفتوحة: عند إعادة تشغيل الخادم يتغيّر المعرّف فيُعاد تحميل المتصفح."""
        return {"id": BUILD_STAMP}

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return _render(
        request,
        {
            "results": None,
            "error": None,
            "match": None,
            "report_download": None,
            "thresholds": MATCH_SCORE_THRESHOLDS,
            "software_name": SOFTWARE_NAME,
            "app_version": APP_VERSION,
            "min_minutiae_recommended": MIN_MINUTIAE_RECOMMENDED,
            "operator_name": "",
            "case_reference": "",
            "DEFAULT_BORDER_MARGIN": DEFAULT_BORDER_MARGIN,
            "DEFAULT_MIN_DISTANCE": DEFAULT_MIN_DISTANCE,
            "DEFAULT_MIN_CONTRAST": DEFAULT_MIN_CONTRAST,
            "DEFAULT_MIN_ANGLE_DIFF": DEFAULT_MIN_ANGLE_DIFF,
        },
    )


@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    icon_path = BASE_DIR / "static" / "favicon.ico"
    if icon_path.exists():
        return FileResponse(str(icon_path))
    # Avoid noisy 404 in browser console when favicon is absent.
    return Response(status_code=204)


@router.get("/editor", response_class=HTMLResponse)
async def editor_page(request: Request, fingerprint_id: int = None):
    return _render(
        request,
        {
            "fingerprint_id": fingerprint_id,
            "software_name": SOFTWARE_NAME,
            "app_version": APP_VERSION,
        },
        template_name="manual_editor.html",
    )
