import time
import os
from pathlib import Path
from fastapi import Request
from fastapi.templating import Jinja2Templates
from utils.translations import TRANSLATIONS

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

BUILD_STAMP = time.time()
LIVE_RELOAD_ENABLED = os.environ.get("LIVE_RELOAD", "").lower() in ("1", "true", "yes")

def _render(request: Request, context: dict):
    lang = request.query_params.get("lang", "ar")
    if lang not in TRANSLATIONS:
        lang = "ar"
    
    context = {
        **context,
        "live_reload": LIVE_RELOAD_ENABLED,
        "build_stamp": BUILD_STAMP,
        "lang": lang,
        "trans": TRANSLATIONS[lang],
    }
    return templates.TemplateResponse(request, "index.html", context)
