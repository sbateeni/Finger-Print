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

def resolve_ui_lang(code: str | None) -> str:
    c = (code or "ar").strip().lower()
    return c if c in TRANSLATIONS else "ar"


def _render(request: Request, context: dict, template_name: str = "index.html"):
    lang = resolve_ui_lang(request.query_params.get("lang"))
    
    context = {
        **context,
        "live_reload": LIVE_RELOAD_ENABLED,
        "build_stamp": BUILD_STAMP,
        "lang": lang,
        "trans": TRANSLATIONS[lang],
    }
    return templates.TemplateResponse(request, template_name, context)
