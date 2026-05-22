"""
واجهة FastAPI — مناسبة للإنتاج وREST API مقارنةً بـ Streamlit.

تطوير مع إعادة تحميل تلقائية للمتصفح عند تغيير الكود:
  PowerShell: .\\run_dev.ps1
  أو (في PowerShell لا تضع *.py بين علامات اقتباس مزدوجة لأنها تُوسَّع إلى أسماء ملفات):
  $env:LIVE_RELOAD='1'; uvicorn server:app --reload --host 127.0.0.1 --port 8000 --reload-dir .
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
import threading

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import APP_VERSION
from routers import pages, report, analysis
from utils.git_updater import run_startup_auto_update, start_periodic_auto_update

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    run_startup_auto_update()
    stop_event = threading.Event()
    start_periodic_auto_update(stop_event)
    yield
    stop_event.set()


app = FastAPI(
    title="نظام تحليل البصمات — وضع مخبري",
    version=APP_VERSION,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

app.include_router(pages.router)
app.include_router(report.router)
app.include_router(analysis.router)
