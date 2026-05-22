"""
واجهة FastAPI — مناسبة للإنتاج وREST API مقارنةً بـ Streamlit.

تطوير مع إعادة تحميل تلقائية للمتصفح عند تغيير الكود:
  PowerShell: .\\run_dev.ps1
  أو (في PowerShell لا تضع *.py بين علامات اقتباس مزدوجة لأنها تُوسَّع إلى أسماء ملفات):
  $env:LIVE_RELOAD='1'; uvicorn server:app --reload --host 127.0.0.1 --port 8000 --reload-dir .
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import APP_VERSION
from routers import pages, report, analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="نظام تحليل البصمات — وضع مخبري", version=APP_VERSION)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

app.include_router(pages.router)
app.include_router(report.router)
app.include_router(analysis.router)
