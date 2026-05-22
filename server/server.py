"""
واجهة FastAPI — مناسبة للإنتاج وREST API مقارنةً بـ Streamlit.

تطوير مع إعادة تحميل تلقائية للمتصفح عند تغيير الكود:
  PowerShell: .\\run_dev.ps1
  أو (في PowerShell لا تضع *.py بين علامات اقتباس مزدوجة لأنها تُوسَّع إلى أسماء ملفات):
  $env:LIVE_RELOAD='1'; uvicorn server:app --reload --host 127.0.0.1 --port 8000 --reload-dir .
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
import threading

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import APP_VERSION
from routers import pages, report, analysis
from services.analysis_queue import start_analysis_queue, stop_analysis_queue
from utils.git_updater import run_startup_auto_update, start_periodic_auto_update
from utils.runtime_platform import is_linux, telegram_stop_script_hint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


def _telegram_enabled() -> bool:
    if (os.getenv("TELEGRAM_ENABLED") or "").strip().lower() in ("0", "false", "no", "off"):
        return False
    return bool((os.getenv("TELEGRAM_BOT_TOKEN") or "").strip())


def _telegram_embedded() -> bool:
    return (os.getenv("TELEGRAM_EMBEDDED") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    if is_linux():
        logger.info("Server starting on Linux — %s", __import__("platform").platform())
    run_startup_auto_update(echo=False)
    stop_event = threading.Event()
    start_periodic_auto_update(stop_event)

    embedded_bot = None
    if _telegram_enabled() and _telegram_embedded():
        try:
            from pathlib import Path

            from bot.telegram_bot import start_embedded_bot
            from utils.telegram_polling import kill_stale_local_bot_processes

            n = kill_stale_local_bot_processes(BASE_DIR)
            if n:
                logger.info("Cleared %s stale worker(s) before Telegram start", n)
            embedded_bot = await start_embedded_bot()
            if embedded_bot is None:
                logger.warning(
                    "Telegram polling unavailable — run %s (one token = one machine)",
                    telegram_stop_script_hint(),
                )
        except Exception as e:
            logger.error("Embedded Telegram bot failed to start: %s", e)
    else:
        await start_analysis_queue()

    yield

    if embedded_bot is not None:
        try:
            from bot.telegram_bot import stop_embedded_bot

            await stop_embedded_bot()
        except Exception as e:
            logger.warning("Embedded Telegram shutdown: %s", e)
    else:
        await stop_analysis_queue()
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
