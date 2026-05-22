"""
Telegram bot: receive two fingerprint photos, run analysis, send PDF + summary.

Env:
  TELEGRAM_BOT_TOKEN          — required
  TELEGRAM_ALLOWED_CHAT_IDS   — optional comma-separated user/chat IDs (recommended)
"""

from __future__ import annotations

import asyncio
import logging
import os
import ssl
from typing import Optional, Set

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from services.analysis_queue import get_analysis_queue
from services.pair_analysis import format_match_summary_ar

logger = logging.getLogger(__name__)

SESSION_KEY = "fp_session"
_embedded_app: Optional[Application] = None


def _allowed_chat_ids() -> Optional[Set[int]]:
    raw = (os.getenv("TELEGRAM_ALLOWED_CHAT_IDS") or "").strip()
    if not raw:
        return None
    out: Set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit() or (part.startswith("-") and part[1:].isdigit()):
            out.add(int(part))
    return out or None


def _is_allowed(update: Update) -> bool:
    allowed = _allowed_chat_ids()
    if allowed is None:
        return True
    uid = update.effective_user.id if update.effective_user else None
    cid = update.effective_chat.id if update.effective_chat else None
    return uid in allowed or cid in allowed


def _get_session(context: ContextTypes.DEFAULT_TYPE) -> dict:
    if SESSION_KEY not in context.user_data:
        context.user_data[SESSION_KEY] = {"reference": None, "query": None}
    return context.user_data[SESSION_KEY]


async def _download_photo_bytes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[bytes]:
    if not update.message or not update.message.photo:
        return None
    photo = update.message.photo[-1]
    tg_file = await context.bot.get_file(photo.file_id)
    data = await tg_file.download_as_bytearray()
    return bytes(data)


async def _download_document_bytes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[bytes]:
    if not update.message or not update.message.document:
        return None
    doc = update.message.document
    mime = (doc.mime_type or "").lower()
    name = (doc.file_name or "").lower()
    if mime and not mime.startswith("image/") and not name.endswith(
        (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    ):
        return None
    tg_file = await context.bot.get_file(doc.file_id)
    data = await tg_file.download_as_bytearray()
    return bytes(data)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        await update.message.reply_text("غير مصرح لك باستخدام هذا البوت.")
        return
    context.user_data.pop(SESSION_KEY, None)
    await update.message.reply_text(
        "مرحبًا — بوت مطابقة البصمات 🔬\n\n"
        "1️⃣ أرسل صورة *البصمة الأصلية* (مرجعية)\n"
        "2️⃣ ثم أرسل صورة *البصمة المقارنة* (جزئية)\n\n"
        "أو أرسل صورتين متتاليتين كـ photo أو ملف image.\n"
        "الأوامر:\n"
        "/start — البداية\n"
        "/reset — إلغاء الصور الحالية\n"
        "/status — ما تم استلامه\n\n"
        "يمكنك أيضًا استخدام الواجهة على http://127.0.0.1:8000\n\n"
        "عند وجود طلبات متعددة يُوضَع طلبك في *طابور انتظار* ويُرسل موقعك في الدور.\n\n"
        "التحليل *عميق* على الكمبيوتر (محاذاة تلقائية + نفس المحرك كالواجهة) — "
        "تيليجرام يرسل الصور ويستقبل النتائج فقط.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return
    context.user_data.pop(SESSION_KEY, None)
    await update.message.reply_text("تم إعادة التعيين. أرسل البصمة الأصلية.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return
    s = _get_session(context)
    ref = "✅" if s.get("reference") else "❌"
    qry = "✅" if s.get("query") else "❌"
    await update.message.reply_text(f"المرجعية: {ref}\nالمقارنة: {qry}")


async def _run_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    s = _get_session(context)
    ref_b: bytes = s.get("reference")
    qry_b: bytes = s.get("query")
    if not ref_b or not qry_b:
        await update.message.reply_text("لم تكتمل الصورتان بعد.")
        return

    chat_id = update.effective_chat.id
    user = update.effective_user
    operator = user.username or str(user.id) if user else "telegram"

    q = get_analysis_queue()
    job, ahead = await q.enqueue_pair(
        ref_b,
        qry_b,
        source="telegram",
        chat_id=chat_id,
        source_label="تيليجرام",
        operator_name=f"telegram:{operator}",
        case_reference=f"TG-{chat_id}",
        write_report_and_audit=True,
    )
    if ahead == 0:
        await update.message.reply_text(
            "⏳ تم استلام طلبك — *تحليل عميق* على الكمبيوتر (قد يستغرق دقائق)…",
            parse_mode=ParseMode.MARKDOWN,
        )
    else:
        await update.message.reply_text(
            f"✅ تم إدخال طلبك في الطابور.\n"
            f"موقعك: *{ahead + 1}* — ستصل النتيجة هنا تلقائيًا.",
            parse_mode=ParseMode.MARKDOWN,
        )

    try:
        await q.wait_result(job)
    except Exception as e:
        logger.exception("telegram analysis failed")
        await update.message.reply_text(f"❌ خطأ أثناء التحليل: {e}")

    context.user_data.pop(SESSION_KEY, None)


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        await update.message.reply_text("غير مصرح.")
        return

    raw = await _download_photo_bytes(update, context)
    if raw is None:
        raw = await _download_document_bytes(update, context)
    if raw is None:
        await update.message.reply_text("أرسل صورة (photo) أو ملف png/jpg.")
        return

    s = _get_session(context)
    if s.get("reference") is None:
        s["reference"] = raw
        await update.message.reply_text("✅ تم استلام البصمة *الأصلية*.\nأرسل الآن البصمة *المقارنة*.", parse_mode=ParseMode.MARKDOWN)
        return

    if s.get("query") is None:
        s["query"] = raw
        await _run_analysis(update, context)
        return


def _telegram_request() -> HTTPXRequest:
    """SSL for httpx on Windows — default context uses OS store (works with urllib)."""
    raw = (os.getenv("TELEGRAM_SSL_VERIFY") or "").strip()
    if raw.lower() in ("0", "false", "no", "off"):
        verify: bool | ssl.SSLContext = False
    elif raw:
        verify = raw
    else:
        verify = ssl.create_default_context()
    return HTTPXRequest(httpx_kwargs={"verify": verify})


def _telegram_embedded() -> bool:
    return (os.getenv("TELEGRAM_EMBEDDED") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


async def _bot_send_markdown(chat_id: int, text: str) -> None:
    if _embedded_app is None:
        raise RuntimeError("embedded bot not started")
    await _embedded_app.bot.send_message(chat_id, text, parse_mode=ParseMode.MARKDOWN)


async def start_embedded_bot() -> Application:
    """Start polling inside the web server process (shared analysis queue)."""
    global _embedded_app
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing")

    from services.analysis_queue import start_analysis_queue

    await start_analysis_queue()
    q = get_analysis_queue()
    q.set_telegram_sender(_bot_send_markdown)

    application = build_application(token)
    await application.initialize()
    await application.start()
    await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    _embedded_app = application
    logger.info("Telegram bot polling (embedded in web server)")
    return application


async def stop_embedded_bot() -> None:
    global _embedded_app
    from services.analysis_queue import stop_analysis_queue

    if _embedded_app is not None:
        try:
            if _embedded_app.updater.running:
                await _embedded_app.updater.stop()
        except Exception as e:
            logger.warning("updater stop: %s", e)
        try:
            await _embedded_app.stop()
            await _embedded_app.shutdown()
        except Exception as e:
            logger.warning("application shutdown: %s", e)
        _embedded_app = None
    await stop_analysis_queue()


def build_application(
    token: str,
    *,
    post_init=None,
    post_shutdown=None,
) -> Application:
    # PTB uses a separate HTTP client for long-polling getUpdates — both need SSL config.
    request = _telegram_request()
    builder = (
        Application.builder()
        .token(token)
        .request(request)
        .get_updates_request(_telegram_request())
    )
    if post_init is not None:
        builder = builder.post_init(post_init)
    if post_shutdown is not None:
        builder = builder.post_shutdown(post_shutdown)
    app = builder.build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_image))
    return app


async def _post_init(application: Application) -> None:
    from services.analysis_queue import start_analysis_queue

    await start_analysis_queue()
    q = get_analysis_queue()

    async def send_md(chat_id: int, text: str) -> None:
        await application.bot.send_message(chat_id, text, parse_mode=ParseMode.MARKDOWN)

    q.set_telegram_sender(send_md)


async def _post_shutdown(application: Application) -> None:
    from services.analysis_queue import stop_analysis_queue

    await stop_analysis_queue()


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    if _telegram_embedded():
        raise SystemExit(
            "TELEGRAM_EMBEDDED=1: run the web server (python run.py / run_dev.ps1) "
            "instead of python -m bot alone."
        )
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise SystemExit(
            "Set TELEGRAM_BOT_TOKEN in .env (get token from @BotFather on Telegram)."
        )
    allowed = _allowed_chat_ids()
    if allowed:
        logger.info("Telegram allowlist: %s", allowed)
    else:
        logger.warning("TELEGRAM_ALLOWED_CHAT_IDS not set — bot accepts any user.")

    application = build_application(
        token,
        post_init=_post_init,
        post_shutdown=_post_shutdown,
    )
    logger.info("Telegram bot polling (standalone process)…")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
