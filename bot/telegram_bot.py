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
from telegram.error import Conflict
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

WELCOME_TEXT = (
    "مرحبًا — بوت مطابقة البصمات\n\n"
    "1) أرسل صورة البصمة الأصلية (مرجعية)\n"
    "2) ثم أرسل صورة البصمة المقارنة (جزئية)\n\n"
    "أو أرسل صورتين متتاليتين.\n"
    "الأوامر: /start  /reset  /status\n\n"
    "الواجهة: http://127.0.0.1:8000\n\n"
    "التحليل عميق على الكمبيوتر (محاذاة تلقائية + تقرير PDF).\n"
    "عند ازدحام الطلبات يُخبرك برقم دورك في الانتظار."
)


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


async def _reply_safe(update: Update, text: str, *, markdown: bool = False) -> None:
    if not update.message:
        return
    try:
        if markdown:
            await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(text)
    except Exception as e:
        logger.warning("reply failed (%s), retry plain: %s", type(e).__name__, e)
        await update.message.reply_text(text)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    if context.application.bot_data.get("polling_ready") is False:
        await _reply_safe(
            update,
            "البوت لم يكتمل تشغيله (تعارض 409). على Kali: bash scripts/stop_telegram_bot.sh ثم ./run_dev.sh\n"
            "أوقف التشغيل على Windows إن كان يعمل بنفس التوكن.",
        )
        return
    if not _is_allowed(update):
        await _reply_safe(update, "غير مصرح لك باستخدام هذا البوت.")
        return
    context.user_data.pop(SESSION_KEY, None)
    await _reply_safe(update, WELCOME_TEXT)


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


async def start_embedded_bot() -> Optional[Application]:
    """Start polling inside the web server process (shared analysis queue)."""
    global _embedded_app
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing")

    from pathlib import Path

    from services.analysis_queue import start_analysis_queue
    from utils.telegram_polling import (
        acquire_polling_lock,
        kill_stale_local_bot_processes,
        prepare_bot_session,
    )

    killed = kill_stale_local_bot_processes(Path(__file__).resolve().parent.parent)
    if killed:
        logger.info("Stopped %s stale local bot/web worker(s) before polling", killed)
        await asyncio.sleep(1.0)

    if not acquire_polling_lock():
        logger.error(
            "Telegram bot NOT started — another poller is active. "
            "Run: .\\scripts\\stop_telegram_bot.ps1 then restart."
        )
        await start_analysis_queue()
        return None

    await start_analysis_queue()
    q = get_analysis_queue()
    q.set_telegram_sender(_bot_send_markdown)

    application = build_application(token)
    await application.initialize()
    await application.start()
    await prepare_bot_session(application.bot)
    try:
        await application.updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )
    except Conflict as e:
        from utils.telegram_polling import release_polling_lock

        release_polling_lock()
        logger.error(
            "Telegram 409 Conflict — stop other bots (Kali/other PC): %s", e
        )
        await application.stop()
        await application.shutdown()
        return None

    _embedded_app = application
    logger.info("Telegram bot polling (embedded in web server)")
    return application


async def stop_embedded_bot() -> None:
    global _embedded_app
    from services.analysis_queue import stop_analysis_queue
    from utils.telegram_polling import release_polling_lock

    if _embedded_app is not None:
        try:
            await _embedded_app.updater.stop()
        except Exception as e:
            logger.warning("updater stop: %s", e)
        try:
            await _embedded_app.stop()
            await _embedded_app.shutdown()
        except Exception as e:
            logger.warning("application shutdown: %s", e)
        _embedded_app = None
    release_polling_lock()
    await stop_analysis_queue()


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err = context.error
    if isinstance(err, Conflict):
        logger.error(
            "Telegram Conflict (two pollers). Run scripts/stop_telegram_bot.ps1 — %s",
            err,
        )
        return
    logger.exception("Telegram handler error: %s", err)


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
    app.add_error_handler(_on_error)
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(
        MessageHandler(filters.Regex(r"(?i)^(?:/start|start|بداية|البدء|ستارت)$"), cmd_start)
    )
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_image))
    return app


async def _post_init(application: Application) -> None:
    from pathlib import Path

    from services.analysis_queue import start_analysis_queue
    from utils.telegram_polling import (
        acquire_polling_lock,
        kill_stale_local_bot_processes,
        prepare_bot_session,
    )

    root = Path(__file__).resolve().parent.parent
    killed = kill_stale_local_bot_processes(root)
    if killed:
        logger.info("Stopped %s stale worker(s) before standalone polling", killed)
        await asyncio.sleep(1.0)

    if not acquire_polling_lock():
        application.bot_data["polling_ready"] = False
        raise RuntimeError(
            "Telegram polling lock busy — stop other PC/process "
            "(bash scripts/stop_telegram_bot.sh on Kali, stop_telegram_bot.ps1 on Windows)"
        )

    await prepare_bot_session(application.bot)
    await start_analysis_queue()
    q = get_analysis_queue()

    async def send_md(chat_id: int, text: str) -> None:
        await application.bot.send_message(chat_id, text, parse_mode=ParseMode.MARKDOWN)

    q.set_telegram_sender(send_md)
    application.bot_data["polling_ready"] = True


async def _post_shutdown(application: Application) -> None:
    from services.analysis_queue import stop_analysis_queue
    from utils.telegram_polling import release_polling_lock

    release_polling_lock()
    await stop_analysis_queue()


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    standalone = (os.getenv("FP_TELEGRAM_STANDALONE") or "").strip() in ("1", "true", "yes")
    if _telegram_embedded() and not standalone:
        raise SystemExit(
            "TELEGRAM_EMBEDDED=1: run ./run_dev.sh or run_dev.ps1 (not python -m bot alone). "
            "Or: TELEGRAM_EMBEDDED=0 / FP_TELEGRAM_STANDALONE=1"
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
