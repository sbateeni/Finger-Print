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
from pathlib import Path
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

from services.pair_analysis import analyze_fingerprint_pair, format_match_summary_ar

logger = logging.getLogger(__name__)

SESSION_KEY = "fp_session"


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
        "يمكنك أيضًا استخدام الواجهة على http://127.0.0.1:8000",
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

    await update.message.reply_text("⏳ جاري التحليل… قد يستغرق دقيقة.")

    try:
        result = await asyncio.to_thread(
            analyze_fingerprint_pair,
            ref_b,
            qry_b,
            operator_name=f"telegram:{operator}",
            case_reference=f"TG-{chat_id}",
            write_report_and_audit=True,
        )
    except Exception as e:
        logger.exception("telegram analysis failed")
        await update.message.reply_text(f"❌ خطأ أثناء التحليل: {e}")
        context.user_data.pop(SESSION_KEY, None)
        return

    summary = format_match_summary_ar(result)
    await update.message.reply_text(summary, parse_mode=ParseMode.MARKDOWN)

    pdf_path = result.get("report_pdf")
    html_path = result.get("report_html")
    if pdf_path and Path(pdf_path).is_file():
        with open(pdf_path, "rb") as f:
            await update.message.reply_document(
                document=f,
                filename=Path(pdf_path).name,
                caption="تقرير PDF",
            )
    elif html_path and Path(html_path).is_file():
        with open(html_path, "rb") as f:
            await update.message.reply_document(
                document=f,
                filename=Path(html_path).name,
                caption="تقرير HTML",
            )

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


def build_application(token: str) -> Application:
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_image))
    return app


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
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

    application = build_application(token)
    logger.info("Telegram bot polling…")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
