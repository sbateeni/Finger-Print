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
from pathlib import Path
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

from services.pair_analysis import analyze_fingerprint_pair, format_match_summary_ar
from bot.fingerprint_store import (
    extract_minutiae_from_bytes,
    list_templates,
    register_template,
    search_best_match,
)
from utils.telegram_process import claim_telegram_bot_pid, release_telegram_bot_pid

logger = logging.getLogger(__name__)

SESSION_KEY = "fp_session"
REGISTER_PENDING_KEY = "fp_register_pending"
SWEEP_MODE_KEY = "fp_sweep_mode"


def _telegram_auto_sweep_enabled() -> bool:
    return (os.getenv("TELEGRAM_AUTO_SWEEP") or "1").strip().lower() in ("1", "true", "yes", "on")


def _get_sweep_mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    mode = (context.user_data.get(SWEEP_MODE_KEY) or os.getenv("TELEGRAM_SWEEP_MODE") or "quick").strip().lower()
    return "wide" if mode == "wide" else "quick"


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
        "/status — ما تم استلامه\n"
        "/register — تسجيل بصمتك في قاعدة البوت\n"
        "/match — مقارنة صورة مع المسجّلين\n"
        "/templates — عرض المسجّلين\n"
        "/sweep — وضع بحث واسع (wide) للتكبير التلقائي\n"
        "/sweep_quick — وضع بحث سريع (افتراضي)\n\n"
        "عند إرسال صورتين يُطبَّق *Auto-sweep* تلقائيًا لتحسين النتيجة.\n\n"
        "يمكنك أيضًا استخدام الواجهة على http://127.0.0.1:8000",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_sweep_wide(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return
    context.user_data[SWEEP_MODE_KEY] = "wide"
    await update.message.reply_text(
        "✅ وضع Auto-sweep: *wide* (بحث أوسع — أبطأ قليلًا).\nأرسل الصورتين للتحليل.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_sweep_quick(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return
    context.user_data[SWEEP_MODE_KEY] = "quick"
    await update.message.reply_text(
        "✅ وضع Auto-sweep: *quick* (افتراضي).\nأرسل الصورتين للتحليل.",
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

    await update.message.reply_text("⏳ جاري التحليل… قد يستغرق 1–3 دقائق.")
    if _telegram_auto_sweep_enabled():
        mode = _get_sweep_mode(context)
        await update.message.reply_text(
            f"🔍 Auto-sweep ({mode}) — البحث عن أفضل تكبير/موضع للبصمة الجزئية…",
            parse_mode=ParseMode.MARKDOWN,
        )

    try:
        result = await asyncio.to_thread(
            analyze_fingerprint_pair,
            ref_b,
            qry_b,
            operator_name=f"telegram:{operator}",
            case_reference=f"TG-{chat_id}",
            write_report_and_audit=True,
            auto_sweep=_telegram_auto_sweep_enabled(),
            sweep_mode=_get_sweep_mode(context),
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


async def cmd_templates(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return
    rows = list_templates(context.application.bot_data)
    if not rows:
        await update.message.reply_text("لا يوجد مسجّلون بعد. استخدم /register")
        return
    lines = ["📋 *قوالب مسجّلة:*", ""]
    for uid, label, n in rows:
        lines.append(f"• `{uid}` — {label} ({n} نقطة)")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_register(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return
    context.user_data[REGISTER_PENDING_KEY] = True
    await update.message.reply_text(
        "📥 *تسجيل بصمة*\nأرسل صورة بصمة واحدة (photo أو ملف) لتُخزَّن للمقارنة لاحقًا.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_match_db(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return
    context.user_data["fp_match_pending"] = True
    await update.message.reply_text(
        "🔍 *مقارنة مع القاعدة*\nأرسل صورة بصمة للبحث عن أقرب تطابق بين المسجّلين.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def _handle_register_image(
    update: Update, context: ContextTypes.DEFAULT_TYPE, raw: bytes
) -> bool:
    """Process register flow. Returns True if consumed."""
    if not context.user_data.get(REGISTER_PENDING_KEY):
        return False
    context.user_data.pop(REGISTER_PENDING_KEY, None)
    uid = update.effective_user.id if update.effective_user else 0
    await update.message.reply_text("⏳ جاري استخراج النقاط الدقيقة…")

    def _work():
        return extract_minutiae_from_bytes(raw)

    mins, err = await asyncio.to_thread(_work)
    if err or not mins:
        await update.message.reply_text(f"❌ {err or 'فشل التسجيل'}")
        return True

    register_template(context.application.bot_data, uid, mins)
    await update.message.reply_text(
        f"✅ تم تسجيل بصمتك ({len(mins)} نقطة دقيقة).\n"
        f"المعرّف: `{uid}`\nاستخدم /match للمقارنة.",
        parse_mode=ParseMode.MARKDOWN,
    )
    return True


async def _handle_match_db_image(
    update: Update, context: ContextTypes.DEFAULT_TYPE, raw: bytes
) -> bool:
    if not context.user_data.pop("fp_match_pending", None):
        return False
    uid = update.effective_user.id if update.effective_user else 0
    await update.message.reply_text("⏳ جاري البحث في القاعدة…")

    bot_data = context.application.bot_data

    def _work():
        mins, err = extract_minutiae_from_bytes(raw)
        if err or not mins:
            return None, err, 0.0, False
        best_uid, score, matched = search_best_match(bot_data, mins, exclude_user_id=None)
        return best_uid, None, score, matched

    best_uid, err, score, matched = await asyncio.to_thread(_work)
    if err:
        await update.message.reply_text(f"❌ {err}")
        return True
    if best_uid is None:
        await update.message.reply_text("❌ لا يوجد مسجّلون. استخدم /register أولًا.")
        return True
    if matched:
        await update.message.reply_text(
            f"✅ *تطابق* مع المستخدم `{best_uid}`\nدرجة Bozorth: *{score:.1f}*",
            parse_mode=ParseMode.MARKDOWN,
        )
    else:
        await update.message.reply_text(
            f"❌ *لا تطابق كافٍ*\nأعلى درجة: `{score:.1f}` (مع `{best_uid}`)",
            parse_mode=ParseMode.MARKDOWN,
        )
    return True


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

    if await _handle_register_image(update, context, raw):
        return
    if await _handle_match_db_image(update, context, raw):
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


def build_application(token: str) -> Application:
    # PTB uses a separate HTTP client for long-polling getUpdates — both need SSL config.
    request = _telegram_request()
    app = (
        Application.builder()
        .token(token)
        .request(request)
        .get_updates_request(_telegram_request())
        .build()
    )

    async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        err = context.error
        if isinstance(err, Conflict):
            logger.error(
                "409 Conflict: another bot instance is polling (Kali, old terminal, or run_telegram.py). "
                "Stop other instances, then restart run_dev.ps1."
            )
            release_telegram_bot_pid()
            raise SystemExit(3) from err

    app.add_error_handler(on_error)
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("register", cmd_register))
    app.add_handler(CommandHandler("match", cmd_match_db))
    app.add_handler(CommandHandler("templates", cmd_templates))
    app.add_handler(CommandHandler("sweep", cmd_sweep_wide))
    app.add_handler(CommandHandler("sweep_quick", cmd_sweep_quick))
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
    claim_telegram_bot_pid()
    logger.info("Telegram bot polling…")
    try:
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )
    finally:
        release_telegram_bot_pid()


if __name__ == "__main__":
    main()
