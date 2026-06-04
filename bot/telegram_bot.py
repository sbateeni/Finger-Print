"""
Telegram bot: upload fingerprint photos to disk + optional deep analysis.

Env:
  TELEGRAM_BOT_TOKEN          — required
  TELEGRAM_ALLOWED_CHAT_IDS   — optional comma-separated user/chat IDs (recommended)
  TELEGRAM_AUTO_ANALYZE       — 1 = run analysis after 2nd photo (default 0 = upload only)
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import ssl
from typing import Optional, Set

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.constants import ParseMode
from telegram.error import Conflict
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)
from telegram.request import HTTPXRequest

from services.analysis_queue import get_analysis_queue
from services.telegram_inbox import (
    format_paths_message,
    get_inbox_status,
    infer_next_role,
    load_pair_bytes,
    role_from_caption,
    save_inbox_image,
)
from utils.runtime_platform import is_linux, telegram_stop_script_hint

logger = logging.getLogger(__name__)

SESSION_KEY = "fp_session"
NEXT_ROLE_KEY = "fp_next_role"
_embedded_app: Optional[Application] = None

WELCOME_TEXT = (
    "مرحبًا — بوت رفع ومطابقة البصمات\n\n"
    "📤 *رفع للكود (افتراضي):*\n"
    "1) أرسل البصمة *المرجعية* (أو /ref ثم صورة)\n"
    "2) ثم البصمة *المقارنة* (أو /qry ثم صورة)\n"
    "تُحفظ الصور على القرص — تابع من الواجهة أو السكربتات.\n\n"
    "أو أضف تعليقًا: `ref` / `qry` / `مرجع` / `مقارنة`\n\n"
    "أوامر:\n"
    "/paths — مسارات الملفات المحفوظة\n"
    "/analyze — تشغيل التحليل العميق من البوت\n"
    "/reset  /status  /ref  /qry\n"
    "/register  /match — قوالب سريعة\n\n"
    "الواجهة: http://127.0.0.1:8000\n"
    "مثال: `python scripts/telegram_inbox_latest.py`"
)


def _telegram_auto_analyze() -> bool:
    return (os.getenv("TELEGRAM_AUTO_ANALYZE") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
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
            f"البوت لم يكتمل تشغيله (تعارض 409).\n"
            f"على Linux: {telegram_stop_script_hint()} ثم ./run_dev.sh\n"
            "أوقف أي تشغيل آخر بنفس TELEGRAM_BOT_TOKEN.",
        )
        return
    if not _is_allowed(update):
        await _reply_safe(update, "غير مصرح لك باستخدام هذا البوت.")
        return
    context.user_data.pop(SESSION_KEY, None)
    
    keyboard = [
        [
            InlineKeyboardButton("ضابط ميداني 👮‍♂️", callback_data='role_field'),
            InlineKeyboardButton("ضابط مختبر 🔬", callback_data='role_lab')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await _reply_safe(
        update, 
        "مرحباً بك في نظام مطابقة البصمات الجنائي.\n\nيرجى تحديد صفتك الوظيفية للبدء:",
        markdown=False
    )
    if update.message:
        await update.message.reply_text("اختر من القائمة أدناه:", reply_markup=reply_markup)

async def handle_role_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    
    role = query.data
    context.user_data['user_role'] = role
    
    if role == 'role_field':
        msg = (
            "✅ *تم تحديد الصلاحية: ضابط ميداني*\n\n"
            "صلاحياتك تتيح لك رفع **البصمة المجهولة (بصمة مسرح الجريمة)** فقط لمطابقتها مع قاعدة بيانات المختبر.\n\n"
            "الرجاء إرسال صورة البصمة المرفوعة من مسرح الجريمة الآن."
        )
    else:
        msg = (
            "✅ *تم تحديد الصلاحية: ضابط مختبر*\n\n"
            "صلاحياتك كاملة. يمكنك رفع البصمة المرجعية، وبصمة مسرح الجريمة، وإجراء عمليات التحليل.\n\n"
            "الرجاء إرسال البصمة المرجعية أولاً (أو استخدم /ref)، ثم بصمة المقارنة (أو استخدم /qry)."
        )
        
    await query.edit_message_text(text=msg, parse_mode=ParseMode.MARKDOWN)


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return
    context.user_data.pop(SESSION_KEY, None)
    await update.message.reply_text("تم إعادة التعيين. أرسل البصمة الأصلية.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return
    chat_id = update.effective_chat.id if update.effective_chat else 0
    st = get_inbox_status(chat_id)
    ref = "✅" if st["has_reference"] else "❌"
    qry = "✅" if st["has_query"] else "❌"
    lines = [f"المرجعية: {ref}", f"المقارنة: {qry}"]
    if st["reference_path"]:
        lines.append(f"ref: {st['reference_path']}")
    if st["query_path"]:
        lines.append(f"qry: {st['query_path']}")
    await update.message.reply_text("\n".join(lines))


async def cmd_paths(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return
    chat_id = update.effective_chat.id if update.effective_chat else 0
    await update.message.reply_text(
        format_paths_message(chat_id),
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_ref(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return
    if context.user_data.get('user_role') == 'role_field':
        await update.message.reply_text("عذراً، رفع البصمة المرجعية مخصص لضباط المختبر فقط.")
        return
    context.user_data[NEXT_ROLE_KEY] = "reference"
    await update.message.reply_text("📌 الصورة التالية = *المرجعية*.", parse_mode=ParseMode.MARKDOWN)


async def cmd_qry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return
    context.user_data[NEXT_ROLE_KEY] = "query"
    await update.message.reply_text("📌 الصورة التالية = *المقارنة*.", parse_mode=ParseMode.MARKDOWN)


async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return
    chat_id = update.effective_chat.id if update.effective_chat else 0
    ref_b, qry_b = load_pair_bytes(chat_id)
    if not ref_b or not qry_b:
        await update.message.reply_text(
            "❌ الصورتان غير مكتملتين على القرص.\nأرسل المرجعية ثم المقارنة، أو /paths للتحقق."
        )
        return
    s = _get_session(context)
    s["reference"] = ref_b
    s["query"] = qry_b
    await _run_analysis(update, context)


TEMPLATE_MODE_KEY = "fp_template_mode"


async def cmd_register(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return
    context.user_data[TEMPLATE_MODE_KEY] = "register"
    context.user_data.pop(SESSION_KEY, None)
    await update.message.reply_text(
        "📌 وضع التسجيل: أرسل *صورة بصمة واحدة* (واضحة) لتخزين قالبك.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_match(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return
    context.user_data[TEMPLATE_MODE_KEY] = "match"
    context.user_data.pop(SESSION_KEY, None)
    await update.message.reply_text(
        "🔍 وضع المطابقة: أرسل *صورة بصمة* للمقارنة مع القوالب المسجلة.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def _handle_template_mode(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    raw: bytes,
    mode: str,
) -> bool:
    """Returns True if handled (register/match)."""
    from services.telegram_templates import match_against_templates, register_template

    user = update.effective_user
    if not user:
        return False

    context.user_data.pop(TEMPLATE_MODE_KEY, None)
    loop = asyncio.get_event_loop()

    if mode == "register":
        await update.message.reply_text("⏳ جاري استخراج النقاط وتسجيل القالب…")
        result = await loop.run_in_executor(
            None, lambda: register_template(user.id, raw)
        )
    else:
        await update.message.reply_text("⏳ جاري المطابقة مع القوالب المسجلة…")
        result = await loop.run_in_executor(
            None, lambda: match_against_templates(raw)
        )

    await update.message.reply_text(result.get("message", "تم."))
    return True


async def _run_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id if update.effective_chat else 0
    s = _get_session(context)
    ref_b: bytes = s.get("reference")
    qry_b: bytes = s.get("query")
    if not ref_b or not qry_b:
        ref_b, qry_b = load_pair_bytes(chat_id)
    if not ref_b or not qry_b:
        await update.message.reply_text("لم تكتمل الصورتان بعد.")
        return
    s["reference"] = ref_b
    s["query"] = qry_b
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

    user_role = context.user_data.get('user_role')
    if not user_role:
        await update.message.reply_text("الرجاء اختيار صفتك الوظيفية أولاً عبر الأمر /start")
        return

    template_mode = context.user_data.get(TEMPLATE_MODE_KEY)
    raw = await _download_photo_bytes(update, context)
    original_name = None
    if raw is None:
        raw = await _download_document_bytes(update, context)
        if update.message and update.message.document:
            original_name = update.message.document.file_name
    if raw is None:
        await update.message.reply_text("أرسل صورة (photo) أو ملف png/jpg.")
        return

    if template_mode in ("register", "match"):
        await _handle_template_mode(update, context, raw, template_mode)
        return

    chat_id = update.effective_chat.id if update.effective_chat else 0
    user_id = update.effective_user.id if update.effective_user else None
    caption = update.message.caption if update.message else None

    forced = context.user_data.pop(NEXT_ROLE_KEY, None)
    role = role_from_caption(caption) or infer_next_role(chat_id, forced=forced)
    
    if user_role == 'role_field':
        role = "query"  # Field officer always uploads the crime scene query

    try:
        entry = save_inbox_image(
            chat_id,
            user_id,
            role,
            raw,
            original_name=original_name,
        )
    except Exception as e:
        logger.exception("telegram inbox save failed")
        await update.message.reply_text(f"❌ فشل حفظ الصورة: {e}")
        return

    s = _get_session(context)
    if role == "reference":
        s["reference"] = raw
    else:
        s["query"] = raw

    label = "المرجعية" if role == "reference" else "المقارنة"
    st = get_inbox_status(chat_id)
    lines = [
        f"✅ تم حفظ *{label}* على القرص.",
        f"`{entry['path']}`",
    ]
    if st["has_reference"] and st["has_query"]:
        lines.append("")
        lines.append("✅ الصورتان جاهزتان — تابع من الكود أو الواجهة.")
        lines.append("/paths — المسارات  |  /analyze — تحليل من البوت")
        if _telegram_auto_analyze():
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
            await _run_analysis(update, context)
            return
    else:
        other = "المقارنة" if role == "reference" else "المرجعية"
        lines.append(f"أرسل الآن صورة *{other}* (أو /qry / /ref).")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


def _telegram_timeouts() -> tuple[float, float, float, float]:
    """connect, read, write, pool — Kali/slow networks need longer timeouts."""
    def _f(name: str, default: float) -> float:
        raw = (os.getenv(name) or "").strip()
        try:
            return float(raw) if raw else default
        except ValueError:
            return default

    base = 60.0 if is_linux() else 30.0
    return (
        _f("TELEGRAM_CONNECT_TIMEOUT", base),
        _f("TELEGRAM_READ_TIMEOUT", base),
        _f("TELEGRAM_WRITE_TIMEOUT", base),
        _f("TELEGRAM_POOL_TIMEOUT", base),
    )


def _telegram_request() -> HTTPXRequest:
    """SSL: Linux uses system CAs; Windows may need TELEGRAM_SSL_VERIFY=0."""
    raw = (os.getenv("TELEGRAM_SSL_VERIFY") or "").strip()
    if raw.lower() in ("0", "false", "no", "off"):
        verify: bool | ssl.SSLContext = False
    elif raw:
        verify = raw
    else:
        verify = ssl.create_default_context()
    connect_t, read_t, write_t, pool_t = _telegram_timeouts()
    return HTTPXRequest(
        connect_timeout=connect_t,
        read_timeout=read_t,
        write_timeout=write_t,
        pool_timeout=pool_t,
        httpx_kwargs={"verify": verify},
    )


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

    killed = kill_stale_local_bot_processes(
        Path(__file__).resolve().parent.parent, bots_only=True
    )
    if killed:
        logger.info("Stopped %s stale bot process(es) before polling", killed)
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
    app.add_handler(CommandHandler("paths", cmd_paths))
    app.add_handler(CommandHandler("ref", cmd_ref))
    app.add_handler(CommandHandler("qry", cmd_qry))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("register", cmd_register))
    app.add_handler(CommandHandler("match", cmd_match))
    app.add_handler(CallbackQueryHandler(handle_role_selection, pattern='^role_'))
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
    killed = kill_stale_local_bot_processes(root, bots_only=True)
    if killed:
        logger.info("Stopped %s stale bot process(es) before standalone polling", killed)
        await asyncio.sleep(1.0)

    if not acquire_polling_lock():
        application.bot_data["polling_ready"] = False
        raise RuntimeError(
            f"Telegram polling lock busy — run: {telegram_stop_script_hint()}"
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
    plat = "Linux" if is_linux() else platform.system()
    logger.info("Telegram bot polling (standalone, %s)…", plat)
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
