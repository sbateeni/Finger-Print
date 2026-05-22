"""
Send analysis summaries to Telegram via Bot HTTP API (no polling required).
"""

from __future__ import annotations

import logging
import os
import ssl
from pathlib import Path
from typing import Any, Iterable, Optional, Set

import httpx

from services.pair_analysis import format_match_summary_ar

logger = logging.getLogger(__name__)


def _truthy(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def notify_on_web_enabled() -> bool:
    return _truthy("TELEGRAM_NOTIFY_ON_WEB", default=True)


def _ssl_verify() -> bool | ssl.SSLContext:
    raw = (os.getenv("TELEGRAM_SSL_VERIFY") or "").strip()
    if raw.lower() in ("0", "false", "no", "off"):
        return False
    if raw:
        return raw
    return ssl.create_default_context()


def notify_chat_ids() -> list[int]:
    raw = (os.getenv("TELEGRAM_NOTIFY_CHAT_IDS") or os.getenv("TELEGRAM_ALLOWED_CHAT_IDS") or "").strip()
    if not raw:
        return []
    out: list[int] = []
    seen: Set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit() or (part.startswith("-") and part[1:].isdigit()):
            cid = int(part)
            if cid not in seen:
                seen.add(cid)
                out.append(cid)
    return out


def _api_base(token: str) -> str:
    return f"https://api.telegram.org/bot{token}"


def _post_sync(token: str, method: str, data: dict | None = None, files: dict | None = None) -> bool:
    url = f"{_api_base(token)}/{method}"
    try:
        with httpx.Client(verify=_ssl_verify(), timeout=120.0) as client:
            r = client.post(url, data=data, files=files)
            if r.status_code >= 400:
                logger.warning("Telegram %s failed: %s %s", method, r.status_code, r.text[:300])
                return False
            body = r.json()
            if not body.get("ok"):
                logger.warning("Telegram %s not ok: %s", method, body)
                return False
            return True
    except Exception as e:
        logger.warning("Telegram %s error: %s", method, e)
        return False


def send_text_sync(chat_id: int, text: str, *, parse_mode: str = "Markdown") -> bool:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        return False
    data: dict[str, Any] = {"chat_id": chat_id, "text": text[:4090]}
    if parse_mode:
        data["parse_mode"] = parse_mode
    return _post_sync(token, "sendMessage", data=data)


def send_document_sync(
    chat_id: int,
    path: Path,
    *,
    caption: str = "",
    filename: Optional[str] = None,
) -> bool:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token or not path.is_file():
        return False
    fname = filename or path.name
    data: dict[str, Any] = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption[:1020]
    with path.open("rb") as f:
        files = {"document": (fname, f)}
        return _post_sync(token, "sendDocument", data=data, files=files)


def deliver_result_sync(
    chat_id: int,
    result: dict[str, Any],
    *,
    source_label: str = "",
) -> None:
    prefix = ""
    if source_label:
        prefix = f"📡 *مصدر الطلب:* {source_label}\n\n"
    summary = format_match_summary_ar(result)
    send_text_sync(chat_id, prefix + summary)

    pdf = result.get("report_pdf")
    html = result.get("report_html")
    if pdf and Path(pdf).is_file():
        send_document_sync(chat_id, Path(pdf), caption="تقرير PDF")
    elif html and Path(html).is_file():
        send_document_sync(chat_id, Path(html), caption="تقرير HTML")


def notify_result_to_all_chats(
    result: dict[str, Any],
    *,
    source_label: str = "الواجهة",
    chat_ids: Optional[Iterable[int]] = None,
) -> None:
    if not notify_on_web_enabled():
        return
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        return
    ids = list(chat_ids) if chat_ids is not None else notify_chat_ids()
    if not ids:
        logger.debug("TELEGRAM_NOTIFY: no chat ids configured")
        return
    for cid in ids:
        deliver_result_sync(cid, result, source_label=source_label)
