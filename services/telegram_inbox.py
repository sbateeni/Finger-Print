"""
Save fingerprint images received via Telegram for local use (web, scripts, IDE).

Layout:
  output/telegram_inbox/{chat_id}/latest/reference.jpg
  output/telegram_inbox/{chat_id}/latest/query.jpg
  output/telegram_inbox/{chat_id}/latest/manifest.json
  output/telegram_inbox/{chat_id}/batches/batch_YYYYMMDD_HHMMSS/...
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from config import OUTPUT_DIR

logger = logging.getLogger(__name__)

Role = Literal["reference", "query"]

INBOX_ROOT = Path(OUTPUT_DIR) / "telegram_inbox"
INBOX_ROOT.mkdir(parents=True, exist_ok=True)

_REF_HINTS = re.compile(
    r"(?i)(?:^|\b)(ref|reference|original|orig|مرجع|أصل|المرجع|الاصل|الأصل)(?:\b|$)"
)
_QRY_HINTS = re.compile(
    r"(?i)(?:^|\b)(qry|query|partial|probe|مقار|جزء|الجزء|المقارنة|مقارنة)(?:\b|$)"
)


def guess_image_suffix(raw: bytes) -> str:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if raw[:2] == b"\xff\xd8":
        return ".jpg"
    if raw[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tif"
    if raw[:4] == b"RIFF" and len(raw) > 12 and raw[8:12] == b"WEBP":
        return ".webp"
    if raw[:2] == b"BM":
        return ".bmp"
    return ".jpg"


def role_from_caption(caption: str | None) -> Role | None:
    text = (caption or "").strip()
    if not text:
        return None
    if _REF_HINTS.search(text):
        return "reference"
    if _QRY_HINTS.search(text):
        return "query"
    return None


def _chat_root(chat_id: int) -> Path:
    return INBOX_ROOT / str(chat_id)


def _latest_dir(chat_id: int) -> Path:
    return _chat_root(chat_id) / "latest"


def _manifest_path(chat_id: int) -> Path:
    return _latest_dir(chat_id) / "manifest.json"


def _load_manifest(chat_id: int) -> dict[str, Any]:
    path = _manifest_path(chat_id)
    if not path.is_file():
        return {"chat_id": chat_id, "reference": None, "query": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("inbox manifest read failed %s: %s", chat_id, exc)
    return {"chat_id": chat_id, "reference": None, "query": None}


def _write_manifest(chat_id: int, manifest: dict[str, Any]) -> None:
    latest = _latest_dir(chat_id)
    latest.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    _manifest_path(chat_id).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _archive_batch(chat_id: int, manifest: dict[str, Any]) -> None:
    ref = manifest.get("reference")
    qry = manifest.get("query")
    if not ref or not qry:
        return
    stamp = datetime.now(timezone.utc).strftime("batch_%Y%m%d_%H%M%S")
    batch_dir = _chat_root(chat_id) / "batches" / stamp
    batch_dir.mkdir(parents=True, exist_ok=True)
    for entry, role in ((ref, "reference"), (qry, "query")):
        src = Path(entry["path"])
        if src.is_file():
            dest = batch_dir / f"{role}{src.suffix}"
            dest.write_bytes(src.read_bytes())
    meta = dict(manifest)
    meta["batch_id"] = stamp
    (batch_dir / "manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_inbox_image(
    chat_id: int,
    user_id: int | None,
    role: Role,
    raw: bytes,
    *,
    original_name: str | None = None,
) -> dict[str, Any]:
    """Persist one image under latest/ and update manifest."""
    if not raw:
        raise ValueError("empty image")

    suffix = guess_image_suffix(raw)
    latest = _latest_dir(chat_id)
    latest.mkdir(parents=True, exist_ok=True)

    filename = f"{role}{suffix}"
    dest = latest / filename
    dest.write_bytes(raw)

    sha = hashlib.sha256(raw).hexdigest()
    out_root = Path(OUTPUT_DIR).resolve()
    try:
        rel = str(dest.resolve().relative_to(out_root)).replace("\\", "/")
    except ValueError:
        rel = str(dest.resolve()).replace("\\", "/")
    entry = {
        "role": role,
        "path": str(dest.resolve()),
        "relative_path": rel,
        "filename": filename,
        "original_name": original_name,
        "sha256": sha,
        "size_bytes": len(raw),
    }

    manifest = _load_manifest(chat_id)
    manifest["chat_id"] = chat_id
    manifest["user_id"] = user_id
    manifest[role] = entry
    _write_manifest(chat_id, manifest)

    if manifest.get("reference") and manifest.get("query"):
        _archive_batch(chat_id, manifest)

    return entry


def get_inbox_status(chat_id: int) -> dict[str, Any]:
    manifest = _load_manifest(chat_id)
    ref = manifest.get("reference")
    qry = manifest.get("query")
    return {
        "chat_id": chat_id,
        "has_reference": bool(ref and Path(ref["path"]).is_file()),
        "has_query": bool(qry and Path(qry["path"]).is_file()),
        "reference_path": ref["path"] if ref else None,
        "query_path": qry["path"] if qry else None,
        "reference_relative": ref.get("relative_path") if ref else None,
        "query_relative": qry.get("relative_path") if qry else None,
        "manifest_path": str(_manifest_path(chat_id).resolve()),
        "updated_at": manifest.get("updated_at"),
    }


def load_pair_bytes(chat_id: int) -> tuple[bytes | None, bytes | None]:
    status = get_inbox_status(chat_id)
    ref_b: bytes | None = None
    qry_b: bytes | None = None
    if status["reference_path"]:
        p = Path(status["reference_path"])
        if p.is_file():
            ref_b = p.read_bytes()
    if status["query_path"]:
        p = Path(status["query_path"])
        if p.is_file():
            qry_b = p.read_bytes()
    return ref_b, qry_b


def format_paths_message(chat_id: int) -> str:
    st = get_inbox_status(chat_id)
    lines = ["📁 *صندوق رفع تيليجرام*", ""]
    if st["reference_path"]:
        lines.append(f"• المرجعية:\n`{st['reference_path']}`")
    else:
        lines.append("• المرجعية: ❌")
    if st["query_path"]:
        lines.append(f"• المقارنة:\n`{st['query_path']}`")
    else:
        lines.append("• المقارنة: ❌")
    lines.append("")
    lines.append(f"manifest:\n`{st['manifest_path']}`")
    if st["has_reference"] and st["has_query"]:
        lines.append("")
        lines.append("مثال من الطرفية:")
        lines.append(
            f"`python scripts/test_pair_local.py \"{st['reference_path']}\" \"{st['query_path']}\"`"
        )
    return "\n".join(lines)


def infer_next_role(chat_id: int, *, forced: Role | None = None) -> Role:
    if forced:
        return forced
    st = get_inbox_status(chat_id)
    if not st["has_reference"]:
        return "reference"
    return "query"
