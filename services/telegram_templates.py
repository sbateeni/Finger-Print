"""
In-memory + JSON persistence for Telegram /register and /match (1:N probe).
Uses the same forensic branch pipeline as the web UI (not raw pyfing-only).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_BORDER_MARGIN,
    DEFAULT_MIN_ANGLE_DIFF,
    DEFAULT_MIN_CONTRAST,
    DEFAULT_MIN_DISTANCE,
    OUTPUT_DIR,
)
from matching.compare_engine import FingerprintMatcher
from preprocessing.quality import QualityChecker
from services.analysis_service import _process_branch
from utils.image_utils import _decode_upload_type
from utils.quality_gate import quality_gate_enabled, quality_min_score

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(OUTPUT_DIR) / "telegram_templates"
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)


def _template_path(user_id: int) -> Path:
    return TEMPLATE_DIR / f"{user_id}.json"


def _default_branch_kwargs() -> dict[str, Any]:
    return {
        "denoise_method": "fastNlMeans",
        "fast_denoise_h": 10,
        "gauss_ksize": 5,
        "border_margin": DEFAULT_BORDER_MARGIN,
        "min_distance": DEFAULT_MIN_DISTANCE,
        "min_contrast": DEFAULT_MIN_CONTRAST,
        "min_angle_diff": DEFAULT_MIN_ANGLE_DIFF,
    }


def extract_branch_from_bytes(image_bytes: bytes) -> dict[str, Any]:
    gray = _decode_upload_type(image_bytes)
    branch = _process_branch(gray, **_default_branch_kwargs())
    if branch.get("error"):
        return {"error": branch["error"]}
    sk = branch.get("skeleton")
    if sk is None:
        return {"error": "فشل استخراج الهيكل"}
    h, w = sk.shape[:2]
    branch["image_shape"] = [int(h), int(w)]
    return branch


def register_template(user_id: int, image_bytes: bytes) -> dict[str, Any]:
    if quality_gate_enabled():
        gray = _decode_upload_type(image_bytes)
        ok, score, method = QualityChecker.is_acceptable(gray, threshold=quality_min_score())
        if not ok:
            return {
                "ok": False,
                "message": f"❌ جودة الصورة منخفضة ({score:.0f}/100، {method}).",
            }

    branch = extract_branch_from_bytes(image_bytes)
    if branch.get("error"):
        return {"ok": False, "message": f"❌ {branch['error']}"}

    minutiae = branch.get("minutiae") or []
    if len(minutiae) < 10:
        return {
            "ok": False,
            "message": f"❌ نقاط دقيقة قليلة ({len(minutiae)}). أعد التقاط صورة أوضح.",
        }

    payload = {
        "user_id": user_id,
        "minutiae": minutiae,
        "cores": branch.get("cores") or [],
        "image_shape": branch.get("image_shape"),
        "minutiae_count": len(minutiae),
        "extraction": branch.get("minutiae_extraction"),
    }
    _template_path(user_id).write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "ok": True,
        "message": (
            f"✅ تم تسجيل بصمتك ({len(minutiae)} نقطة، "
            f"{branch.get('minutiae_extraction', 'pipeline')})."
        ),
        "count": len(minutiae),
    }


def load_template(user_id: int) -> dict[str, Any] | None:
    path = _template_path(user_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("template load failed %s: %s", user_id, exc)
        return None


def list_registered_user_ids() -> list[int]:
    ids: list[int] = []
    for p in TEMPLATE_DIR.glob("*.json"):
        try:
            ids.append(int(p.stem))
        except ValueError:
            continue
    return ids


def match_against_templates(
    query_bytes: bytes,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    if quality_gate_enabled():
        gray = _decode_upload_type(query_bytes)
        ok, score, method = QualityChecker.is_acceptable(gray, threshold=quality_min_score())
        if not ok:
            return {
                "ok": False,
                "message": f"❌ جودة الصورة منخفضة ({score:.0f}/100، {method}).",
            }

    q_branch = extract_branch_from_bytes(query_bytes)
    if q_branch.get("error"):
        return {"ok": False, "message": f"❌ {q_branch['error']}"}

    q_min = q_branch.get("minutiae") or []
    if len(q_min) < 5:
        return {"ok": False, "message": f"❌ نقاط قليلة على البصمة ({len(q_min)})."}

    user_ids = list_registered_user_ids()
    if not user_ids:
        return {"ok": False, "message": "❌ لا يوجد مستخدمون مسجلون. استخدم /register أولاً."}

    engine = FingerprintMatcher(threshold=threshold)
    shape = tuple(q_branch.get("image_shape") or [500, 500])

    best_uid: int | None = None
    best_score = 0.0
    best_match = False

    for uid in user_ids:
        tpl = load_template(uid)
        if not tpl or not tpl.get("minutiae"):
            continue
        t_shape = tuple(tpl.get("image_shape") or shape)
        img_shape = (max(shape[0], t_shape[0]), max(shape[1], t_shape[1]))
        score, is_match, _ = engine.compare_fingerprints(
            tpl["minutiae"],
            q_min,
            img_shape,
            cores_ref=tpl.get("cores"),
            cores_qry=q_branch.get("cores"),
        )
        if score > best_score:
            best_score = score
            best_uid = uid
            best_match = is_match

    if best_uid is None:
        return {"ok": False, "message": "❌ لا توجد قوالب صالحة للمقارنة."}

    if best_score >= 45 or best_match:
        verdict = f"✅ تطابق قوي مع المستخدم {best_uid} — {best_score:.1f}%"
    elif best_score >= 30:
        verdict = f"⚠️ تشابه غير حاسم مع {best_uid} — {best_score:.1f}% (مراجعة خبير)"
    else:
        verdict = f"❌ لا تطابق كافٍ. أعلى درجة {best_score:.1f}% (مستخدم {best_uid})"

    return {
        "ok": True,
        "message": verdict,
        "best_user_id": best_uid,
        "best_score": round(best_score, 2),
        "is_match": best_match,
    }
