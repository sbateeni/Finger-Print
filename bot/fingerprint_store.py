"""
In-memory fingerprint template store for Telegram /register and /match.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    DEFAULT_BORDER_MARGIN,
    DEFAULT_MIN_ANGLE_DIFF,
    DEFAULT_MIN_CONTRAST,
    DEFAULT_MIN_DISTANCE,
)
from services.analysis_service import _process_branch
from matching.bozorth_matcher import BozorthMatcher
from config.config import BOZORTH_MATCH_THRESHOLD


def _templates_db(app_data: dict) -> Dict[int, dict]:
    if "fp_templates" not in app_data:
        app_data["fp_templates"] = {}
    return app_data["fp_templates"]


def extract_minutiae_from_bytes(
    image_bytes: bytes,
    *,
    border_margin: int = DEFAULT_BORDER_MARGIN,
    min_distance: int = DEFAULT_MIN_DISTANCE,
    min_contrast: int = DEFAULT_MIN_CONTRAST,
    min_angle_diff: int = DEFAULT_MIN_ANGLE_DIFF,
) -> Tuple[Optional[List[dict]], Optional[str]]:
    """Decode image and run preprocessing branch; return minutiae list or error."""
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, "تعذر فك ترميز الصورة."
        branch = _process_branch(
            img,
            denoise_method="fastNlMeans",
            fast_denoise_h=10,
            gauss_ksize=3,
            border_margin=border_margin,
            min_distance=min_distance,
            min_contrast=min_contrast,
            min_angle_diff=min_angle_diff,
        )
        if branch.get("error"):
            return None, str(branch["error"])
        mins = branch.get("minutiae") or []
        if len(mins) < 5:
            return None, f"نقاط دقيقة قليلة ({len(mins)}) — أرسل صورة أوضح."
        return mins, None
    except Exception as e:
        return None, str(e)


def register_template(
    app_data: dict,
    user_id: int,
    minutiae: List[dict],
    *,
    label: str = "",
) -> None:
    db = _templates_db(app_data)
    db[user_id] = {
        "minutiae": minutiae,
        "label": label or f"user_{user_id}",
        "count": len(minutiae),
    }


def list_templates(app_data: dict) -> List[Tuple[int, str, int]]:
    db = _templates_db(app_data)
    return [(uid, v.get("label", ""), int(v.get("count", 0))) for uid, v in db.items()]


def search_best_match(
    app_data: dict,
    query_minutiae: List[dict],
    *,
    exclude_user_id: Optional[int] = None,
) -> Tuple[Optional[int], float, bool]:
    """Return (best_user_id, raw_score, is_match)."""
    db = _templates_db(app_data)
    if not db:
        return None, 0.0, False

    matcher = BozorthMatcher(match_threshold=BOZORTH_MATCH_THRESHOLD)
    best_uid: Optional[int] = None
    best_score = 0.0
    best_match = False

    for uid, entry in db.items():
        if exclude_user_id is not None and uid == exclude_user_id:
            continue
        template = entry.get("minutiae") or []
        if not template:
            continue
        score, matched = matcher.compare_fingerprints(query_minutiae, template)
        if score > best_score:
            best_score = score
            best_uid = uid
            best_match = matched

    return best_uid, best_score, best_match
