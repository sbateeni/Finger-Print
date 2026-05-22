"""
Pre-processing image quality gate (NFIQ2-style heuristic until native NFIQ2 is wired).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from preprocessing.image_quality import assess_image_quality


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def quality_gate_enabled() -> bool:
    return _env_bool("QUALITY_GATE_ENABLED", True)


def quality_min_score() -> float:
    raw = (os.getenv("QUALITY_GATE_MIN_SCORE") or "").strip()
    try:
        return float(raw) if raw else 30.0
    except ValueError:
        return 30.0


def check_fingerprint_quality(gray_image: np.ndarray, *, label: str = "image") -> dict[str, Any]:
    """
    Returns ok, quality_score (0–100), message, metrics.
    Uses existing contrast/sharpness/noise heuristics (not certified NFIQ2).
    """
    if gray_image is None or gray_image.size == 0:
        return {
            "ok": False,
            "quality_score": 0.0,
            "message": f"{label}: صورة فارغة أو غير صالحة.",
            "metrics": {},
        }

    assessment = assess_image_quality(gray_image)
    # assess_image_quality returns 0–1 overall
    score_100 = round(float(assessment.get("quality_score", 0)) * 100.0, 1)
    min_score = quality_min_score()
    ok = score_100 >= min_score

    if ok:
        message = ""
    else:
        recs = assessment.get("recommendations") or []
        hint = "؛ ".join(recs[:2]) if recs else "أعد التقاط الصورة بإضاءة وتباين أفضل"
        message = (
            f"{label}: جودة الصورة منخفضة ({score_100:.0f}/100، الحد الأدنى {min_score:.0f}). "
            f"{hint}"
        )

    return {
        "ok": ok,
        "quality_score": score_100,
        "message": message,
        "metrics": assessment.get("metrics") or {},
        "recommendations": assessment.get("recommendations") or [],
        "quality_method": "heuristic_v1",
    }
