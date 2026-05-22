"""
Pre-processing image quality gate (NFIQ2-style heuristic until native NFIQ2 is wired).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from preprocessing.quality import quality_result_dict


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

    min_score = quality_min_score()
    result = quality_result_dict(gray_image, label=label, threshold=min_score)
    return result
