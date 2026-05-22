"""
Minutiae extraction: skeleton pipeline (default) + optional pyfing (LEADER) supplement.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_pyfing_checked = False
_pyfing_ok = False


def use_pyfing_extraction() -> bool:
    return (os.getenv("USE_PYFING_EXTRACTION") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def pyfing_available() -> bool:
    global _pyfing_checked, _pyfing_ok
    if _pyfing_checked:
        return _pyfing_ok
    _pyfing_checked = True
    if not use_pyfing_extraction():
        _pyfing_ok = False
        return False
    try:
        import pyfing  # noqa: F401

        _pyfing_ok = True
    except Exception as exc:
        logger.info("pyfing unavailable: %s", exc)
        _pyfing_ok = False
    return _pyfing_ok


def extract_minutiae_pyfing(image: np.ndarray) -> list[dict[str, Any]]:
    """
    End-to-end minutiae via pyfing LEADER (requires keras + tensorflow).
    Returns project-standard dicts: x, y, type, angle.
    """
    if not pyfing_available():
        return []

    import cv2
    import pyfing as pf

    gray = image
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    try:
        raw = pf.minutiae_extraction(gray)
    except Exception as exc:
        logger.warning("pyfing minutiae_extraction failed: %s", exc)
        return []

    return _normalize_pyfing_minutiae(raw)


def _normalize_pyfing_minutiae(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if raw is None:
        return out

    items = raw
    if isinstance(raw, np.ndarray):
        if raw.ndim == 1:
            items = [raw]
        else:
            items = [raw[i] for i in range(len(raw))]

    for m in items:
        try:
            if isinstance(m, dict):
                x = float(m.get("x", m.get("col", 0)))
                y = float(m.get("y", m.get("row", 0)))
                ang = float(m.get("angle", m.get("direction", m.get("theta", 0))))
                t = m.get("type", "endpoint")
            else:
                arr = np.asarray(m).ravel()
                if arr.size < 3:
                    continue
                x, y = float(arr[0]), float(arr[1])
                ang = float(arr[2]) if arr.size > 2 else 0.0
                t = "bifurcation" if arr.size > 3 and int(arr[3]) == 1 else "endpoint"
            ang_deg = float(np.degrees(ang)) if abs(ang) <= np.pi + 0.1 else float(ang)
            ang_deg = ang_deg % 360.0
            mtype = "bifurcation" if str(t).lower() in ("bifurcation", "b", "1") else "endpoint"
            out.append({"x": int(round(x)), "y": int(round(y)), "type": mtype, "angle": ang_deg})
        except (TypeError, ValueError, IndexError):
            continue
    return out


def merge_minutiae_sets(
    primary: list[dict[str, Any]],
    supplement: list[dict[str, Any]],
    *,
    min_distance: float = 12.0,
    max_total: int = 500,
) -> list[dict[str, Any]]:
    """Merge pyfing points into skeleton list without duplicates."""
    if not supplement:
        return primary
    merged = list(primary)
    for s in supplement:
        too_close = False
        for p in merged:
            if np.hypot(p["x"] - s["x"], p["y"] - s["y"]) < min_distance:
                too_close = True
                break
        if not too_close:
            merged.append(s)
        if len(merged) >= max_total:
            break
    return merged


def enhance_minutiae_from_image(
    processed_image: np.ndarray,
    skeleton_minutiae: list[dict[str, Any]],
    *,
    min_distance: float = 12.0,
) -> tuple[list[dict[str, Any]], str]:
    """
    Optionally supplement skeleton minutiae with pyfing.
    Returns (minutiae, extraction_note).
    """
    if not use_pyfing_extraction():
        return skeleton_minutiae, "skeleton_cn"

    extra = extract_minutiae_pyfing(processed_image)
    if not extra:
        note = "skeleton_cn" if not pyfing_available() else "skeleton_cn_pyfing_failed"
        return skeleton_minutiae, note

    merged = merge_minutiae_sets(skeleton_minutiae, extra, min_distance=min_distance)
    return merged, f"skeleton_cn+pyfing({len(extra)})"
