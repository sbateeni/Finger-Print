"""Image transforms before minutiae extraction (zoom, shift, auto-scale)."""

from __future__ import annotations

import cv2
import numpy as np

from config import AUTO_SCALE_MAX_FACTOR, AUTO_SCALE_MIN_FACTOR
from utils.quality_gate import check_fingerprint_quality, quality_gate_enabled


def _reject_low_upload_quality(o_gray, p_gray) -> dict | None:
    """Return error dict for ro/rp if pre-processing quality gate fails."""
    if not quality_gate_enabled():
        return None
    q_ref = check_fingerprint_quality(o_gray, label="المرجعية")
    if not q_ref["ok"]:
        return {"error": q_ref["message"], "quality_score": q_ref["quality_score"]}
    q_qry = check_fingerprint_quality(p_gray, label="المقارنة")
    if not q_qry["ok"]:
        return {"error": q_qry["message"], "quality_score": q_qry["quality_score"]}
    return None


def _clamp_zoom_percent(v: int) -> int:
    try:
        n = int(v)
    except Exception:
        return 100
    return max(50, min(250, n))


def _apply_zoom(gray: np.ndarray, zoom_percent: int, enabled: bool) -> np.ndarray:
    if not enabled:
        return gray
    z = _clamp_zoom_percent(zoom_percent)
    if z == 100:
        return gray
    scale = z / 100.0
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interpolation)


def _clamp_shift_px(v: int) -> int:
    try:
        n = int(v)
    except Exception:
        return 0
    return max(-300, min(300, n))


def _apply_shift(gray: np.ndarray, shift_x: int, shift_y: int, enabled: bool) -> np.ndarray:
    if not enabled:
        return gray
    tx = _clamp_shift_px(shift_x)
    ty = _clamp_shift_px(shift_y)
    if tx == 0 and ty == 0:
        return gray
    h, w = gray.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def _apply_manual_transform(
    gray: np.ndarray,
    zoom_percent: int,
    shift_x: int,
    shift_y: int,
    enabled: bool,
) -> np.ndarray:
    """
    Apply preview-like transform in a fixed frame:
    zoom around center, translate, keep original canvas size.
    """
    if not enabled:
        return gray
    z = _clamp_zoom_percent(zoom_percent) / 100.0
    tx = _clamp_shift_px(shift_x)
    ty = _clamp_shift_px(shift_y)
    if abs(z - 1.0) < 1e-6 and tx == 0 and ty == 0:
        return gray

    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, 0.0, z)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(
        gray,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def _largest_foreground_bbox(gray: np.ndarray) -> tuple[int, int] | None:
    if gray is None or gray.size == 0:
        return None
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w <= 1 or h <= 1:
        return None
    return int(w), int(h)


def _normalize_query_scale(
    reference_gray: np.ndarray,
    query_gray: np.ndarray,
    enabled: bool,
) -> tuple[np.ndarray, float]:
    if not enabled:
        return query_gray, 1.0
    ref_box = _largest_foreground_bbox(reference_gray)
    qry_box = _largest_foreground_bbox(query_gray)
    if ref_box is None or qry_box is None:
        return query_gray, 1.0

    ref_area = float(ref_box[0] * ref_box[1])
    qry_area = float(qry_box[0] * qry_box[1])
    if ref_area <= 0 or qry_area <= 0:
        return query_gray, 1.0

    factor = (ref_area / qry_area) ** 0.5
    factor = float(max(AUTO_SCALE_MIN_FACTOR, min(AUTO_SCALE_MAX_FACTOR, factor)))
    if abs(factor - 1.0) < 0.03:
        return query_gray, 1.0

    interpolation = cv2.INTER_CUBIC if factor > 1.0 else cv2.INTER_AREA
    scaled = cv2.resize(query_gray, None, fx=factor, fy=factor, interpolation=interpolation)
    return scaled, factor
