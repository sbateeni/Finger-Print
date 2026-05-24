"""Decode uploads and run reference/query preprocessing branches."""

from __future__ import annotations

import hashlib

from utils.image_utils import _decode_upload_type

from .branch import _process_branch
from .transforms import (
    _apply_manual_transform,
    _normalize_query_scale,
    _reject_low_upload_quality,
)


def process_form_analysis(
    o_raw: bytes,
    p_raw: bytes,
    border_margin: int,
    min_distance: int,
    min_contrast: int,
    min_angle_diff: int,
    denoise_method: str,
    fast_denoise_h: int,
    gauss_ksize: int,
    original_zoom: int,
    partial_zoom: int,
    partial_shift_x: int,
    partial_shift_y: int,
    apply_preview_scale: bool,
    auto_scale_normalization: bool,
    operator_name: str,
    case_reference: str,
):
    same_file = hashlib.sha256(o_raw).digest() == hashlib.sha256(p_raw).digest()
    sha_o = hashlib.sha256(o_raw).hexdigest()
    sha_p = hashlib.sha256(p_raw).hexdigest()

    o_gray = _decode_upload_type(o_raw)
    p_gray = _decode_upload_type(p_raw)
    qerr = _reject_low_upload_quality(o_gray, p_gray)
    if qerr:
        dm = denoise_method if denoise_method in ("None", "fastNlMeans", "GaussianBlur") else "fastNlMeans"
        return same_file, sha_o, sha_p, qerr, qerr, dm

    o_gray = _apply_manual_transform(o_gray, original_zoom, 0, 0, apply_preview_scale)
    p_gray = _apply_manual_transform(
        p_gray, partial_zoom, partial_shift_x, partial_shift_y, apply_preview_scale
    )
    p_gray, auto_scale_factor = _normalize_query_scale(o_gray, p_gray, auto_scale_normalization)

    dm = denoise_method if denoise_method in ("None", "fastNlMeans", "GaussianBlur") else "fastNlMeans"

    ro = _process_branch(
        o_gray,
        dm,
        fast_denoise_h,
        gauss_ksize,
        border_margin,
        min_distance,
        min_contrast,
        min_angle_diff,
    )
    rp = _process_branch(
        p_gray,
        dm,
        fast_denoise_h,
        gauss_ksize,
        border_margin,
        min_distance,
        min_contrast,
        min_angle_diff,
    )

    ro["auto_scale_factor_applied"] = auto_scale_factor
    rp["auto_scale_factor_applied"] = auto_scale_factor
    return same_file, sha_o, sha_p, ro, rp, dm
