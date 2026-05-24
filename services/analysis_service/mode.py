"""Analysis mode (fast vs deep) and pre-pipeline auto-sweep."""

from __future__ import annotations

import os

from .sweep import run_auto_sweep


def resolve_analysis_mode(mode: str | None) -> str:
    raw = (mode if mode is not None and str(mode).strip() else None) or os.getenv(
        "DEFAULT_ANALYSIS_MODE", "deep"
    )
    return (
        "deep"
        if str(raw).strip().lower() in ("deep", "1", "true", "on", "wide")
        else "fast"
    )


def is_deep_analysis(mode: str | None) -> bool:
    return resolve_analysis_mode(mode) == "deep"


def _apply_deep_sweep_to_transforms(
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
    analysis_mode: str,
) -> tuple[int, int, int, dict | None]:
    """Wide auto-sweep before pipeline when analysis_mode is deep."""
    if not is_deep_analysis(analysis_mode):
        return partial_zoom, partial_shift_x, partial_shift_y, None
    sweep = run_auto_sweep(
        o_raw,
        p_raw,
        border_margin,
        min_distance,
        min_contrast,
        min_angle_diff,
        denoise_method,
        fast_denoise_h,
        gauss_ksize,
        original_zoom,
        partial_zoom,
        partial_shift_x,
        partial_shift_y,
        apply_preview_scale,
        auto_scale_normalization,
        sweep_mode="wide",
    )
    if sweep.get("ok") and sweep.get("best"):
        best = sweep["best"]
        return (
            int(best["partial_zoom"]),
            int(best["partial_shift_x"]),
            int(best["partial_shift_y"]),
            sweep,
        )
    return partial_zoom, partial_shift_x, partial_shift_y, sweep
