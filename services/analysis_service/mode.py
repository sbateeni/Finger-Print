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


def resolve_auto_align_sweep(
    form_value: str | None,
    analysis_mode: str | None,
) -> bool:
    """
    Wide Zoom/Shift grid before extraction (only in deep mode).
    Checkbox off → skip sweep; deep still runs the full live pipeline once.
    """
    if not is_deep_analysis(analysis_mode):
        return False
    if form_value is None:
        raw = os.getenv("DEEP_AUTO_ALIGN_SWEEP", "1")
        return str(raw).strip().lower() not in ("0", "false", "no", "off")
    return str(form_value).strip().lower() in ("1", "true", "on", "yes")


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
    auto_align_sweep: bool = False,
    ref_grid_divisions: int = 1,
    ref_grid_cell: int = 0,
    ref_grid_cells: str = "",
    ref_region: str = "0,0,1,1",
    partial_grid_divisions: int = 1,
    partial_grid_cells: str = "",
    partial_region: str = "0,0,1,1",
) -> tuple[int, int, int, dict | None]:
    """Wide auto-sweep before pipeline when deep mode and auto_align_sweep enabled."""
    if not is_deep_analysis(analysis_mode) or not auto_align_sweep:
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
        ref_grid_divisions=ref_grid_divisions,
        ref_grid_cell=ref_grid_cell,
        ref_grid_cells=ref_grid_cells,
        ref_region=ref_region,
        partial_grid_divisions=partial_grid_divisions,
        partial_grid_cells=partial_grid_cells,
        partial_region=partial_region,
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
