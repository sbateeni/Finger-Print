"""Auto-search best manual transform (zoom/shift) for partial fingerprint."""

from __future__ import annotations

from typing import Any

from utils.image_utils import _decode_upload_type
from utils.matcher import match_fingerprints_with_partial_alignment

from .branch import _process_branch
from .ref_grid import apply_fingerprint_region, normalize_ref_grid
from .transforms import (
    _apply_manual_transform,
    effective_preview_transform,
    _clamp_shift_px,
    _clamp_zoom_percent,
    _normalize_query_scale,
)


def run_auto_sweep(
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
    sweep_mode: str = "quick",
    ref_grid_divisions: int = 1,
    ref_grid_cell: int = 0,
    ref_grid_cells: str = "",
    ref_region: str = "0,0,1,1",
    partial_grid_divisions: int = 1,
    partial_grid_cells: str = "",
    partial_region: str = "0,0,1,1",
) -> dict[str, Any]:
    """Auto-search best manual transform around current partial parameters."""
    o_gray = _decode_upload_type(o_raw)
    p_gray = _decode_upload_type(p_raw)
    dm = denoise_method if denoise_method in ("None", "fastNlMeans", "GaussianBlur") else "fastNlMeans"

    ref_grid_divisions, ref_grid_cell = normalize_ref_grid(ref_grid_divisions, ref_grid_cell)
    cells_ref = ref_grid_cells or (str(ref_grid_cell) if ref_grid_divisions > 1 else "")
    base_ref = apply_fingerprint_region(
        o_gray, ref_region, grid_divisions=ref_grid_divisions, grid_cells=cells_ref
    )
    ref_z, ref_sx, ref_sy, _ = effective_preview_transform(
        original_zoom, 0, 0, ref_region, apply_preview_scale
    )
    base_ref = _apply_manual_transform(base_ref, ref_z, ref_sx, ref_sy, apply_preview_scale)
    ro = _process_branch(
        base_ref,
        dm,
        fast_denoise_h,
        gauss_ksize,
        border_margin,
        min_distance,
        min_contrast,
        min_angle_diff,
    )
    if ro.get("error"):
        return {"ok": False, "message": f"المرجعية: {ro.get('error')}"}

    mode = (sweep_mode or "quick").strip().lower()
    if mode == "wide":
        zoom_offsets = (-14, -10, -6, -2, 0, 2, 6, 10, 14)
        shift_offsets = (-36, -24, -12, 0, 12, 24, 36)
    else:
        zoom_offsets = (-8, -4, 0, 4, 8)
        shift_offsets = (-12, 0, 12)

    zoom_candidates = sorted(set(_clamp_zoom_percent(partial_zoom + d) for d in zoom_offsets))
    sx_candidates = sorted(set(_clamp_shift_px(partial_shift_x + d) for d in shift_offsets))
    sy_candidates = sorted(set(_clamp_shift_px(partial_shift_y + d) for d in shift_offsets))

    best = None
    top_runs: list[dict[str, Any]] = []
    tested = 0

    for z in zoom_candidates:
        for sx in sx_candidates:
            for sy in sy_candidates:
                tested += 1
                qry = apply_fingerprint_region(
                    p_gray,
                    partial_region,
                    grid_divisions=partial_grid_divisions,
                    grid_cells=partial_grid_cells,
                )
                # Sweep varies alignment zoom/shift only when partial is full-frame.
                _, _, _, region_set = effective_preview_transform(
                    z, sx, sy, partial_region, apply_preview_scale
                )
                if region_set:
                    qry = _apply_manual_transform(qry, 100, 0, 0, apply_preview_scale)
                else:
                    qry = _apply_manual_transform(qry, z, sx, sy, apply_preview_scale)
                qry, auto_scale_factor = _normalize_query_scale(base_ref, qry, auto_scale_normalization)
                rp = _process_branch(
                    qry,
                    dm,
                    fast_denoise_h,
                    gauss_ksize,
                    border_margin,
                    min_distance,
                    min_contrast,
                    min_angle_diff,
                )
                if rp.get("error"):
                    continue

                mo, mp = ro.get("minutiae") or [], rp.get("minutiae") or []
                if not mo or not mp:
                    continue

                mr = match_fingerprints_with_partial_alignment(
                    mo,
                    mp,
                    ro["skeleton"].shape,
                    cores_ref=ro.get("cores"),
                    cores_qry=rp.get("cores"),
                )
                match_score = float(mr.get("match_score") or 0.0)
                mcc_score = float(mr.get("mcc_score") or 0.0)
                matched_points = int(mr.get("matched_points") or 0)
                gain = int(mr.get("alignment_gain_matches") or 0)
                objective = 0.55 * mcc_score + 0.35 * match_score + 0.10 * min(100.0, matched_points * 2.0)

                rec = {
                    "partial_zoom": int(z),
                    "partial_shift_x": int(sx),
                    "partial_shift_y": int(sy),
                    "auto_scale_factor_applied": round(float(auto_scale_factor), 4),
                    "objective_score": round(float(objective), 2),
                    "match_score": round(match_score, 2),
                    "mcc_score": round(mcc_score, 2),
                    "matched_points": matched_points,
                    "alignment_gain_matches": gain,
                }
                top_runs.append(rec)
                if best is None or rec["objective_score"] > best["objective_score"]:
                    best = rec

    if best is None:
        return {"ok": False, "message": "تعذر إيجاد توليفة مناسبة في نطاق Auto-sweep."}

    top_runs.sort(key=lambda r: r["objective_score"], reverse=True)
    return {
        "ok": True,
        "mode": mode,
        "tested": tested,
        "best": best,
        "top3": top_runs[:3],
    }
