"""
Fingerprint analysis service — modular package.

Import from ``services.analysis_service`` (same API as before the split).
"""

from .branch import _iter_branch_live, _process_branch
from .form_analysis import process_form_analysis
from .mode import (
    _apply_deep_sweep_to_transforms,
    is_deep_analysis,
    resolve_analysis_mode,
)
from .pipeline import run_matching_pipeline
from .reports import _ensure_pdf_from_html
from .results import (
    _apply_partial_verify_step_audit,
    _build_visual_ctx,
    _make_inconclusive_result,
    _sanitize_match_for_json,
    build_report_pipeline,
)
from .streaming import analysis_event_generator
from .sweep import run_auto_sweep
from .transforms import (
    _apply_manual_transform,
    _apply_shift,
    _apply_zoom,
    _clamp_shift_px,
    _clamp_zoom_percent,
    _largest_foreground_bbox,
    _normalize_query_scale,
    _reject_low_upload_quality,
)

__all__ = [
    "analysis_event_generator",
    "process_form_analysis",
    "run_matching_pipeline",
    "run_auto_sweep",
    "resolve_analysis_mode",
    "is_deep_analysis",
    "build_report_pipeline",
    "_process_branch",
    "_iter_branch_live",
    "_apply_deep_sweep_to_transforms",
    "_ensure_pdf_from_html",
    "_sanitize_match_for_json",
    "_apply_partial_verify_step_audit",
    "_make_inconclusive_result",
    "_build_visual_ctx",
    "_reject_low_upload_quality",
    "_apply_manual_transform",
    "_apply_shift",
    "_apply_zoom",
    "_clamp_shift_px",
    "_clamp_zoom_percent",
    "_largest_foreground_bbox",
    "_normalize_query_scale",
]
