"""
Unified fingerprint pair analysis for web, CLI, Telegram, etc.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from config import (
    DEFAULT_BORDER_MARGIN,
    DEFAULT_MIN_ANGLE_DIFF,
    DEFAULT_MIN_CONTRAST,
    DEFAULT_MIN_DISTANCE,
)
from services.analysis_service import (
    _ensure_pdf_from_html,
    should_generate_pdf,
    process_form_analysis,
    run_auto_sweep,
    run_matching_pipeline,
)


def telegram_deep_analysis_enabled() -> bool:
    raw = (os.getenv("TELEGRAM_DEEP_ANALYSIS") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def telegram_sweep_mode() -> str:
    return (os.getenv("TELEGRAM_SWEEP_MODE") or "wide").strip().lower() or "wide"


def analyze_fingerprint_pair(
    reference_bytes: bytes,
    query_bytes: bytes,
    *,
    border_margin: int = DEFAULT_BORDER_MARGIN,
    min_distance: int = DEFAULT_MIN_DISTANCE,
    min_contrast: int = DEFAULT_MIN_CONTRAST,
    min_angle_diff: int = DEFAULT_MIN_ANGLE_DIFF,
    denoise_method: str = "fastNlMeans",
    fast_denoise_h: int = 10,
    gauss_ksize: int = 3,
    original_zoom: int = 100,
    partial_zoom: int = 100,
    partial_shift_x: int = 0,
    partial_shift_y: int = 0,
    apply_preview_scale: bool = True,
    auto_scale_normalization: bool = True,
    operator_name: str = "",
    case_reference: str = "",
    write_report_and_audit: bool = True,
    auto_sweep_before: bool = False,
    sweep_mode: str = "wide",
) -> dict[str, Any]:
    """
    Run full pipeline on two image byte blobs.
    Returns dict with keys: ok, error, match, report_html, report_pdf, audit, ...

    When auto_sweep_before=True (Telegram deep mode), searches best zoom/shift
    like the web Auto-sweep, then runs the same pipeline as the workstation.
    """
    if not reference_bytes or not query_bytes:
        return {"ok": False, "error": "ملفان فارغان أو غير صالحين."}

    sweep_meta: Optional[dict[str, Any]] = None
    eff_zoom = partial_zoom
    eff_shift_x = partial_shift_x
    eff_shift_y = partial_shift_y

    if auto_sweep_before:
        sweep_meta = run_auto_sweep(
            reference_bytes,
            query_bytes,
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
            sweep_mode=sweep_mode,
        )
        if sweep_meta.get("ok") and sweep_meta.get("best"):
            best = sweep_meta["best"]
            eff_zoom = int(best["partial_zoom"])
            eff_shift_x = int(best["partial_shift_x"])
            eff_shift_y = int(best["partial_shift_y"])

    try:
        same_file, sha_o, sha_p, ro, rp, dm = process_form_analysis(
            reference_bytes,
            query_bytes,
            border_margin,
            min_distance,
            min_contrast,
            min_angle_diff,
            denoise_method,
            fast_denoise_h,
            gauss_ksize,
            original_zoom,
            eff_zoom,
            eff_shift_x,
            eff_shift_y,
            apply_preview_scale,
            auto_scale_normalization,
            operator_name or "api",
            case_reference or "pair-analysis",
        )
    except Exception as e:
        return {"ok": False, "error": f"تعذر فك ترميز الصورة: {e}"}

    if ro.get("error"):
        return {"ok": False, "error": f"المرجعية: {ro['error']}"}
    if rp.get("error"):
        return {"ok": False, "error": f"المقارنة: {rp['error']}"}

    mo, mp = ro.get("minutiae") or [], rp.get("minutiae") or []
    if not mo or not mp:
        return {
            "ok": False,
            "error": "لا توجد نقاط دقيقة كافية في إحدى الصورتين.",
            "partial": True,
        }

    form_ctx = {
        "border_margin": border_margin,
        "min_distance": min_distance,
        "min_contrast": min_contrast,
        "min_angle_diff": min_angle_diff,
        "denoise_method": dm,
        "fast_denoise_h": fast_denoise_h,
        "gauss_ksize": gauss_ksize,
        "original_zoom": original_zoom,
        "partial_zoom": eff_zoom,
        "partial_shift_x": eff_shift_x,
        "partial_shift_y": eff_shift_y,
        "apply_preview_scale": apply_preview_scale,
        "auto_scale_normalization": auto_scale_normalization,
        "auto_scale_factor_applied": round(float(ro.get("auto_scale_factor_applied", 1.0)), 4),
        "auto_sweep_applied": bool(auto_sweep_before),
        "auto_sweep_mode": sweep_mode if auto_sweep_before else None,
    }

    match_result, _vis, report_rel, audit, quality_warning = run_matching_pipeline(
        ro,
        rp,
        sha_o,
        sha_p,
        dm,
        form_ctx,
        operator_name or "api",
        case_reference or "pair-analysis",
        border_margin,
        min_distance,
        min_contrast,
        min_angle_diff,
        fast_denoise_h,
        gauss_ksize,
        write_report_and_audit=write_report_and_audit,
    )

    report_html: Optional[Path] = None
    report_pdf: Optional[Path] = None
    if report_rel:
        from config import OUTPUT_DIR

        report_html = (Path(OUTPUT_DIR) / report_rel.replace("\\", "/")).resolve()
        if report_html.exists() and write_report_and_audit:
            rlang = (audit or {}).get("report_lang") or "ar"
            if should_generate_pdf(report_html, lang=rlang):
                try:
                    report_pdf = _ensure_pdf_from_html(report_html, lang=rlang)
                except Exception:
                    report_pdf = None

    out: dict[str, Any] = {
        "ok": True,
        "error": None,
        "same_file_warning": same_file,
        "match": match_result,
        "audit": audit,
        "report_rel": report_rel,
        "report_html": str(report_html) if report_html and report_html.exists() else None,
        "report_pdf": str(report_pdf) if report_pdf and report_pdf.exists() else None,
        "forensic_quality_warning": quality_warning,
        "deep_analysis": bool(auto_sweep_before),
    }
    if sweep_meta is not None:
        out["auto_sweep"] = sweep_meta
    return out


def format_match_summary_ar(result: dict[str, Any]) -> str:
    """Short Arabic summary for Telegram / notifications."""
    if not result.get("ok"):
        return f"❌ فشل التحليل: {result.get('error', 'خطأ غير معروف')}"

    m = result.get("match") or {}
    lines = [
        "🔬 *نتيجة مطابقة البصمات*",
        "",
    ]
    if result.get("deep_analysis"):
        lines.append("• التحليل: *عميق* (نفس محرك الواجهة + محاذاة تلقائية)")
        sweep = result.get("auto_sweep") or {}
        if sweep.get("ok") and sweep.get("best"):
            b = sweep["best"]
            lines.append(
                f"• محاذاة: zoom *{b.get('partial_zoom')}%* | "
                f"إزاحة ({b.get('partial_shift_x')}, {b.get('partial_shift_y')}) | "
                f"اختبر {sweep.get('tested', '?')} توليفة"
            )
        elif sweep and not sweep.get("ok"):
            lines.append("• محاذاة: افتراضية (فشل Auto-sweep)")
        lines.append("")
    lines.extend([
        f"• الحالة: *{m.get('status', '—')}*",
        f"• Fused Score: *{float(m.get('fused_score') or 0):.2f}%*",
        f"• Match score: *{float(m.get('match_score') or 0):.2f}%*",
        f"• MCC: *{float(m.get('mcc_score') or 0):.2f}%*",
        f"• تطابقات: *{m.get('matched_points', 0)}*",
        f"• نقاط مرجعية: {m.get('total_original', 0)} | مقارنة: {m.get('total_partial', 0)}",
    ])
    if m.get("decision_mode"):
        lines.append(f"• وضع القرار: `{m.get('decision_mode')}`")
    if m.get("combined_verdict"):
        lines.append(f"• الحكم: {m.get('combined_verdict')}")
    if result.get("same_file_warning"):
        lines.append("")
        lines.append("⚠️ الملفان متطابقان بايتًا.")
    if result.get("forensic_quality_warning"):
        lines.append("⚠️ جودة/عدد نقاط منخفض — راجع النتيجة يدويًا.")
    lines.append("")
    lines.append("_للاستخدام المخبري فقط — يتطلب مراجعة خبير._")
    return "\n".join(lines)
