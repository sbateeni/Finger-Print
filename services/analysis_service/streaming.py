"""SSE live analysis stream for the web UI."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Iterator

from config import (
    MATCH_ANGLE_THRESHOLD_DEG,
    MATCH_DISTANCE_THRESHOLD,
    MIN_MINUTIAE_RECOMMENDED,
    OUTPUT_DIR,
    PARTIAL_VERIFY_SEARCH_RADIUS,
    QUALITY_GATE_MIN_MINUTIAE,
    QUALITY_GATE_MIN_SCORE,
)
from utils.forensic import append_audit_record, build_audit_record, enrich_match_for_forensics
from utils.fusion import apply_fusion_to_match, use_orb_fusion
from utils.image_utils import _decode_upload_type, _img_data_uri
from utils.matcher import (
    match_fingerprints_with_partial_alignment,
    visualize_alignment_on_reference,
    visualize_matches,
)
from utils.orb_matcher import match_with_orb
from utils.report_generator import generate_report
from utils.sse_utils import _sse_line

from .branch import _iter_branch_live
from .mode import _apply_deep_sweep_to_transforms, is_deep_analysis, resolve_analysis_mode
from .reports import _ensure_pdf_from_html
from .results import (
    _apply_partial_verify_step_audit,
    _make_inconclusive_result,
    _sanitize_match_for_json,
    build_report_pipeline,
)
from .transforms import (
    _apply_manual_transform,
    _clamp_shift_px,
    _clamp_zoom_percent,
    _normalize_query_scale,
    _reject_low_upload_quality,
)

logger = logging.getLogger(__name__)


def analysis_event_generator(
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
    analysis_mode: str = "deep",
    report_lang: str = "ar",
) -> Iterator[bytes]:
    try:
        yield _sse_line({"type": "log", "message": "جاري فك ترميز الصورتين…"})
        if not o_raw or not p_raw:
            yield _sse_line({"type": "fatal", "message": "يرفع ملفين صالحين."})
            return

        same_file = hashlib.sha256(o_raw).digest() == hashlib.sha256(p_raw).digest()
        sha_o = hashlib.sha256(o_raw).hexdigest()
        sha_p = hashlib.sha256(p_raw).hexdigest()
        yield _sse_line(
            {
                "type": "hashes",
                "same_file_warning": same_file,
                "sha256_original": sha_o,
                "sha256_partial": sha_p,
            }
        )

        try:
            o_gray = _decode_upload_type(o_raw)
            p_gray = _decode_upload_type(p_raw)
        except Exception as e:
            yield _sse_line({"type": "fatal", "message": f"تعذر فك ترميز الصورة: {e}"})
            return

        qerr = _reject_low_upload_quality(o_gray, p_gray)
        if qerr:
            yield _sse_line({"type": "fatal", "message": qerr["error"]})
            return

        dm_pre = denoise_method if denoise_method in ("None", "fastNlMeans", "GaussianBlur") else "fastNlMeans"
        if is_deep_analysis(analysis_mode):
            yield _sse_line(
                {
                    "type": "log",
                    "message": "فحص عميق: جاري البحث التلقائي الواسع عن أفضل محاذاة (Zoom/Shift)…",
                }
            )
            partial_zoom, partial_shift_x, partial_shift_y, sweep_meta = _apply_deep_sweep_to_transforms(
                o_raw,
                p_raw,
                border_margin,
                min_distance,
                min_contrast,
                min_angle_diff,
                dm_pre,
                fast_denoise_h,
                gauss_ksize,
                original_zoom,
                partial_zoom,
                partial_shift_x,
                partial_shift_y,
                apply_preview_scale,
                auto_scale_normalization,
                analysis_mode,
            )
            if sweep_meta and sweep_meta.get("ok") and sweep_meta.get("best"):
                b = sweep_meta["best"]
                yield _sse_line(
                    {
                        "type": "log",
                        "message": (
                            f"تم ضبط المحاذاة تلقائيًا: Zoom {b['partial_zoom']}%, "
                            f"X {b['partial_shift_x']}, Y {b['partial_shift_y']} "
                            f"(score {b.get('objective_score', 0)})."
                        ),
                    }
                )
            elif sweep_meta and not sweep_meta.get("ok"):
                yield _sse_line(
                    {
                        "type": "log",
                        "message": "تعذر إكمال البحث الواسع — يُتابع بالإعدادات اليدوية الحالية.",
                    }
                )
        else:
            yield _sse_line({"type": "log", "message": "فحص سريع — بدون بحث محاذاة واسع مسبق."})

        o_gray = _apply_manual_transform(o_gray, original_zoom, 0, 0, apply_preview_scale)
        p_gray = _apply_manual_transform(
            p_gray, partial_zoom, partial_shift_x, partial_shift_y, apply_preview_scale
        )
        p_gray, auto_scale_factor = _normalize_query_scale(o_gray, p_gray, auto_scale_normalization)
        if auto_scale_normalization and abs(auto_scale_factor - 1.0) >= 0.03:
            yield _sse_line(
                {
                    "type": "log",
                    "message": f"تطبيع مقياس المقارنة تلقائيًا ×{auto_scale_factor:.2f} قبل الاستخراج.",
                }
            )

        dm = denoise_method if denoise_method in ("None", "fastNlMeans", "GaussianBlur") else "fastNlMeans"

        holder_r: list[Any] = [None]
        for ev in _iter_branch_live(
            o_gray,
            dm,
            fast_denoise_h,
            gauss_ksize,
            border_margin,
            min_distance,
            min_contrast,
            min_angle_diff,
            "reference",
            holder_r,
        ):
            yield _sse_line(ev)
            if ev.get("type") == "error":
                yield _sse_line({"type": "fatal", "message": "المرجعية: " + ev.get("message", "")})
                return
        ro = holder_r[0]

        yield _sse_line({"type": "log", "message": "جاري معالجة البصمة المقارنة…"})
        holder_p: list[Any] = [None]
        for ev in _iter_branch_live(
            p_gray,
            dm,
            fast_denoise_h,
            gauss_ksize,
            border_margin,
            min_distance,
            min_contrast,
            min_angle_diff,
            "partial",
            holder_p,
        ):
            yield _sse_line(ev)
            if ev.get("type") == "error":
                yield _sse_line({"type": "fatal", "message": "المقارنة: " + ev.get("message", "")})
                return
        rp = holder_p[0]

        mo, mp = ro.get("minutiae") or [], rp.get("minutiae") or []
        if not mo or not mp:
            yield _sse_line(
                {
                    "type": "fatal",
                    "message": "لا توجد نقاط دقيقة كافية في إحدى الصورتين.",
                    "partial_results": True,
                }
            )
            return

        q_ref = float(ro.get("quality_score") or 0.0)
        q_qry = float(rp.get("quality_score") or 0.0)
        low_q = min(q_ref, q_qry)
        low_m = min(len(mo), len(mp))
        if low_q < QUALITY_GATE_MIN_SCORE or low_m < QUALITY_GATE_MIN_MINUTIAE:
            reason = (
                f"Quality Gate: quality={low_q:.1f} (min {QUALITY_GATE_MIN_SCORE:.1f}) "
                f"or minutiae={low_m} (min {QUALITY_GATE_MIN_MINUTIAE})"
            )
            yield _sse_line(
                {"type": "log", "message": "تم تفعيل Quality Gate — النتيجة غير حاسمة (INCONCLUSIVE)."}
            )
            match_result = enrich_match_for_forensics(_make_inconclusive_result(ro, rp, reason))
            form_params = _stream_form_params(
                border_margin,
                min_distance,
                min_contrast,
                min_angle_diff,
                dm,
                fast_denoise_h,
                gauss_ksize,
                original_zoom,
                partial_zoom,
                partial_shift_x,
                partial_shift_y,
                apply_preview_scale,
                auto_scale_normalization,
                auto_scale_factor,
                analysis_mode=analysis_mode,
                report_lang=report_lang,
            )
            _apply_partial_verify_step_audit(form_params, match_result)
            audit = _stream_audit(
                sha_o, sha_p, operator_name, case_reference, form_params, report_lang
            )
            pipeline = build_report_pipeline(ro, rp, matches_vis=None, include_singular=False)
            report_rel = _write_stream_report(
                ro["skeleton"], rp["skeleton"], match_result, audit, pipeline, report_lang
            )
            append_audit_record(
                build_audit_record(
                    sha256_original=sha_o,
                    sha256_partial=sha_p,
                    operator_name=operator_name,
                    case_reference=case_reference,
                    form_params=form_params,
                    match_result=match_result,
                    report_filename=report_rel,
                )
            )
            yield _sse_line(
                {
                    "type": "done",
                    "match": _sanitize_match_for_json(match_result),
                    "report_download": report_rel,
                    "audit": {"sha256_original": sha_o, "sha256_partial": sha_p},
                    "forensic_quality_warning": True,
                }
            )
            return

        yield _sse_line({"type": "log", "message": "جاري بحث المحاذاة والمطابقة…"})
        if min(len(mo), len(mp)) > 250:
            yield _sse_line(
                {
                    "type": "log",
                    "message": (
                        "عدد النقاط الدقيقة مرتفع — هذه الخطوة CPU‑intensive "
                        "وقد تستغرق عشرات الثواني؛ الصفحة لم تتجمد."
                    ),
                }
            )
        sk_o = ro["skeleton"]
        sk_p = rp["skeleton"]
        match_result = enrich_match_for_forensics(
            match_fingerprints_with_partial_alignment(
                mo,
                mp,
                sk_o.shape,
                cores_ref=ro.get("cores"),
                cores_qry=rp.get("cores"),
            )
        )
        yield _sse_line({"type": "log", "message": "اكتمل بحث المحاذاة — جاري تجهيز تصور المطابقة…"})
        matches_vis = visualize_alignment_on_reference(sk_o, match_result)
        if matches_vis is None:
            matches_vis = visualize_matches(sk_o, sk_p, match_result)

        yield _sse_line(
            {
                "type": "image",
                "branch": "match",
                "stage": "alignment",
                "src": _img_data_uri(matches_vis),
            }
        )

        yield _sse_line({"type": "log", "message": "جاري دمج النتائج (Minutiae + MCC)…"})
        try:
            if use_orb_fusion():
                yield _sse_line({"type": "log", "message": "جاري التحقق البصري (ORB)…"})
                orb_res = match_with_orb(ro["processed"], rp["processed"])
                if orb_res.get("visualization") is not None:
                    orb_res["orb_visualization"] = _img_data_uri(orb_res["visualization"])
                    yield _sse_line(
                        {
                            "type": "image",
                            "branch": "orb",
                            "stage": "orb_vis",
                            "src": orb_res["orb_visualization"],
                        }
                    )
                    del orb_res["visualization"]
            match_result = apply_fusion_to_match(match_result, ro["processed"], rp["processed"])
            match_result = enrich_match_for_forensics(match_result)
        except Exception as fusion_err:
            logger.error("Fusion failed: %s", fusion_err)
            yield _sse_line({"type": "log", "message": "فشل دمج النتائج — الاعتماد على مطابقة النقاط فقط."})

        form_params = _stream_form_params(
            border_margin,
            min_distance,
            min_contrast,
            min_angle_diff,
            dm,
            fast_denoise_h,
            gauss_ksize,
            original_zoom,
            partial_zoom,
            partial_shift_x,
            partial_shift_y,
            apply_preview_scale,
            auto_scale_normalization,
            auto_scale_factor,
            include_quality_gate=False,
            analysis_mode=analysis_mode,
            report_lang=report_lang,
        )
        _apply_partial_verify_step_audit(form_params, match_result)
        audit = _stream_audit(
            sha_o, sha_p, operator_name, case_reference, form_params, report_lang
        )

        yield _sse_line({"type": "log", "message": "جاري توليد التقرير…"})
        pipeline = build_report_pipeline(ro, rp, matches_vis=matches_vis, include_singular=True)
        report_rel = _write_stream_report(
            sk_o, sk_p, match_result, audit, pipeline, report_lang
        )

        append_audit_record(
            build_audit_record(
                sha256_original=sha_o,
                sha256_partial=sha_p,
                operator_name=operator_name,
                case_reference=case_reference,
                form_params=form_params,
                match_result=match_result,
                report_filename=report_rel,
            )
        )

        low_n = min(ro["minutiae_count"], rp["minutiae_count"])
        yield _sse_line(
            {
                "type": "done",
                "match": _sanitize_match_for_json(match_result),
                "report_download": report_rel,
                "audit": {"sha256_original": sha_o, "sha256_partial": sha_p},
                "forensic_quality_warning": low_n < MIN_MINUTIAE_RECOMMENDED,
            }
        )
    except Exception as e:
        logger.exception(e)
        yield _sse_line({"type": "fatal", "message": str(e)})


def _stream_form_params(
    border_margin,
    min_distance,
    min_contrast,
    min_angle_diff,
    dm,
    fast_denoise_h,
    gauss_ksize,
    original_zoom,
    partial_zoom,
    partial_shift_x,
    partial_shift_y,
    apply_preview_scale,
    auto_scale_normalization,
    auto_scale_factor,
    *,
    include_quality_gate: bool = True,
    analysis_mode: str = "deep",
    report_lang: str = "ar",
) -> dict[str, Any]:
    params = {
        "border_margin": border_margin,
        "min_distance": min_distance,
        "min_contrast": min_contrast,
        "min_angle_diff": min_angle_diff,
        "denoise_method": dm,
        "fast_denoise_h": fast_denoise_h,
        "gauss_ksize": gauss_ksize,
        "original_zoom": _clamp_zoom_percent(original_zoom),
        "partial_zoom": _clamp_zoom_percent(partial_zoom),
        "partial_shift_x": _clamp_shift_px(partial_shift_x),
        "partial_shift_y": _clamp_shift_px(partial_shift_y),
        "apply_preview_scale": bool(apply_preview_scale),
        "auto_scale_normalization": bool(auto_scale_normalization),
        "auto_scale_factor_applied": round(float(auto_scale_factor), 4),
        "MATCH_DISTANCE_THRESHOLD": MATCH_DISTANCE_THRESHOLD,
        "MATCH_ANGLE_THRESHOLD_DEG": MATCH_ANGLE_THRESHOLD_DEG,
        "PARTIAL_VERIFY_SEARCH_RADIUS": PARTIAL_VERIFY_SEARCH_RADIUS,
    }
    if include_quality_gate:
        params["QUALITY_GATE_MIN_SCORE"] = QUALITY_GATE_MIN_SCORE
        params["QUALITY_GATE_MIN_MINUTIAE"] = QUALITY_GATE_MIN_MINUTIAE
    params["analysis_mode"] = resolve_analysis_mode(analysis_mode)
    params["report_lang"] = report_lang
    return params


def _stream_audit(
    sha_o, sha_p, operator_name, case_reference, form_params, report_lang: str
) -> dict[str, Any]:
    return {
        "sha256_original": sha_o,
        "sha256_partial": sha_p,
        "operator_name": operator_name.strip(),
        "case_reference": case_reference.strip(),
        "report_lang": report_lang,
        "form_params": form_params,
    }


def _write_stream_report(
    sk_o, sk_p, match_result, audit, pipeline, report_lang: str
) -> str | None:
    report_path = generate_report(
        sk_o, sk_p, match_result, audit=audit, pipeline=pipeline, lang=report_lang
    )
    if report_path:
        try:
            _ensure_pdf_from_html(Path(report_path))
        except Exception as pdf_err:
            logger.warning("Auto PDF generation failed (stream path): %s", pdf_err)
    return str(Path(report_path).relative_to(OUTPUT_DIR)).replace("\\", "/") if report_path else None
