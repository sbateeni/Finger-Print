import hashlib
import logging
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from config import (
    MATCH_ANGLE_THRESHOLD_DEG,
    MATCH_DISTANCE_THRESHOLD,
    MATCH_SCORE_THRESHOLDS,
    MIN_MINUTIAE_RECOMMENDED,
    OUTPUT_DIR,
    PARTIAL_VERIFY_SEARCH_RADIUS,
    PARTIAL_VERIFY_STEP_PX,
    QUALITY_GATE_MIN_SCORE,
    QUALITY_GATE_MIN_MINUTIAE,
)
from utils.image_processing import (
    detect_edges,
    enhance_image,
    preprocess_image,
    assess_fingerprint_quality,
    detect_singular_points,
)
from utils.matcher import (
    match_fingerprints_with_partial_alignment,
    visualize_alignment_on_reference,
    visualize_matches,
)
from utils.minutiae_extractor import (
    extract_minutiae,
    visualize_minutiae,
    visualize_singular_points,
)
from utils.orb_matcher import combined_verdict, match_with_orb
from utils.forensic import append_audit_record, build_audit_record, enrich_match_for_forensics
from utils.report_generator import generate_report
from utils.image_utils import _img_data_uri, _decode_upload_type
from utils.sse_utils import _sse_line

logger = logging.getLogger(__name__)


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


def _apply_manual_transform(gray: np.ndarray, zoom_percent: int, shift_x: int, shift_y: int, enabled: bool) -> np.ndarray:
    """
    Apply preview-like transform in a fixed frame:
    - zoom in/out around image center
    - translate by (shift_x, shift_y)
    - keep original canvas size (crop when zoom-in, pad white when zoom-out)
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


def _normalize_query_scale(reference_gray: np.ndarray, query_gray: np.ndarray, enabled: bool) -> tuple[np.ndarray, float]:
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

    # Stable scaling: area-ratio converted to linear factor.
    factor = (ref_area / qry_area) ** 0.5
    factor = float(max(0.70, min(1.40, factor)))
    if abs(factor - 1.0) < 0.03:
        return query_gray, 1.0

    interpolation = cv2.INTER_CUBIC if factor > 1.0 else cv2.INTER_AREA
    scaled = cv2.resize(query_gray, None, fx=factor, fy=factor, interpolation=interpolation)
    return scaled, factor


def _process_branch(
    gray: np.ndarray,
    denoise_method: str,
    fast_denoise_h: int,
    gauss_ksize: int,
    border_margin: int,
    min_distance: int,
    min_contrast: int,
    min_angle_diff: int,
):
    out = {}
    proc = preprocess_image(
        gray,
        denoise_method=denoise_method,
        fast_denoise_h=fast_denoise_h,
        gauss_ksize=gauss_ksize,
    )
    if proc is None:
        out["error"] = "فشلت المعالجة المسبقة"
        return out
    out["processed"] = proc
    out["white_pre"] = int(np.sum(proc == 255))

    quality_score, _qm = assess_fingerprint_quality(proc)
    out["quality_score"] = quality_score

    ridges, _omap = detect_edges(proc)
    if ridges is None:
        out["error"] = "فشل استخراج التموجات"
        return out
    out["ridges"] = ridges
    out["white_ridges"] = int(np.sum(ridges == 255))

    skel = enhance_image(ridges)
    if skel is None:
        out["error"] = "فشلت الهيكلة"
        return out
    out["skeleton"] = skel
    out["white_skel"] = int(np.sum(skel == 255))

    minutiae = extract_minutiae(
        skel,
        border_margin=border_margin,
        min_distance=min_distance,
        original_image=proc,
        min_contrast=min_contrast,
        min_angle_diff=min_angle_diff,
    )
    out["minutiae"] = minutiae
    out["minutiae_count"] = len(minutiae)
    if minutiae:
        vis = visualize_minutiae(skel, minutiae)
        out["vis_minutiae"] = vis
    return out


def _iter_branch_live(
    gray: np.ndarray,
    denoise_method: str,
    fast_denoise_h: int,
    gauss_ksize: int,
    border_margin: int,
    min_distance: int,
    min_contrast: int,
    min_angle_diff: int,
    branch_key: str,
    holder: list,
) -> Iterator[dict[str, Any]]:
    """يُصدِر أحداث صورة تلو الأخرى ثم يضع الناتج الكامل في holder[0]."""
    holder[0] = None
    proc = preprocess_image(
        gray,
        denoise_method=denoise_method,
        fast_denoise_h=fast_denoise_h,
        gauss_ksize=gauss_ksize,
    )
    if proc is None:
        yield {"type": "error", "branch": branch_key, "message": "فشلت المعالجة المسبقة"}
        return
    yield {
        "type": "image",
        "branch": branch_key,
        "stage": "processed",
        "src": _img_data_uri(proc),
        "white": int(np.sum(proc > 127)),
    }

    # 4. تقييم الجودة (Quality Assessment)
    quality_score, q_map = assess_fingerprint_quality(proc)
    # تلوين خريطة الجودة (Heatmap) - أحمر للجودة المنخفضة، أزرق/أخضر للعالية
    q_map_color = cv2.applyColorMap(q_map, cv2.COLORMAP_JET)
    yield {
        "type": "image",
        "branch": branch_key,
        "stage": "quality_map",
        "src": _img_data_uri(q_map_color),
        "quality_score": round(quality_score, 1)
    }

    ridges, _omap = detect_edges(proc)
    if ridges is None:
        yield {"type": "error", "branch": branch_key, "message": "فشل استخراج التموجات"}
        return
    yield {
        "type": "image",
        "branch": branch_key,
        "stage": "ridges",
        "src": _img_data_uri(ridges),
        "white": int(np.sum(ridges > 127)),
    }

    skel = enhance_image(ridges)
    if skel is None:
        yield {"type": "error", "branch": branch_key, "message": "فشلت الهيكلة"}
        return
    yield {
        "type": "image",
        "branch": branch_key,
        "stage": "skeleton",
        "src": _img_data_uri(skel),
        "white": int(np.sum(skel > 0)),
    }

    minutiae = extract_minutiae(
        skel,
        border_margin=border_margin,
        min_distance=min_distance,
        original_image=proc,
        min_contrast=min_contrast,
        min_angle_diff=min_angle_diff,
    )
    vis = None
    if minutiae:
        vis = visualize_minutiae(skel, minutiae)
    if vis is not None:
        yield {
            "type": "image",
            "branch": branch_key,
            "stage": "minutiae_vis",
            "src": _img_data_uri(vis),
            "n_min": len(minutiae),
        }

    # 6. اكتشاف النقاط المفردة (Singular Points)
    cores, deltas = detect_singular_points(_omap, proc > 127)
    if cores or deltas:
        vis_sp = visualize_singular_points(proc, cores, deltas)
        yield {
            "type": "image",
            "branch": branch_key,
            "stage": "singular_vis",
            "src": _img_data_uri(vis_sp),
        }

    out = {
        "processed": proc,
        "ridges": ridges,
        "skeleton": skel,
        "minutiae": minutiae,
        "minutiae_count": len(minutiae),
        "quality_score": quality_score,
        "quality_map": q_map_color,
        "cores": cores,
        "deltas": deltas,
        "white_pre": int(np.sum(proc > 127)),
        "white_ridges": int(np.sum(ridges > 127)),
        "white_skel": int(np.sum(skel > 0)),
        "vis_minutiae": vis,
    }
    holder[0] = out


def _sanitize_match_for_json(mr: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in mr.items() if k != "matched_details"}


def _apply_partial_verify_step_audit(form_params: dict[str, Any], match_result: dict[str, Any]) -> None:
    """يسجّل خطوة شبكة المحاذاة الفعلية المستخدمة (قد تختلف عن config عند التكييف التلقائي)."""
    eff = match_result.get("partial_verify_step_px_effective")
    cfg = match_result.get("partial_verify_step_px_config")
    form_params["PARTIAL_VERIFY_STEP_PX"] = int(eff) if eff is not None else PARTIAL_VERIFY_STEP_PX
    if cfg is not None and eff is not None and int(cfg) != int(eff):
        form_params["PARTIAL_VERIFY_STEP_PX_configured"] = int(cfg)


def _ensure_pdf_from_html(html_path: Path) -> Path:
    """
    Convert report HTML into a real PDF file using xhtml2pdf.
    Returns the PDF path (same folder, same basename).
    """
    pdf_path = html_path.with_suffix(".pdf")
    if pdf_path.exists() and pdf_path.stat().st_mtime >= html_path.stat().st_mtime:
        return pdf_path

    try:
        from xhtml2pdf import pisa  # lazy import: optional runtime dependency
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail=f"PDF engine not installed. Run: pip install xhtml2pdf ({e})",
        ) from e

    html_text = html_path.read_text(encoding="utf-8", errors="replace")
    with pdf_path.open("wb") as pdf_file:
        result = pisa.CreatePDF(src=html_text, dest=pdf_file, encoding="utf-8")
    if result.err:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail="Failed to generate PDF from report.")
    return pdf_path


def _make_inconclusive_result(ro: dict[str, Any], rp: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "matched_points": 0,
        "total_original": int(ro.get("minutiae_count") or 0),
        "total_partial": int(rp.get("minutiae_count") or 0),
        "match_score": 0.0,
        "dice_score": 0.0,
        "status": "INCONCLUSIVE",
        "quality_gate_failed": True,
        "quality_gate_reason": reason,
        "partial_verify": False,
        "alignment": None,
        "baseline_matched": 0,
        "baseline_match_score": 0.0,
        "alignment_gain_matches": 0,
        "alignment_gain_score": 0.0,
        "fused_score": 0.0,
        "fusion_components": {
            "minutiae_score": 0.0,
            "mcc_score": 0.0,
            "orb_score": 0.0,
        },
        "orb_matches": 0,
        "orb_score": 0.0,
        "orb_confidence": "INSUFFICIENT",
        "mcc_score": 0.0,
        "mcc_matches": 0,
    }


def _build_visual_ctx(ro, rp, match_result, matches_vis):
    return {
        "original": {
            "processed": _img_data_uri(ro["processed"]),
            "ridges": _img_data_uri(ro["ridges"]),
            "skeleton": _img_data_uri(ro["skeleton"]),
            "vis": _img_data_uri(ro.get("vis_minutiae")) if ro.get("vis_minutiae") is not None else "",
            "white_pre": ro["white_pre"],
            "white_ridges": ro["white_ridges"],
            "white_skel": ro["white_skel"],
            "n_min": ro["minutiae_count"],
        },
        "partial": {
            "processed": _img_data_uri(rp["processed"]),
            "ridges": _img_data_uri(rp["ridges"]),
            "skeleton": _img_data_uri(rp["skeleton"]),
            "vis": _img_data_uri(rp.get("vis_minutiae")) if rp.get("vis_minutiae") is not None else "",
            "white_pre": rp["white_pre"],
            "white_ridges": rp["white_ridges"],
            "white_skel": rp["white_skel"],
            "n_min": rp["minutiae_count"],
        },
        "matches_vis": _img_data_uri(matches_vis) if matches_vis is not None else "",
        "has_match": match_result is not None,
    }


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

        o_gray = _apply_manual_transform(
            o_gray,
            original_zoom,
            0,
            0,
            apply_preview_scale,
        )
        p_gray = _apply_manual_transform(
            p_gray,
            partial_zoom,
            partial_shift_x,
            partial_shift_y,
            apply_preview_scale,
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
            yield _sse_line({"type": "log", "message": "تم تفعيل Quality Gate — النتيجة غير حاسمة (INCONCLUSIVE)."})
            match_result = enrich_match_for_forensics(_make_inconclusive_result(ro, rp, reason))
            form_params = {
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
                "QUALITY_GATE_MIN_SCORE": QUALITY_GATE_MIN_SCORE,
                "QUALITY_GATE_MIN_MINUTIAE": QUALITY_GATE_MIN_MINUTIAE,
            }
            _apply_partial_verify_step_audit(form_params, match_result)
            audit = {
                "sha256_original": sha_o,
                "sha256_partial": sha_p,
                "operator_name": operator_name.strip(),
                "case_reference": case_reference.strip(),
                "form_params": form_params,
            }
            pipeline = {
                "reference": {
                    "processed": ro["processed"],
                    "ridges": ro["ridges"],
                    "skeleton": ro["skeleton"],
                    "minutiae_vis": ro.get("vis_minutiae"),
                    "quality_map": ro.get("quality_map"),
                    "white_pre": ro["white_pre"],
                    "white_ridges": ro["white_ridges"],
                    "white_skel": ro["white_skel"],
                    "n_min": ro["minutiae_count"],
                },
                "query": {
                    "processed": rp["processed"],
                    "ridges": rp["ridges"],
                    "skeleton": rp["skeleton"],
                    "minutiae_vis": rp.get("vis_minutiae"),
                    "quality_map": rp.get("quality_map"),
                    "white_pre": rp["white_pre"],
                    "white_ridges": rp["white_ridges"],
                    "white_skel": rp["white_skel"],
                    "n_min": rp["minutiae_count"],
                },
                "matches_vis": None,
            }
            report_path = generate_report(ro["skeleton"], rp["skeleton"], match_result, audit=audit, pipeline=pipeline)
            if report_path:
                try:
                    _ensure_pdf_from_html(Path(report_path))
                except Exception as pdf_err:
                    logger.warning("Auto PDF generation failed (inconclusive path): %s", pdf_err)
            report_rel = str(Path(report_path).relative_to(OUTPUT_DIR)).replace("\\", "/") if report_path else None
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
                    "audit": {
                        "sha256_original": sha_o,
                        "sha256_partial": sha_p,
                    },
                    "forensic_quality_warning": True,
                }
            )
            return

        yield _sse_line({"type": "log", "message": "جاري بحث المحاذاة والمطابقة…"})
        if min(len(mo), len(mp)) > 250:
            yield _sse_line(
                {
                    "type": "log",
                    "message": "عدد النقاط الدقيقة مرتفع — هذه الخطوة CPU‑intensive وقد تستغرق عشرات الثواني؛ الصفحة لم تتجمد.",
                }
            )
        sk_o = ro["skeleton"]
        sk_p = rp["skeleton"]
        match_result = enrich_match_for_forensics(
            match_fingerprints_with_partial_alignment(mo, mp, sk_o.shape)
        )
        yield _sse_line(
            {"type": "log", "message": "اكتمل بحث المحاذاة — جاري تجهيز تصور المطابقة…"}
        )
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

        # 10. طبقة التحقق الثانية: ORB Matching
        yield _sse_line({"type": "log", "message": "جاري التحقق البصري المستقل (ORB)…"})
        try:
            orb_res = match_with_orb(ro["processed"], rp["processed"])
            if orb_res.get("visualization") is not None:
                orb_res["orb_visualization"] = _img_data_uri(orb_res["visualization"])
                yield _sse_line({
                    "type": "image",
                    "branch": "orb",
                    "stage": "orb_vis",
                    "src": orb_res["orb_visualization"],
                })
                # إزالة مصفوفة البكسلات لجعل الكائن JSON serializable
                del orb_res["visualization"]
            
            verdict = combined_verdict(
                match_result["match_score"], 
                orb_res["orb_confidence"],
                mcc_score=match_result.get("mcc_score", 0.0),
                orb_score=orb_res.get("orb_score", 0.0),
                partial_verify=bool(match_result.get("partial_verify")),
                matched_points=int(match_result.get("matched_points") or 0),
                alignment_gain_matches=int(match_result.get("alignment_gain_matches") or 0),
                total_original=int(match_result.get("total_original") or 0),
                total_partial=int(match_result.get("total_partial") or 0),
            )
            match_result.update(orb_res)
            match_result.update(verdict)
            if verdict.get("decision_status"):
                match_result["status"] = verdict["decision_status"]
            match_result = enrich_match_for_forensics(match_result)
        except Exception as orb_err:
            logger.error(f"ORB matching failed: {orb_err}")
            yield _sse_line({"type": "log", "message": "فشل التحقق البصري (ORB) - سيتم الاعتماد على النقاط الدقيقة فقط."})

        form_params = {
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
        _apply_partial_verify_step_audit(form_params, match_result)
        audit = {
            "sha256_original": sha_o,
            "sha256_partial": sha_p,
            "operator_name": operator_name.strip(),
            "case_reference": case_reference.strip(),
            "form_params": form_params,
        }

        yield _sse_line({"type": "log", "message": "جاري توليد التقرير…"})
        pipeline = {
            "reference": {
                "processed": ro["processed"],
                "ridges": ro["ridges"],
                "skeleton": ro["skeleton"],
                "minutiae_vis": ro.get("vis_minutiae"),
                "quality_map": ro.get("quality_map"),
                "singular_vis": visualize_singular_points(ro["processed"], ro.get("cores", []), ro.get("deltas", [])) if ro.get("cores") or ro.get("deltas") else None,
                "white_pre": ro["white_pre"],
                "white_ridges": ro["white_ridges"],
                "white_skel": ro["white_skel"],
                "n_min": ro["minutiae_count"],
            },
            "query": {
                "processed": rp["processed"],
                "ridges": rp["ridges"],
                "skeleton": rp["skeleton"],
                "minutiae_vis": rp.get("vis_minutiae"),
                "quality_map": rp.get("quality_map"),
                "singular_vis": visualize_singular_points(rp["processed"], rp.get("cores", []), rp.get("deltas", [])) if rp.get("cores") or rp.get("deltas") else None,
                "white_pre": rp["white_pre"],
                "white_ridges": rp["white_ridges"],
                "white_skel": rp["white_skel"],
                "n_min": rp["minutiae_count"],
            },
            "matches_vis": matches_vis,
        }
        report_path = generate_report(sk_o, sk_p, match_result, audit=audit, pipeline=pipeline)
        if report_path:
            try:
                _ensure_pdf_from_html(Path(report_path))
            except Exception as pdf_err:
                logger.warning("Auto PDF generation failed (stream path): %s", pdf_err)
        report_rel = str(Path(report_path).relative_to(OUTPUT_DIR)).replace("\\", "/") if report_path else None

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
                "audit": {
                    "sha256_original": sha_o,
                    "sha256_partial": sha_p,
                },
                "forensic_quality_warning": low_n < MIN_MINUTIAE_RECOMMENDED,
            }
        )
    except Exception as e:
        logger.exception(e)
        yield _sse_line({"type": "fatal", "message": str(e)})


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
    o_gray = _apply_manual_transform(
        o_gray,
        original_zoom,
        0,
        0,
        apply_preview_scale,
    )
    p_gray = _apply_manual_transform(
        p_gray,
        partial_zoom,
        partial_shift_x,
        partial_shift_y,
        apply_preview_scale,
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


def run_matching_pipeline(
    ro,
    rp,
    sha_o,
    sha_p,
    dm,
    form_ctx,
    operator_name,
    case_reference,
    border_margin,
    min_distance,
    min_contrast,
    min_angle_diff,
    fast_denoise_h,
    gauss_ksize,
    *,
    write_report_and_audit: bool = True,
    quality_gate_enabled: bool = True,
):
    mo, mp = ro.get("minutiae") or [], rp.get("minutiae") or []
    
    q_ref = float(ro.get("quality_score") or 0.0)
    q_qry = float(rp.get("quality_score") or 0.0)
    low_q = min(q_ref, q_qry)
    low_m = min(len(mo), len(mp))
    if quality_gate_enabled and (low_q < QUALITY_GATE_MIN_SCORE or low_m < QUALITY_GATE_MIN_MINUTIAE):
        reason = (
            f"Quality Gate: quality={low_q:.1f} (min {QUALITY_GATE_MIN_SCORE:.1f}) "
            f"or minutiae={low_m} (min {QUALITY_GATE_MIN_MINUTIAE})"
        )
        match_result = enrich_match_for_forensics(_make_inconclusive_result(ro, rp, reason))
        form_params = dict(form_ctx)
        form_params["MATCH_DISTANCE_THRESHOLD"] = MATCH_DISTANCE_THRESHOLD
        form_params["MATCH_ANGLE_THRESHOLD_DEG"] = MATCH_ANGLE_THRESHOLD_DEG
        form_params["PARTIAL_VERIFY_SEARCH_RADIUS"] = PARTIAL_VERIFY_SEARCH_RADIUS
        form_params["QUALITY_GATE_MIN_SCORE"] = QUALITY_GATE_MIN_SCORE
        form_params["QUALITY_GATE_MIN_MINUTIAE"] = QUALITY_GATE_MIN_MINUTIAE
        _apply_partial_verify_step_audit(form_params, match_result)
        audit = {
            "sha256_original": sha_o,
            "sha256_partial": sha_p,
            "operator_name": operator_name.strip(),
            "case_reference": case_reference.strip(),
            "form_params": form_params,
        }
        pipeline = {
            "reference": {
                "processed": ro["processed"],
                "ridges": ro["ridges"],
                "skeleton": ro["skeleton"],
                "minutiae_vis": ro.get("vis_minutiae"),
                "quality_map": ro.get("quality_map"),
                "white_pre": ro["white_pre"],
                "white_ridges": ro["white_ridges"],
                "white_skel": ro["white_skel"],
                "n_min": ro["minutiae_count"],
            },
            "query": {
                "processed": rp["processed"],
                "ridges": rp["ridges"],
                "skeleton": rp["skeleton"],
                "minutiae_vis": rp.get("vis_minutiae"),
                "quality_map": rp.get("quality_map"),
                "white_pre": rp["white_pre"],
                "white_ridges": rp["white_ridges"],
                "white_skel": rp["white_skel"],
                "n_min": rp["minutiae_count"],
            },
            "matches_vis": None,
        }
        report_path = None
        report_rel = None
        if write_report_and_audit:
            report_path = generate_report(ro["skeleton"], rp["skeleton"], match_result, audit=audit, pipeline=pipeline)
            if report_path:
                try:
                    _ensure_pdf_from_html(Path(report_path))
                except Exception as pdf_err:
                    logger.warning("Auto PDF generation failed (form inconclusive path): %s", pdf_err)
            report_rel = str(Path(report_path).relative_to(OUTPUT_DIR)).replace("\\", "/") if report_path else None
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
        return match_result, None, report_rel, audit, True

    sk_o = ro["skeleton"]
    sk_p = rp["skeleton"]
    match_result = enrich_match_for_forensics(
        match_fingerprints_with_partial_alignment(mo, mp, sk_o.shape)
    )
    matches_vis = visualize_alignment_on_reference(sk_o, match_result)
    if matches_vis is None:
        matches_vis = visualize_matches(sk_o, sk_p, match_result)

    # طبقة ORB + قرار دمج نهائي (مماثل لمسار البث المباشر)
    try:
        orb_res = match_with_orb(ro["processed"], rp["processed"])
        if orb_res.get("visualization") is not None:
            orb_res["orb_visualization"] = _img_data_uri(orb_res["visualization"])
            del orb_res["visualization"]
        verdict = combined_verdict(
            match_result["match_score"],
            orb_res["orb_confidence"],
            mcc_score=match_result.get("mcc_score", 0.0),
            orb_score=orb_res.get("orb_score", 0.0),
            partial_verify=bool(match_result.get("partial_verify")),
            matched_points=int(match_result.get("matched_points") or 0),
            alignment_gain_matches=int(match_result.get("alignment_gain_matches") or 0),
            total_original=int(match_result.get("total_original") or 0),
            total_partial=int(match_result.get("total_partial") or 0),
        )
        match_result.update(orb_res)
        match_result.update(verdict)
        if verdict.get("decision_status"):
            match_result["status"] = verdict["decision_status"]
        match_result = enrich_match_for_forensics(match_result)
    except Exception as orb_err:
        logger.error("ORB matching failed (form path): %s", orb_err)

    form_params = dict(form_ctx)
    form_params["MATCH_DISTANCE_THRESHOLD"] = MATCH_DISTANCE_THRESHOLD
    form_params["MATCH_ANGLE_THRESHOLD_DEG"] = MATCH_ANGLE_THRESHOLD_DEG
    form_params["PARTIAL_VERIFY_SEARCH_RADIUS"] = PARTIAL_VERIFY_SEARCH_RADIUS
    _apply_partial_verify_step_audit(form_params, match_result)

    audit = {
        "sha256_original": sha_o,
        "sha256_partial": sha_p,
        "operator_name": operator_name.strip(),
        "case_reference": case_reference.strip(),
        "form_params": form_params,
    }
    pipeline = {
        "reference": {
            "processed": ro["processed"],
            "ridges": ro["ridges"],
            "skeleton": ro["skeleton"],
            "minutiae_vis": ro.get("vis_minutiae"),
            "quality_map": ro.get("quality_map"),
            "singular_vis": visualize_singular_points(ro["processed"], ro.get("cores", []), ro.get("deltas", [])) if ro.get("cores") or ro.get("deltas") else None,
            "white_pre": ro["white_pre"],
            "white_ridges": ro["white_ridges"],
            "white_skel": ro["white_skel"],
            "n_min": ro["minutiae_count"],
        },
        "query": {
            "processed": rp["processed"],
            "ridges": rp["ridges"],
            "skeleton": rp["skeleton"],
            "minutiae_vis": rp.get("vis_minutiae"),
            "quality_map": rp.get("quality_map"),
            "singular_vis": visualize_singular_points(rp["processed"], rp.get("cores", []), rp.get("deltas", [])) if rp.get("cores") or rp.get("deltas") else None,
            "white_pre": rp["white_pre"],
            "white_ridges": rp["white_ridges"],
            "white_skel": rp["white_skel"],
            "n_min": rp["minutiae_count"],
        },
        "matches_vis": matches_vis,
    }
    report_path = None
    report_rel = None
    if write_report_and_audit:
        report_path = generate_report(sk_o, sk_p, match_result, audit=audit, pipeline=pipeline)
        if report_path:
            try:
                _ensure_pdf_from_html(Path(report_path))
            except Exception as pdf_err:
                logger.warning("Auto PDF generation failed (form path): %s", pdf_err)
        report_rel = str(Path(report_path).relative_to(OUTPUT_DIR)).replace("\\", "/") if report_path else None

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
    forensic_quality_warning = low_n < MIN_MINUTIAE_RECOMMENDED
    return match_result, matches_vis, report_rel, audit, forensic_quality_warning


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
) -> dict[str, Any]:
    """Auto-search best manual transform around current partial parameters."""
    o_gray = _decode_upload_type(o_raw)
    p_gray = _decode_upload_type(p_raw)
    dm = denoise_method if denoise_method in ("None", "fastNlMeans", "GaussianBlur") else "fastNlMeans"

    base_ref = _apply_manual_transform(o_gray, original_zoom, 0, 0, apply_preview_scale)
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
                qry = _apply_manual_transform(p_gray, z, sx, sy, apply_preview_scale)
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

                mr = match_fingerprints_with_partial_alignment(mo, mp, ro["skeleton"].shape)
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
