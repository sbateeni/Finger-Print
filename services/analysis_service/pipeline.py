"""Matching, fusion, report generation, and audit for form/Telegram paths."""

from __future__ import annotations

import logging
from pathlib import Path

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
from utils.fusion import apply_fusion_to_match
from utils.matcher import (
    match_fingerprints_with_partial_alignment,
    visualize_alignment_on_reference,
    visualize_matches,
)
from utils.report_generator import generate_report

from .reports import _ensure_pdf_from_html
from .results import (
    _apply_partial_verify_step_audit,
    _make_inconclusive_result,
    build_report_pipeline,
)

logger = logging.getLogger(__name__)


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
    report_lang = str((form_ctx or {}).get("report_lang") or "ar")
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
            "report_lang": report_lang,
            "form_params": form_params,
        }
        pipeline = build_report_pipeline(ro, rp, matches_vis=None, include_singular=False)
        report_path = None
        report_rel = None
        if write_report_and_audit:
            report_path = generate_report(
                ro["skeleton"],
                rp["skeleton"],
                match_result,
                audit=audit,
                pipeline=pipeline,
                lang=report_lang,
            )
            if report_path:
                try:
                    _ensure_pdf_from_html(Path(report_path))
                except Exception as pdf_err:
                    logger.warning("Auto PDF generation failed (form inconclusive path): %s", pdf_err)
            report_rel = (
                str(Path(report_path).relative_to(OUTPUT_DIR)).replace("\\", "/")
                if report_path
                else None
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
        return match_result, None, report_rel, audit, True

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
    matches_vis = visualize_alignment_on_reference(sk_o, match_result)
    if matches_vis is None:
        matches_vis = visualize_matches(sk_o, sk_p, match_result)

    try:
        match_result = apply_fusion_to_match(match_result, ro["processed"], rp["processed"])
        match_result = enrich_match_for_forensics(match_result)
    except Exception as fusion_err:
        logger.error("Fusion failed (form path): %s", fusion_err)

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
        "report_lang": report_lang,
        "form_params": form_params,
    }
    pipeline = build_report_pipeline(ro, rp, matches_vis=matches_vis, include_singular=True)
    report_path = None
    report_rel = None
    if write_report_and_audit:
        report_path = generate_report(
            sk_o, sk_p, match_result, audit=audit, pipeline=pipeline, lang=report_lang
        )
        if report_path:
            try:
                _ensure_pdf_from_html(Path(report_path))
            except Exception as pdf_err:
                logger.warning("Auto PDF generation failed (form path): %s", pdf_err)
        report_rel = (
            str(Path(report_path).relative_to(OUTPUT_DIR)).replace("\\", "/") if report_path else None
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
    forensic_quality_warning = low_n < MIN_MINUTIAE_RECOMMENDED
    return match_result, matches_vis, report_rel, audit, forensic_quality_warning
