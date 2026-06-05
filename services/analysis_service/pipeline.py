"""Matching, fusion, report generation, and audit for form/Telegram paths."""

from __future__ import annotations

import logging
import os
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
from features.fingerprint_classifier import FingerprintClassifier, FingerprintClassification
from features.minutiae_landmarks import extract_landmarks, landmark_statistics

from database import SessionLocal
from database import crud

from .reports import should_generate_pdf, _ensure_pdf_from_html
from .results import (
    _apply_partial_verify_step_audit,
    _make_inconclusive_result,
    build_report_pipeline,
)

logger = logging.getLogger(__name__)


def _handle_inconclusive_pipeline(
    ro, rp, sha_o, sha_p, match_result, form_ctx, 
    operator_name, case_reference, report_lang, 
    write_report_and_audit
):
    form_params = dict(form_ctx or {})
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
        if report_path and should_generate_pdf(Path(report_path), lang=report_lang):
            try:
                _ensure_pdf_from_html(Path(report_path), lang=report_lang)
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

    # 1. Classification Gate (Phase 1)
    classifier = FingerprintClassifier()
    # Assume processed images are available in 'processed' key
    img_o = ro.get("processed")
    img_p = rp.get("processed")
    
    # Metadata can contain finger_type hints from form
    meta_o = {"finger_type": (form_ctx or {}).get("original_finger_type")}
    meta_p = {"finger_type": (form_ctx or {}).get("partial_finger_type")}
    
    class_o = classifier.classify(image=img_o, minutiae=mo, metadata=meta_o,
                                   cores=ro.get("cores"), deltas=ro.get("deltas"))
    class_p = classifier.classify(image=img_p, minutiae=mp, metadata=meta_p,
                                   cores=rp.get("cores"), deltas=rp.get("deltas"))
    
    # Store classification in results for report
    ro["classification"] = class_o.to_dict()
    rp["classification"] = class_p.to_dict()
    
    # Ensure landmarks are extracted (Phase 2)
    if "landmarks" not in ro or not ro.get("minutiae") or "landmark_type" not in ro["minutiae"][0]:
        ro["minutiae"] = extract_landmarks(ro.get("minutiae") or [], image=img_o, ridge_image=ro.get("ridges"))
        ro["landmarks"] = landmark_statistics(ro["minutiae"])
    
    if "landmarks" not in rp or not rp.get("minutiae") or "landmark_type" not in rp["minutiae"][0]:
        rp["minutiae"] = extract_landmarks(rp.get("minutiae") or [], image=img_p, ridge_image=rp.get("ridges"))
        rp["landmarks"] = landmark_statistics(rp["minutiae"])
    
    is_compatible, class_reason = class_o.is_compatible_with(class_p)
    if not is_compatible:
        reason = f"Classification Gate: {class_reason}"
        match_result = enrich_match_for_forensics(_make_inconclusive_result(ro, rp, reason))
        match_result["classification_compatible"] = 0
        match_result["classification_check_reason"] = class_reason
        
        return _handle_inconclusive_pipeline(
            ro, rp, sha_o, sha_p, match_result, form_ctx, 
            operator_name, case_reference, report_lang, 
            write_report_and_audit
        )

    # 2. Quality Gate
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
        return _handle_inconclusive_pipeline(
            ro, rp, sha_o, sha_p, match_result, form_ctx, 
            operator_name, case_reference, report_lang, 
            write_report_and_audit
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
    matches_vis = visualize_alignment_on_reference(sk_o, match_result)
    if matches_vis is None:
        matches_vis = visualize_matches(sk_o, sk_p, match_result)

    try:
        match_result = apply_fusion_to_match(match_result, ro, rp)
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
        if report_path and should_generate_pdf(Path(report_path), lang=report_lang):
            try:
                _ensure_pdf_from_html(Path(report_path), lang=report_lang)
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

    # 3. SQL Storage (Phase 4)
    if os.getenv("WRITE_TO_DB", "1") == "1":
        try:
            # Ensure storage directory exists
            storage_dir = Path("static/fingerprints")
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            with SessionLocal() as db:
                # Save images and create fingerprints
                filename_o = f"{sha_o[:16]}.png"
                if ro.get("processed") is not None:
                    cv2.imwrite(str(storage_dir / filename_o), ro["processed"])
                
                fp_o = crud.create_fingerprint(
                    db,
                    filename=filename_o,
                    filepath=str(storage_dir / filename_o),
                    quality_score=ro.get("quality_score"),
                    minutiae_count=len(ro.get("minutiae") or []),
                    minutiae_data={"minutiae": ro.get("minutiae") or []}
                )
                crud.update_fingerprint_landmarks(db, fp_o.id, ro.get("landmarks") or {})
                if ro.get("classification"):
                    fp_o.fingerprint_classification = ro.get("classification")
                    fp_o.fingerprint_type = ro.get("classification").get("finger_type")
                    fp_o.fingerprint_region = ro.get("classification").get("region")
                    fp_o.fingerprint_pattern = ro.get("classification").get("pattern_type")
                
                filename_p = f"{sha_p[:16]}.png"
                if rp.get("processed") is not None:
                    cv2.imwrite(str(storage_dir / filename_p), rp["processed"])
                
                fp_p = crud.create_fingerprint(
                    db,
                    filename=filename_p,
                    filepath=str(storage_dir / filename_p),
                    quality_score=rp.get("quality_score"),
                    minutiae_count=len(rp.get("minutiae") or []),
                    minutiae_data={"minutiae": rp.get("minutiae") or []}
                )
                crud.update_fingerprint_landmarks(db, fp_p.id, rp.get("landmarks") or {})
                if rp.get("classification"):
                    fp_p.fingerprint_classification = rp.get("classification")
                    fp_p.fingerprint_type = rp.get("classification").get("finger_type")
                    fp_p.fingerprint_region = rp.get("classification").get("region")
                    fp_p.fingerprint_pattern = rp.get("classification").get("pattern_type")
                
                # Create match
                db_match = crud.create_match(
                    db,
                    case_reference=case_reference,
                    operator_name=operator_name,
                    original_fingerprint_id=fp_o.id,
                    partial_fingerprint_id=fp_p.id,
                    match_score=float(match_result.get("match_score") or 0.0),
                    fused_score=float(match_result.get("fused_score") or 0.0),
                    matched_points=int(match_result.get("matched_points") or 0),
                    status=match_result.get("status", "UNKNOWN"),
                    match_details=match_result
                )
                db_match.classification_compatible = match_result.get("classification_compatible", 1)
                db_match.classification_check_reason = match_result.get("classification_check_reason")
                db.commit()
                
                # Store IDs in match_result for downstream use (e.g. editor link)
                match_result["db_match_id"] = db_match.id
                match_result["db_original_id"] = fp_o.id
                match_result["db_partial_id"] = fp_p.id
        except Exception as db_err:
            logger.error("SQL storage failed: %s", db_err)

    low_n = min(ro["minutiae_count"], rp["minutiae_count"])
    forensic_quality_warning = low_n < MIN_MINUTIAE_RECOMMENDED
    return match_result, matches_vis, report_rel, audit, forensic_quality_warning
