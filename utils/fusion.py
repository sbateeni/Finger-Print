"""
Match fusion: Minutiae + MCC (+ optional ORB).
"""

from __future__ import annotations

import os
from typing import Any

from utils.orb_matcher import combined_verdict, match_with_orb
from matching.landmark_matcher import LandmarkMatcher


def use_orb_fusion() -> bool:
    return (os.getenv("USE_ORB_FUSION") or "0").strip().lower() in ("1", "true", "yes", "on")


def apply_fusion_to_match(
    match_result: dict[str, Any],
    ro: dict[str, Any],
    rp: dict[str, Any],
) -> dict[str, Any]:
    """
    Run ORB (optional), Landmark Matcher, and combined verdict; merge into match_result.
    """
    ref_processed = ro.get("processed")
    query_processed = rp.get("processed")
    
    mcc_score = float(match_result.get("mcc_score") or 0.0)
    orb_res: dict[str, Any] = {
        "orb_matches": 0,
        "orb_score": 0.0,
        "orb_confidence": "INSUFFICIENT",
    }

    if use_orb_fusion():
        try:
            orb_res = match_with_orb(ref_processed, query_processed)
            if orb_res.get("visualization") is not None:
                del orb_res["visualization"]
        except Exception:
            pass

    # 1. Landmark Matching (Phase 4)
    landmark_matcher = LandmarkMatcher()
    landmark_res = landmark_matcher.compare_landmarks(
        ro.get("minutiae") or [],
        rp.get("minutiae") or []
    )
    
    match_result.update(landmark_res)
    landmark_score = float(landmark_res.get("landmark_similarity") or 0.0)

    verdict = combined_verdict(
        float(match_result.get("match_score") or 0.0),
        orb_res.get("orb_confidence", "INSUFFICIENT"),
        mcc_score=mcc_score,
        orb_score=float(orb_res.get("orb_score") or 0.0),
        landmark_score=landmark_score,
        partial_verify=bool(match_result.get("partial_verify")),
        matched_points=int(match_result.get("matched_points") or 0),
        alignment_gain_matches=int(match_result.get("alignment_gain_matches") or 0),
        total_original=int(match_result.get("total_original") or 0),
        total_partial=int(match_result.get("total_partial") or 0),
        use_orb=use_orb_fusion(),
    )
    match_result.update(orb_res)
    match_result.update(verdict)
    match_result["orb_fusion_enabled"] = use_orb_fusion()
    if verdict.get("decision_status"):
        match_result["status"] = verdict["decision_status"]
    return match_result
