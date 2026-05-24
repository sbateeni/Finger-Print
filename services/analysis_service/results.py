"""Match result shaping, audit helpers, report pipeline dicts."""

from __future__ import annotations

from typing import Any

from config import PARTIAL_VERIFY_STEP_PX
from utils.image_utils import _img_data_uri
from utils.minutiae_extractor import visualize_singular_points


def _sanitize_match_for_json(mr: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in mr.items() if k != "matched_details"}


def _apply_partial_verify_step_audit(form_params: dict[str, Any], match_result: dict[str, Any]) -> None:
    """يسجّل خطوة شبكة المحاذاة الفعلية المستخدمة (قد تختلف عن config عند التكييف التلقائي)."""
    eff = match_result.get("partial_verify_step_px_effective")
    cfg = match_result.get("partial_verify_step_px_config")
    form_params["PARTIAL_VERIFY_STEP_PX"] = int(eff) if eff is not None else PARTIAL_VERIFY_STEP_PX
    if cfg is not None and eff is not None and int(cfg) != int(eff):
        form_params["PARTIAL_VERIFY_STEP_PX_configured"] = int(cfg)


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


def _pipeline_side(branch: dict[str, Any], *, include_singular: bool) -> dict[str, Any]:
    side = {
        "processed": branch["processed"],
        "ridges": branch["ridges"],
        "skeleton": branch["skeleton"],
        "minutiae_vis": branch.get("vis_minutiae"),
        "quality_map": branch.get("quality_map"),
        "white_pre": branch["white_pre"],
        "white_ridges": branch["white_ridges"],
        "white_skel": branch["white_skel"],
        "n_min": branch["minutiae_count"],
    }
    if include_singular and (branch.get("cores") or branch.get("deltas")):
        side["singular_vis"] = visualize_singular_points(
            branch["processed"],
            branch.get("cores", []),
            branch.get("deltas", []),
        )
    return side


def build_report_pipeline(
    ro: dict[str, Any],
    rp: dict[str, Any],
    *,
    matches_vis=None,
    include_singular: bool = True,
) -> dict[str, Any]:
    return {
        "reference": _pipeline_side(ro, include_singular=include_singular),
        "query": _pipeline_side(rp, include_singular=include_singular),
        "matches_vis": matches_vis,
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
