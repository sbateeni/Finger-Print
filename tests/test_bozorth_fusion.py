"""Bozorth integration in combined verdict."""

from utils.orb_matcher import combined_verdict


def test_fused_includes_bozorth():
    out = combined_verdict(
        minutiae_score=30.0,
        orb_confidence="LOW",
        mcc_score=58.0,
        orb_score=20.0,
        partial_verify=True,
        matched_points=22,
        alignment_gain_matches=5,
        total_original=80,
        total_partial=40,
        bozorth_score_pct=55.0,
        bozorth_match=True,
        bozorth_enabled=True,
    )
    assert out["fusion_components"].get("bozorth_score") is not None
    assert out["bozorth_score_pct"] == 55.0
    assert out["fused_score"] >= 38.0
