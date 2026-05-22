"""Partial-print final decision (user-reported scenario)."""

from utils.orb_matcher import combined_verdict


def test_partial_real_case_reaches_medium_match():
    """MCC ~62%, low match_score, strong alignment — should not stay NO MATCH."""
    out = combined_verdict(
        minutiae_score=16.18,
        orb_confidence="INSUFFICIENT",
        mcc_score=62.28,
        orb_score=2.20,
        partial_verify=True,
        matched_points=28,
        alignment_gain_matches=10,
        total_original=182,
        total_partial=143,
        use_orb=False,
    )
    assert out["is_partial_case"] is True
    assert out["decision_mode"] == "partial-first"
    # Without ORB: 0.7×min + 0.3×MCC ≈ 30; partial-first still upgrades via MCC+alignment
    assert out["fused_score"] >= 28.0
    assert out["decision_status"] == "MEDIUM MATCH"
    assert "مراجعة خبير" in out["combined_verdict"]


def test_impostor_stays_no_match():
    out = combined_verdict(
        minutiae_score=5.0,
        orb_confidence="INSUFFICIENT",
        mcc_score=12.0,
        orb_score=1.0,
        partial_verify=True,
        matched_points=3,
        alignment_gain_matches=0,
        total_original=100,
        total_partial=80,
    )
    assert out["decision_status"] in ("NO MATCH", "LOW MATCH")
