"""
Bozorth3 scoring bridge for the main fusion pipeline.
"""

from __future__ import annotations

from typing import Any, List

from config.config import (
    BOZORTH_MATCH_THRESHOLD,
    BOZORTH_SCORE_NORMALIZE_MAX,
    USE_BOZORTH_MATCHER,
)


def bozorth_score_to_percent(raw_score: float) -> float:
    """Map raw Bozorth-like score to 0–100."""
    cap = float(BOZORTH_SCORE_NORMALIZE_MAX) or 60.0
    if cap <= 0:
        return 0.0
    return max(0.0, min(100.0, (float(raw_score) / cap) * 100.0))


def compute_bozorth_pair(
    minutiae_ref: List[dict],
    minutiae_query: List[dict],
    *,
    force: bool = False,
) -> dict[str, Any]:
    """
    Compare two minutiae lists. Returns dict with raw score, normalized %, and match flag.
    """
    empty = {
        "bozorth_score": 0.0,
        "bozorth_score_pct": 0.0,
        "bozorth_match": False,
        "bozorth_enabled": False,
    }
    if not force and not USE_BOZORTH_MATCHER:
        return empty
    if not minutiae_ref or not minutiae_query:
        return {**empty, "bozorth_enabled": True}

    from matching.bozorth_matcher import BozorthMatcher

    matcher = BozorthMatcher(match_threshold=BOZORTH_MATCH_THRESHOLD)
    raw, matched = matcher.compare_fingerprints(minutiae_ref, minutiae_query)
    pct = bozorth_score_to_percent(raw)
    return {
        "bozorth_score": round(float(raw), 2),
        "bozorth_score_pct": round(pct, 2),
        "bozorth_match": bool(matched),
        "bozorth_enabled": True,
    }
