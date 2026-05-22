"""
Primary fingerprint comparison engine (replaces non-existent pip fingerprint-matcher).
Uses core pre-alignment + partial grid/RANSAC matching + MCC from utils.matcher.
"""

from __future__ import annotations

import os
from typing import Any

from utils.matcher import match_fingerprints_with_partial_alignment


def match_threshold() -> float:
    raw = (os.getenv("MATCH_ENGINE_THRESHOLD") or "40").strip()
    try:
        return float(raw)
    except ValueError:
        return 40.0


class FingerprintMatcher:
    """
    Drop-in for the planned FingerprintMatcher library:
    compare_fingerprints → similarity 0–100 and is_match vs threshold (default 40).
    """

    def __init__(self, threshold: float | None = None):
        self.threshold = float(threshold if threshold is not None else match_threshold())

    def compare_fingerprints(
        self,
        minutiae_ref: list[dict[str, Any]],
        minutiae_qry: list[dict[str, Any]],
        image_shape: tuple[int, ...],
        *,
        cores_ref: list[dict[str, Any]] | None = None,
        cores_qry: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[float, bool, dict[str, Any]]:
        result = match_fingerprints_with_partial_alignment(
            minutiae_ref,
            minutiae_qry,
            image_shape,
            cores_ref=cores_ref,
            cores_qry=cores_qry,
            **kwargs,
        )
        score = float(result.get("match_score") or 0.0)
        fused = float(result.get("fused_score") or score)
        # Prefer fused when present (set later in pipeline); here use minutiae match_score
        similarity = max(score, fused) if fused > 0 else score
        is_match = similarity >= self.threshold
        result["engine_similarity"] = round(similarity, 2)
        result["engine_is_match"] = is_match
        result["engine_threshold"] = self.threshold
        return similarity, is_match, result


def compare_fingerprints(
    minutiae_ref: list[dict[str, Any]],
    minutiae_qry: list[dict[str, Any]],
    image_shape: tuple[int, ...],
    **kwargs: Any,
) -> dict[str, Any]:
    """Module-level helper used by matching.matcher.Matcher."""
    _, _, result = FingerprintMatcher().compare_fingerprints(
        minutiae_ref, minutiae_qry, image_shape, **kwargs
    )
    return result
