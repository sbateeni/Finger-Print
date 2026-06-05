"""
Primary fingerprint comparison engine (replaces non-existent pip fingerprint-matcher).
Uses core pre-alignment + partial grid/RANSAC matching + MCC from utils.matcher.

Phase 1 Enhancement: Fingerprint type/region classification check
- Classifies both fingerprints (reference and query)
- Checks compatibility BEFORE minutiae matching
- Returns immediate REJECT if types don't match
"""

from __future__ import annotations

import os
from typing import Any

from utils.matcher import match_fingerprints_with_partial_alignment
from features.fingerprint_classifier import (
    classify_fingerprint,
    FingerprintClassification,
    check_fingerprints_compatible,
)


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
    
    NEW: Phase 1 - Type/region classification check
    - Classifies both fingerprints if classification not provided
    - Rejects immediately if types/regions don't match
    - Proceeds to minutiae matching only if compatible
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
        image_ref: Any = None,  # For auto-classification
        image_qry: Any = None,  # For auto-classification
        classification_ref: FingerprintClassification | None = None,  # Pre-computed reference classification
        classification_qry: FingerprintClassification | None = None,  # Pre-computed query classification
        metadata_ref: dict[str, Any] | None = None,
        metadata_qry: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[float, bool, dict[str, Any]]:
        """
        Compare two fingerprints with Phase 1 classification check.
        
        Args:
            minutiae_ref, minutiae_qry: Minutiae points
            image_shape: Shape of reference image
            cores_ref, cores_qry: Core points (optional)
            image_ref, image_qry: Images for auto-classification (optional)
            classification_ref, classification_qry: Pre-computed classifications (optional)
            metadata_ref, metadata_qry: Metadata hints for classification
            **kwargs: Additional args passed to minutiae matcher
            
        Returns:
            (similarity, is_match, result_dict)
        """
        result = {
            "engine_similarity": 0.0,
            "engine_is_match": False,
            "engine_threshold": self.threshold,
            "classification_check": {
                "performed": False,
                "ref_classification": None,
                "qry_classification": None,
                "compatible": True,
                "reason": "No classification check",
            },
        }
        
        # ============================================================
        # PHASE 1: Classification Compatibility Check
        # ============================================================
        
        # Get or compute classifications
        if classification_ref is None:
            classification_ref = classify_fingerprint(
                image=image_ref,
                minutiae=minutiae_ref,
                image_shape=image_shape,
                metadata=metadata_ref,
            )
        
        if classification_qry is None:
            classification_qry = classify_fingerprint(
                image=image_qry,
                minutiae=minutiae_qry,
                image_shape=image_shape,
                metadata=metadata_qry,
            )
        
        # Store classifications in result
        result["classification_check"]["performed"] = True
        result["classification_check"]["ref_classification"] = classification_ref.to_dict()
        result["classification_check"]["qry_classification"] = classification_qry.to_dict()
        
        # Check compatibility
        is_compatible, compatibility_reason = check_fingerprints_compatible(
            classification_ref, classification_qry
        )
        
        result["classification_check"]["compatible"] = is_compatible
        result["classification_check"]["reason"] = compatibility_reason
        
        # If incompatible, return immediately (NO match possible)
        if not is_compatible:
            result["match_score"] = 0.0
            result["fused_score"] = 0.0
            result["matched_points"] = 0
            result["engine_similarity"] = 0.0
            result["engine_is_match"] = False
            result["rejection_reason"] = f"Classification incompatibility: {compatibility_reason}"
            return 0.0, False, result
        
        # ============================================================
        # Classification check PASSED - proceed to minutiae matching
        # ============================================================
        
        match_result = match_fingerprints_with_partial_alignment(
            minutiae_ref,
            minutiae_qry,
            image_shape,
            cores_ref=cores_ref,
            cores_qry=cores_qry,
            **kwargs,
        )
        
        # Merge results
        result.update(match_result)
        
        score = float(result.get("match_score") or 0.0)
        fused = float(result.get("fused_score") or score)
        similarity = max(score, fused) if fused > 0 else score
        is_match = similarity >= self.threshold
        
        result["engine_similarity"] = round(similarity, 2)
        result["engine_is_match"] = is_match
        
        return similarity, is_match, result


def compare_fingerprints(
    minutiae_ref: list[dict[str, Any]],
    minutiae_qry: list[dict[str, Any]],
    image_shape: tuple[int, ...],
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Module-level helper used by matching.matcher.Matcher.
    
    NOW INCLUDES Phase 1 classification check.
    Pass image_ref, image_qry, metadata_ref, metadata_qry if available for better classification.
    """
    _, _, result = FingerprintMatcher().compare_fingerprints(
        minutiae_ref, minutiae_qry, image_shape, **kwargs
    )
    return result
