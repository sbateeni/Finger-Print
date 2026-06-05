"""
Enhanced Fingerprint Matching with Landmarks - Phase 4

Updates to compare_engine to:
1. Use anatomical landmarks in matching
2. Weight high-importance landmarks more heavily
3. Calculate landmark-based similarity
4. Combine with minutiae-based similarity
"""

from __future__ import annotations

from typing import Any
import numpy as np
from collections import Counter

from features.minutiae_taxonomy import (
    get_landmark_by_name,
    is_high_importance_landmark,
)


class LandmarkMatcher:
    """
    Enhanced matching algorithm that considers anatomical landmarks.
    """

    def __init__(self, high_importance_weight: float = 1.5):
        """
        Initialize landmark matcher.
        
        Args:
            high_importance_weight: Weight multiplier for high-importance landmarks
        """
        self.high_importance_weight = high_importance_weight

    def compare_landmarks(
        self,
        minutiae_ref: list[dict[str, Any]],
        minutiae_qry: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Compare landmarks between two sets of minutiae.
        
        Args:
            minutiae_ref: Reference fingerprint minutiae
            minutiae_qry: Query fingerprint minutiae
            
        Returns:
            {
                "landmark_similarity": float (0-100),
                "ref_landmark_distribution": dict,
                "qry_landmark_distribution": dict,
                "matching_landmarks": list,
                "landmark_details": dict,
            }
        """
        # Extract landmark types
        ref_types = [m.get("landmark_type", "unknown") for m in minutiae_ref]
        qry_types = [m.get("landmark_type", "unknown") for m in minutiae_qry]

        # Count landmark occurrences
        ref_counts = Counter(ref_types)
        qry_counts = Counter(qry_types)

        # Calculate landmark-based similarity
        similarity = self._calculate_landmark_similarity(ref_counts, qry_counts)

        # Find matching landmarks
        matching = self._find_matching_landmarks(ref_counts, qry_counts)

        # Get distribution details
        ref_dist = self._get_landmark_distribution(ref_counts, len(minutiae_ref))
        qry_dist = self._get_landmark_distribution(qry_counts, len(minutiae_qry))

        return {
            "landmark_similarity": round(similarity, 2),
            "ref_landmark_distribution": ref_dist,
            "qry_landmark_distribution": qry_dist,
            "matching_landmarks": matching,
            "landmark_details": {
                "ref_types": ref_counts,
                "qry_types": qry_counts,
                "total_ref": len(minutiae_ref),
                "total_qry": len(minutiae_qry),
            },
        }

    def _calculate_landmark_similarity(
        self,
        ref_counts: Counter,
        qry_counts: Counter,
    ) -> float:
        """
        Calculate similarity based on landmark distribution.
        
        Higher weight for high-importance landmarks.
        """
        all_types = set(ref_counts.keys()) | set(qry_counts.keys())

        if not all_types:
            return 0.0

        weighted_similarity = 0.0
        total_weight = 0.0

        for landmark_type in all_types:
            ref_count = ref_counts.get(landmark_type, 0)
            qry_count = qry_counts.get(landmark_type, 0)

            # Calculate similarity for this landmark type
            total = max(ref_count, qry_count, 1)
            same = min(ref_count, qry_count)
            type_similarity = (same / total) * 100

            # Apply weight based on importance
            weight = 1.0
            if is_high_importance_landmark(landmark_type):
                weight = self.high_importance_weight

            weighted_similarity += type_similarity * weight
            total_weight += weight

        return weighted_similarity / total_weight if total_weight > 0 else 0.0

    def _find_matching_landmarks(
        self,
        ref_counts: Counter,
        qry_counts: Counter,
    ) -> list[dict[str, Any]]:
        """
        Find landmarks that appear in both distributions.
        """
        matching = []
        all_types = set(ref_counts.keys()) | set(qry_counts.keys())

        for landmark_type in all_types:
            ref_count = ref_counts.get(landmark_type, 0)
            qry_count = qry_counts.get(landmark_type, 0)

            if ref_count > 0 and qry_count > 0:
                matching.append({
                    "landmark_type": landmark_type,
                    "ref_count": ref_count,
                    "qry_count": qry_count,
                    "min_count": min(ref_count, qry_count),
                    "importance": get_landmark_by_name(landmark_type).get("forensic_importance", "unknown") if get_landmark_by_name(landmark_type) else "unknown",
                })

        return matching

    def _get_landmark_distribution(
        self,
        counts: Counter,
        total: int,
    ) -> dict[str, Any]:
        """Get distribution percentages."""
        dist = {}
        for landmark_type, count in counts.items():
            dist[landmark_type] = {
                "count": count,
                "percentage": round((count / total * 100), 2) if total > 0 else 0,
                "importance": get_landmark_by_name(landmark_type).get("forensic_importance", "unknown") if get_landmark_by_name(landmark_type) else "unknown",
            }
        return dist


def calculate_combined_similarity(
    minutiae_score: float,
    landmark_score: float,
    minutiae_weight: float = 0.7,
    landmark_weight: float = 0.3,
) -> float:
    """
    Combine minutiae matching score with landmark matching score.
    
    Args:
        minutiae_score: Score from minutiae matching (0-100)
        landmark_score: Score from landmark matching (0-100)
        minutiae_weight: Weight for minutiae (default 0.7)
        landmark_weight: Weight for landmarks (default 0.3)
        
    Returns:
        Combined similarity score (0-100)
    """
    total_weight = minutiae_weight + landmark_weight
    combined = (minutiae_score * minutiae_weight + landmark_score * landmark_weight) / total_weight
    return round(combined, 2)


# Global matcher instance
_landmark_matcher = LandmarkMatcher()


def compare_landmarks(
    minutiae_ref: list[dict[str, Any]],
    minutiae_qry: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Convenient function to compare landmarks.
    """
    return _landmark_matcher.compare_landmarks(minutiae_ref, minutiae_qry)
