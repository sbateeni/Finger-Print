"""
Public matching API — wraps compare_engine (global minutiae matcher).
Legacy scipy helpers kept for backward compatibility.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from matching.compare_engine import FingerprintMatcher, compare_fingerprints, match_threshold
from matching.alignment import align_by_core_point, find_core_point


class Matcher:
    """Facade requested in development plan (FingerprintMatcher + core alignment)."""

    def __init__(self, threshold: float | None = None):
        self._engine = FingerprintMatcher(threshold=threshold)

    def compare_fingerprints(
        self,
        minutiae1: list[dict[str, Any]],
        minutiae2: list[dict[str, Any]],
        image_shape: tuple[int, ...],
        **kwargs: Any,
    ) -> tuple[float, bool]:
        score, is_match, _ = self._engine.compare_fingerprints(
            minutiae1, minutiae2, image_shape, **kwargs
        )
        return score, is_match


def calculate_distance(p1, p2):
    return np.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2)


def calculate_angle_difference(angle1, angle2):
    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)


def match_fingerprints_legacy(original_minutiae, partial_minutiae):
    """Deprecated simple Hungarian matcher — use compare_fingerprints instead."""
    try:
        original_points = np.array([[m["x"], m["y"]] for m in original_minutiae])
        partial_points = np.array([[m["x"], m["y"]] for m in partial_minutiae])
        distances = cdist(original_points, partial_points)
        row_ind, col_ind = linear_sum_assignment(distances)
        matched_points = []
        for i, j in zip(row_ind, col_ind):
            if distances[i, j] < 10:
                matched_points.append(
                    {
                        "original": original_minutiae[i],
                        "partial": partial_minutiae[j],
                        "distance": distances[i, j],
                    }
                )
        n_match = len(matched_points)
        n_p = len(partial_minutiae)
        match_score = (n_match / n_p * 100.0) if n_p > 0 else 0.0
        return {
            "matched_points": n_match,
            "total_original": len(original_minutiae),
            "total_partial": n_p,
            "match_score": match_score,
            "status": "HIGH MATCH" if match_score > 75 else "NO MATCH",
            "details": {"ridge_analysis": []},
        }
    except Exception as e:
        return {
            "matched_points": 0,
            "total_original": len(original_minutiae),
            "total_partial": len(partial_minutiae),
            "match_score": 0,
            "status": "ERROR",
            "details": {"error": str(e)},
        }


# Default export for new code paths
match_fingerprints = compare_fingerprints

__all__ = [
    "Matcher",
    "FingerprintMatcher",
    "compare_fingerprints",
    "match_fingerprints",
    "match_threshold",
    "align_by_core_point",
    "find_core_point",
    "match_fingerprints_legacy",
]
