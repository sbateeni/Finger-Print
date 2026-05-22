"""
Bozorth3-style scoring with optional external library; fallback to Hungarian minutiae matching.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_BOZORTH_IMPL = None
try:
    from fingerprint_matching import matcher as _fm  # type: ignore

    _BOZORTH_IMPL = _fm
    logger.info("Using fingerprint_matching Bozorth3 wrapper")
except ImportError:
    pass


def _minutiae_to_array(minutiae: List[dict]) -> np.ndarray:
    rows = []
    for m in minutiae:
        t = m.get("type", "endpoint")
        type_code = 1 if t == "endpoint" else 2
        rows.append([m["x"], m["y"], float(m.get("angle", 0)), type_code])
    if not rows:
        return np.zeros((0, 4), dtype=np.float64)
    return np.array(rows, dtype=np.float64)


def _fallback_score(m1: List[dict], m2: List[dict], dist_thresh: float = 15.0) -> float:
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    if not m1 or not m2:
        return 0.0
    p1 = np.array([[m["x"], m["y"]] for m in m1], dtype=np.float64)
    p2 = np.array([[m["x"], m["y"]] for m in m2], dtype=np.float64)
    d = cdist(p1, p2)
    r, c = linear_sum_assignment(d)
    matched = sum(1 for i, j in zip(r, c) if d[i, j] < dist_thresh)
    return float(matched * 4)  # scale ~ Bozorth-like range


class BozorthMatcher:
    def __init__(self, match_threshold: float = 25.0):
        self.match_threshold = match_threshold
        self._engine = None
        if _BOZORTH_IMPL is not None:
            try:
                self._engine = _BOZORTH_IMPL.Bozorth3()
            except Exception as e:
                logger.warning("Bozorth3 init failed: %s", e)

    def extract_template(self, minutiae_list: List[dict]) -> np.ndarray:
        return _minutiae_to_array(minutiae_list)

    def match(self, template1: np.ndarray, template2: np.ndarray) -> float:
        if template1.size == 0 or template2.size == 0:
            return 0.0
        if self._engine is not None:
            try:
                return float(self._engine.match(template1, template2))
            except Exception as e:
                logger.warning("Bozorth match failed: %s", e)
        m1 = [
            {"x": int(t[0]), "y": int(t[1]), "angle": t[2], "type": "endpoint" if t[3] == 1 else "bifurcation"}
            for t in template1
        ]
        m2 = [
            {"x": int(t[0]), "y": int(t[1]), "angle": t[2], "type": "endpoint" if t[3] == 1 else "bifurcation"}
            for t in template2
        ]
        return _fallback_score(m1, m2)

    def compare_fingerprints(self, minutiae1: List[dict], minutiae2: List[dict]) -> tuple[float, bool]:
        from matching.alignment import align_minutiae

        a1, a2 = align_minutiae(minutiae1, minutiae2)
        t1 = self.extract_template(a1)
        t2 = self.extract_template(a2)
        score = self.match(t1, t2)
        return score, score >= self.match_threshold
