"""
Fingerprint quality: NFIQ2 when installed, else built-in coherence score.
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_NFIQ2 = None
try:
    import nfiq2  # type: ignore

    _NFIQ2 = nfiq2
except ImportError:
    pass


def get_quality_score(image: np.ndarray) -> float:
    """
    0–100 (higher is better). Uses NFIQ2 if available.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if _NFIQ2 is not None:
        try:
            if hasattr(_NFIQ2, "compute_quality_score"):
                return float(_NFIQ2.compute_quality_score(gray))
            if hasattr(_NFIQ2, "computeQualityScore"):
                return float(_NFIQ2.computeQualityScore(gray))
        except Exception as e:
            logger.warning("NFIQ2 failed: %s", e)

    from utils.image_processing import assess_fingerprint_quality

    score, _ = assess_fingerprint_quality(gray)
    return float(score)


def quality_before_after(
    raw_gray: np.ndarray, enhanced_gray: np.ndarray
) -> Tuple[float, float]:
    return get_quality_score(raw_gray), get_quality_score(enhanced_gray)
