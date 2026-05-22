"""Smoke tests for global-upgrade modules."""

import numpy as np
import cv2

from preprocessing.enhancer import FingerprintEnhancer
from matching.alignment import align_minutiae
from matching.bozorth_matcher import BozorthMatcher
from evaluation.quality import get_quality_score
from utils.minutiae_extractor import extract_minutiae, _crossing_number


def test_gabor_enhancer_runs():
    img = np.random.randint(40, 200, (128, 128), dtype=np.uint8)
    out = FingerprintEnhancer.enhance(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_crossing_number_endpoint():
    sk = np.zeros((5, 5), dtype=np.uint8)
    sk[2, 1:4] = 1
    assert _crossing_number(sk, 2, 1) == 1


def test_align_minutiae_translation():
    ref = [
        {"x": 10, "y": 10, "type": "endpoint", "angle": 0},
        {"x": 12, "y": 11, "type": "bifurcation", "angle": 0},
        {"x": 50, "y": 50, "type": "endpoint", "angle": 0},
    ]
    qry = [
        {"x": 30, "y": 30, "type": "endpoint", "angle": 0},
        {"x": 31, "y": 29, "type": "bifurcation", "angle": 0},
        {"x": 50, "y": 50, "type": "endpoint", "angle": 0},
    ]
    a1, a2 = align_minutiae(ref, qry)
    assert len(a1) == 3 and len(a2) == 3


def test_bozorth_fallback_score():
    m1 = [{"x": i * 10, "y": 50, "type": "endpoint", "angle": 0} for i in range(8)]
    m2 = [{"x": i * 10 + 1, "y": 51, "type": "endpoint", "angle": 0} for i in range(8)]
    b = BozorthMatcher(match_threshold=10)
    score, ok = b.compare_fingerprints(m1, m2)
    assert score > 0


def test_quality_score_range():
    img = np.full((100, 100), 128, dtype=np.uint8)
    s = get_quality_score(img)
    assert 0 <= s <= 100


def test_extract_minutiae_on_synthetic():
    from utils.image_processing import enhance_image

    binary = np.zeros((64, 64), dtype=np.uint8)
    cv2.line(binary, (10, 32), (54, 32), 255, 2)
    skel = enhance_image(binary)
    assert skel is not None
    pts = extract_minutiae(skel, border_margin=5, min_distance=5)
    assert isinstance(pts, list)
