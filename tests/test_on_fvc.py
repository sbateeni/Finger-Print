"""
FVC2004 benchmark harness (optional — requires dataset path).

Download FVC2004 from https://biometrics.cse.msu.edu/fvc2004.html
Set env: FVC2004_PATH=/path/to/FVC2004

Run: pytest tests/test_on_fvc.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

FVC_ROOT = os.getenv("FVC2004_PATH", "")


@pytest.mark.skipif(not FVC_ROOT or not Path(FVC_ROOT).is_dir(), reason="FVC2004_PATH not set")
def test_fvc_smoke_one_pair():
    import cv2
    from preprocessing.preprocess import preprocess_image
    from features.extractor import extract_minutiae
    from utils.image_processing import enhance_image
    from matching.bozorth_matcher import BozorthMatcher

    root = Path(FVC_ROOT)
    images = list(root.rglob("*.bmp")) + list(root.rglob("*.tif"))
    assert len(images) >= 2

    img1 = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(images[1]), cv2.IMREAD_GRAYSCALE)
    p1 = preprocess_image(img1)
    p2 = preprocess_image(img2)
    assert p1 is not None and p2 is not None
    s1 = enhance_image(p1)
    s2 = enhance_image(p2)
    m1 = extract_minutiae(s1)
    m2 = extract_minutiae(s2)
    score, _ = BozorthMatcher().compare_fingerprints(m1, m2)
    assert score >= 0
