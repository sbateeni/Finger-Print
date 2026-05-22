import numpy as np

from matching.alignment import align_by_core_point, find_core_point
from matching.compare_engine import FingerprintMatcher
from preprocessing.quality import QualityChecker


def test_find_core_from_density():
    minutiae = [{"x": 50, "y": 50, "type": "endpoint", "angle": 0}] * 5 + [
        {"x": 52, "y": 48, "type": "bifurcation", "angle": 90}
    ]
    core = find_core_point(minutiae, image_shape=(100, 100))
    assert core is not None
    assert 40 < core[0] < 60


def test_align_by_core_shifts_query():
    ref = [{"x": 100, "y": 100, "type": "endpoint", "angle": 0}]
    qry = [{"x": 20, "y": 30, "type": "endpoint", "angle": 0}]
    _, aligned, meta = align_by_core_point(ref, qry, image_shape=(200, 200))
    assert meta["method"] == "core_point"
    assert abs(aligned[0]["x"] - 100) < 25


def test_quality_checker_heuristic():
    img = np.full((80, 80), 128, dtype=np.uint8)
    cv2_blur = __import__("cv2").GaussianBlur(img, (5, 5), 0)
    score, method = QualityChecker.get_quality_score(cv2_blur)
    assert 0 <= score <= 100
    assert method in ("heuristic_v1", "nfiq2_cli")


def test_fingerprint_matcher_threshold():
    ref = [{"x": i * 10, "y": 50, "type": "endpoint", "angle": float(i)} for i in range(8)]
    qry = [{"x": i * 10 + 2, "y": 52, "type": "endpoint", "angle": float(i)} for i in range(8)]
    engine = FingerprintMatcher(threshold=5.0)
    score, is_match, _ = engine.compare_fingerprints(ref, qry, (120, 120))
    assert score >= 0
    assert is_match is True
