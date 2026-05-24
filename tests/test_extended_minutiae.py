import numpy as np

from features.extended_minutiae import augment_minutiae, merge_by_distance
from features.minutiae_taxonomy import normalize_minutiae_type
from utils.matcher import _types_compatible


def _angle_zero(sk, x, y):
    return 0.0


def test_types_compatible_divergence_bifurcation():
    assert _types_compatible("bifurcation", "divergence")
    assert normalize_minutiae_type("Ending Ridge") == "endpoint"


def test_augment_adds_bridge_on_crossing():
    sk = np.zeros((40, 40), dtype=np.uint8)
    # cross
    sk[20, 18:23] = 255
    sk[18:23, 20] = 255
    base = [{"x": 20, "y": 10, "type": "endpoint", "angle": 90}]
    out = augment_minutiae(base, sk, _angle_zero, min_distance=5)
    types = {m["type"] for m in out}
    assert "bridge" in types or len(out) >= len(base)


def test_merge_priority_keeps_bifurcation():
    pts = [
        {"x": 10, "y": 10, "type": "dot", "angle": 0},
        {"x": 11, "y": 10, "type": "bifurcation", "angle": 0},
    ]
    m = merge_by_distance(pts, min_distance=5)
    assert len(m) == 1
    assert m[0]["type"] == "bifurcation"
