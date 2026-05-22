import numpy as np

from features.minutiae_filter import filter_minutiae_by_confidence


def test_filter_drops_border_and_isolated():
    sk = np.zeros((40, 40), dtype=np.uint8)
    sk[20, 5:35] = 255
    minutiae = [
        {"x": 2, "y": 20, "type": "endpoint", "angle": 0},
        {"x": 20, "y": 20, "type": "bifurcation", "angle": 90},
        {"x": 38, "y": 20, "type": "endpoint", "angle": 180},
    ]
    out = filter_minutiae_by_confidence(minutiae, sk, edge_margin=5, min_ridge_neighbors=2)
    assert len(out) == 1
    assert out[0]["x"] == 20
