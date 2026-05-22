"""Unit tests for evaluation.metrics (borrowed/adapted from Finger-print-pro)."""

from evaluation.metrics import compute_eer, far_frr


def test_far_frr_behaviour():
    genuine = [90, 80, 75]
    impostor = [10, 20, 25]
    far, frr = far_frr(genuine, impostor, threshold=50)
    assert far == 0
    assert frr == 0


def test_compute_eer_returns_valid_range():
    genuine = [90, 80, 70, 60]
    impostor = [40, 30, 20, 10]
    result = compute_eer(genuine, impostor, thresholds=range(0, 101, 5))
    assert 0 <= result.eer <= 1
    assert 0 <= result.far <= 1
    assert 0 <= result.frr <= 1
