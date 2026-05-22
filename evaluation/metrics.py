from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass
class MetricsResult:
    threshold: float
    far: float
    frr: float
    eer: float
    tp: int
    tn: int
    fp: int
    fn: int


def confusion_counts(
    genuine_scores: Sequence[float], impostor_scores: Sequence[float], threshold: float
) -> Tuple[int, int, int, int]:
    tp = sum(1 for s in genuine_scores if s >= threshold)
    fn = sum(1 for s in genuine_scores if s < threshold)
    fp = sum(1 for s in impostor_scores if s >= threshold)
    tn = sum(1 for s in impostor_scores if s < threshold)
    return tp, tn, fp, fn


def far_frr(
    genuine_scores: Sequence[float], impostor_scores: Sequence[float], threshold: float
) -> Tuple[float, float]:
    tp, tn, fp, fn = confusion_counts(genuine_scores, impostor_scores, threshold)
    far = fp / max(fp + tn, 1)
    frr = fn / max(fn + tp, 1)
    return far, frr


def compute_eer(
    genuine_scores: Sequence[float], impostor_scores: Sequence[float], thresholds: Iterable[float]
) -> MetricsResult:
    best: MetricsResult | None = None
    for threshold in thresholds:
        far, frr = far_frr(genuine_scores, impostor_scores, threshold)
        tp, tn, fp, fn = confusion_counts(genuine_scores, impostor_scores, threshold)
        eer = (far + frr) / 2.0
        candidate = MetricsResult(
            threshold=float(threshold),
            far=far,
            frr=frr,
            eer=eer,
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
        )
        if best is None or abs(candidate.far - candidate.frr) < abs(best.far - best.frr):
            best = candidate
    if best is None:
        return MetricsResult(threshold=50.0, far=1.0, frr=1.0, eer=1.0, tp=0, tn=0, fp=0, fn=0)
    return best


def roc_points(
    genuine_scores: Sequence[float],
    impostor_scores: Sequence[float],
    thresholds: Iterable[float],
) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for threshold in thresholds:
        tp, tn, fp, fn = confusion_counts(genuine_scores, impostor_scores, threshold)
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        points.append((fpr, tpr))
    return points
