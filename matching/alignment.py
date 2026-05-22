"""
Minutiae alignment (translation estimate) before matching.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple, Union

import numpy as np
from scipy.spatial import KDTree

Minutia = Union[dict, Tuple[float, float, str]]


def _to_xy_type(m: Minutia) -> Tuple[float, float, str]:
    if isinstance(m, dict):
        return float(m["x"]), float(m["y"]), str(m.get("type", "endpoint"))
    if isinstance(m, (list, tuple)) and len(m) >= 3 and isinstance(m[0], (int, float)):
        return float(m[0]), float(m[1]), str(m[2])
    raise TypeError(f"Unsupported minutia format: {type(m)}")


def estimate_transform(pairs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float, float]:
    """Translation-only transform from paired points."""
    if not pairs:
        return 0.0, 0.0, 0.0
    src = np.array([p[0] for p in pairs], dtype=np.float64)
    dst = np.array([p[1] for p in pairs], dtype=np.float64)
    dx = float(np.mean(dst[:, 0] - src[:, 0]))
    dy = float(np.mean(dst[:, 1] - src[:, 1]))
    return dx, dy, 0.0


def align_minutiae(
    minutiae1: List[Minutia],
    minutiae2: List[Minutia],
    max_neighbor_dist: float = 30.0,
    max_seed_points: int = 10,
) -> Tuple[List[dict], List[dict]]:
    """
    Align first set toward second via coarse translation from KD-tree pairs.
    Returns dict minutiae lists compatible with utils.matcher.
    """
    if len(minutiae1) < 3 or len(minutiae2) < 3:
        return [_as_dict(m) for m in minutiae1], [_as_dict(m) for m in minutiae2]

    pts1 = np.array([[_to_xy_type(m)[0], _to_xy_type(m)[1]] for m in minutiae1], dtype=np.float64)
    pts2 = np.array([[_to_xy_type(m)[0], _to_xy_type(m)[1]] for m in minutiae2], dtype=np.float64)
    tree2 = KDTree(pts2)

    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(min(max_seed_points, len(pts1))):
        dist, idx = tree2.query(pts1[i])
        if float(dist) < max_neighbor_dist:
            pairs.append((pts1[i], pts2[int(idx)]))

    if len(pairs) >= 3:
        dx, dy, _ = estimate_transform(pairs)
        aligned = []
        for m in minutiae1:
            d = _as_dict(m)
            d["x"] = int(round(d["x"] + dx))
            d["y"] = int(round(d["y"] + dy))
            aligned.append(d)
        return aligned, [_as_dict(m) for m in minutiae2]

    return [_as_dict(m) for m in minutiae1], [_as_dict(m) for m in minutiae2]


def _as_dict(m: Minutia) -> dict:
    x, y, t = _to_xy_type(m)
    if isinstance(m, dict):
        out = dict(m)
        out["x"], out["y"] = int(x), int(y)
        return out
    return {
        "x": int(x),
        "y": int(y),
        "type": t if t in ("endpoint", "bifurcation") else "endpoint",
        "angle": 0.0,
    }
