"""
Extra minutiae filtering for partial prints (edges, isolated ridge noise).
"""

from __future__ import annotations

from typing import Any


def filter_minutiae_by_confidence(
    minutiae: list[dict[str, Any]],
    skeleton,
    *,
    edge_margin: int = 10,
    window: int = 5,
    min_ridge_neighbors: int = 3,
) -> list[dict[str, Any]]:
    """
    Drop minutiae on image borders or on skeleton pixels with too few ridge neighbors
    (often spurs / broken noise on partial crops).
    """
    if not minutiae or skeleton is None:
        return minutiae or []

    h, w = skeleton.shape[:2]
    sk = skeleton
    # Treat 255 as ridge (skeleton from pipeline)
    ridge = (sk >= 128).astype("uint8")

    filtered: list[dict[str, Any]] = []
    for m in minutiae:
        x, y = int(m["x"]), int(m["y"])
        if x < edge_margin or x >= w - edge_margin or y < edge_margin or y >= h - edge_margin:
            continue
        if not (0 <= x < w and 0 <= y < h):
            continue

        neighbor_count = 0
        for dy in range(-window, window + 1):
            for dx in range(-window, window + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and ridge[ny, nx]:
                    neighbor_count += 1
        if neighbor_count < min_ridge_neighbors:
            continue
        filtered.append(m)
    return filtered
