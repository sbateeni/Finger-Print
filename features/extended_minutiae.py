"""
Extended ridge features from PDF (Mosul) + forensic poster taxonomy:
lake, dot, bridge, divergence — in addition to endpoint / bifurcation / island.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np


def extended_minutiae_enabled() -> bool:
    return (os.getenv("ENABLE_EXTENDED_MINUTIAE") or "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _crossing_number(skeleton: np.ndarray, x: int, y: int) -> int:
    p = [
        skeleton[y - 1, x - 1],
        skeleton[y - 1, x],
        skeleton[y - 1, x + 1],
        skeleton[y, x + 1],
        skeleton[y + 1, x + 1],
        skeleton[y + 1, x],
        skeleton[y + 1, x - 1],
        skeleton[y, x - 1],
        skeleton[y - 1, x - 1],
    ]
    cn = 0
    for i in range(8):
        if p[i] == 0 and p[i + 1] == 255:
            cn += 1
    return cn


def _ridge_neighbors(skeleton: np.ndarray, x: int, y: int) -> list[tuple[int, int]]:
    h, w = skeleton.shape
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx] == 255:
                out.append((nx, ny))
    return out


def _trace_ridge(
    skeleton: np.ndarray,
    start: tuple[int, int],
    *,
    max_steps: int = 120,
) -> list[tuple[int, int]]:
    path = [start]
    visited = {start}
    curr = start
    prev = None
    for _ in range(max_steps):
        nbrs = [n for n in _ridge_neighbors(skeleton, curr[0], curr[1]) if n != prev]
        if not nbrs:
            break
        nxt = nbrs[0]
        if len(nbrs) > 1:
            for n in nbrs:
                if n not in visited:
                    nxt = n
                    break
        if nxt in visited and nxt != start:
            path.append(nxt)
            break
        if nxt in visited:
            break
        visited.add(nxt)
        path.append(nxt)
        prev, curr = curr, nxt
    return path


def extract_cn_extras(skeleton: np.ndarray, angle_fn) -> list[dict[str, Any]]:
    """CN=4 → bridge (جسر). CN=2 يُتجاهل هنا (كثير من الإيجابيات الكاذبة على الهيكل)."""
    h, w = skeleton.shape
    out: list[dict[str, Any]] = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] != 255:
                continue
            if _crossing_number(skeleton, x, y) == 4:
                out.append(
                    {
                        "x": x,
                        "y": y,
                        "type": "bridge",
                        "angle": angle_fn(skeleton, x, y),
                    }
                )
    return out


def detect_lake_loops(
    skeleton: np.ndarray,
    angle_fn,
    *,
    min_loop: int = 8,
    max_loop: int = 28,
) -> list[dict[str, Any]]:
    """Closed short loops on skeleton → lake (بحيرة PDF)."""
    h, w = skeleton.shape
    visited_global: set[tuple[int, int]] = set()
    lakes: list[dict[str, Any]] = []

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] != 255 or (x, y) in visited_global:
                continue
            if _crossing_number(skeleton, x, y) != 1:
                continue
            path = _trace_ridge(skeleton, (x, y), max_steps=max_loop + 5)
            for pt in path:
                visited_global.add(pt)
            if len(path) < min_loop:
                continue
            closed = path[0] == path[-1] or (
                len(path) >= min_loop
                and np.hypot(path[0][0] - path[-1][0], path[0][1] - path[-1][1]) <= 2.5
            )
            if closed and min_loop <= len(path) <= max_loop:
                cx = int(round(np.mean([p[0] for p in path])))
                cy = int(round(np.mean([p[1] for p in path])))
                lakes.append(
                    {
                        "x": cx,
                        "y": cy,
                        "type": "lake",
                        "angle": angle_fn(skeleton, cx, cy),
                    }
                )
    return lakes


def detect_dots_on_ridges(
    ridge_image: np.ndarray | None,
    skeleton: np.ndarray,
    *,
    min_area: int = 2,
    max_area: int = 14,
) -> list[dict[str, Any]]:
    """Isolated ridge blobs (نقطة PDF) — not already on skeleton junction."""
    if ridge_image is None:
        thick = cv2.dilate((skeleton > 0).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    else:
        thick = ridge_image if len(ridge_image.shape) == 2 else cv2.cvtColor(ridge_image, cv2.COLOR_BGR2GRAY)
        thick = (thick > 127).astype(np.uint8) * 255

    sk_set = set(zip(*np.where(skeleton == 255)))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(thick, connectivity=8)
    dots: list[dict[str, Any]] = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        cx = int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2)
        cy = int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2)
        if (cx, cy) in sk_set:
            continue
        dots.append({"x": cx, "y": cy, "type": "dot", "angle": 0.0})
    return dots


def mark_divergences(
    minutiae: list[dict[str, Any]],
    skeleton: np.ndarray,
    *,
    angle_spread_deg: float = 55.0,
) -> list[dict[str, Any]]:
    """Wide-angle bifurcations → divergence (حافة بارزة)."""
    out: list[dict[str, Any]] = []
    for m in minutiae:
        if m.get("type") != "bifurcation":
            out.append(m)
            continue
        x, y = int(m["x"]), int(m["y"])
        nbrs = _ridge_neighbors(skeleton, x, y)
        if len(nbrs) < 3:
            out.append(m)
            continue
        angles = [np.degrees(np.arctan2(ny - y, nx - x)) % 360.0 for nx, ny in nbrs]
        angles.sort()
        spreads = [
            min((angles[(i + 1) % len(angles)] - angles[i]) % 360.0, 360.0 - ((angles[(i + 1) % len(angles)] - angles[i]) % 360.0))
            for i in range(len(angles))
        ]
        if spreads and max(spreads) >= angle_spread_deg:
            out.append({**m, "type": "divergence"})
        else:
            out.append(m)
    return out


def merge_by_distance(
    points: list[dict[str, Any]],
    *,
    min_distance: float = 10.0,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    priority = {
        "bifurcation": 5,
        "divergence": 5,
        "bridge": 4,
        "lake": 3,
        "endpoint": 2,
        "island": 2,
        "dot": 1,
    }
    for m in sorted(points, key=lambda p: -priority.get(p.get("type", ""), 0)):
        if any(np.hypot(m["x"] - o["x"], m["y"] - o["y"]) < min_distance for o in merged):
            continue
        merged.append(m)
    return merged


def augment_minutiae(
    base: list[dict[str, Any]],
    skeleton: np.ndarray,
    angle_fn,
    *,
    ridge_image: np.ndarray | None = None,
    min_distance: float = 10.0,
) -> list[dict[str, Any]]:
    if not extended_minutiae_enabled():
        return base
    extra: list[dict[str, Any]] = []
    extra.extend(extract_cn_extras(skeleton, angle_fn))
    extra.extend(detect_lake_loops(skeleton, angle_fn))
    extra.extend(detect_dots_on_ridges(ridge_image, skeleton))
    combined = merge_by_distance(list(base) + extra, min_distance=min_distance)
    return mark_divergences(combined, skeleton)
