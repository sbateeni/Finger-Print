"""
Minutiae alignment: RANSAC similarity transform from tentative matches.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _transform_points_dict(
    points: list[dict[str, Any]],
    dx: float,
    dy: float,
    rot_deg: float,
    cx: float,
    cy: float,
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    rad = np.radians(rot_deg)
    cos_t, sin_t = np.cos(rad) * scale, np.sin(rad) * scale
    out: list[dict[str, Any]] = []
    for p in points:
        x, y = float(p["x"]) - cx, float(p["y"]) - cy
        nx = x * cos_t - y * sin_t + cx + dx
        ny = x * sin_t + y * cos_t + cy + dy
        out.append({**p, "x": nx, "y": ny, "angle": float(p.get("angle", 0)) + rot_deg})
    return out


def _similarity_from_affine(M: np.ndarray) -> tuple[float, float, float, float]:
    """Extract dx, dy, rotation (deg), scale from 2x3 similarity-like affine matrix."""
    a, b, tx = float(M[0, 0]), float(M[0, 1]), float(M[0, 2])
    c, d, ty = float(M[1, 0]), float(M[1, 1]), float(M[1, 2])
    scale = float(np.sqrt(a * a + c * c)) or 1.0
    rot_deg = float(np.degrees(np.arctan2(c, a)))
    return tx, ty, rot_deg, scale


def refine_alignment_ransac(
    original_minutiae: list[dict[str, Any]],
    partial_minutiae: list[dict[str, Any]],
    tentative_matches: list[dict[str, Any]],
    image_shape: tuple[int, ...],
    *,
    reproj_threshold: float = 12.0,
    max_iters: int = 2000,
) -> dict[str, Any] | None:
    """
    Refine (dx, dy, rot, scale) using RANSAC on matched minutiae pairs.
    Returns alignment dict or None if insufficient inliers.
    """
    if len(tentative_matches) < 3:
        return None

    pts_ref = np.float32([[m["original"]["x"], m["original"]["y"]] for m in tentative_matches])
    pts_qry = np.float32([[m["partial"]["x"], m["partial"]["y"]] for m in tentative_matches])

    M, inliers = cv2.estimateAffinePartial2D(
        pts_qry,
        pts_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(reproj_threshold),
        maxIters=int(max_iters),
        confidence=0.99,
    )
    if M is None:
        return None

    inlier_mask = inliers.ravel().astype(bool) if inliers is not None else np.ones(len(tentative_matches), dtype=bool)
    n_inliers = int(inlier_mask.sum())
    if n_inliers < 3:
        return None

    h, w = image_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    dx, dy, rot_deg, scale = _similarity_from_affine(M)

    return {
        "dx": int(round(dx)),
        "dy": int(round(dy)),
        "rot_deg": float(rot_deg),
        "scale": float(scale),
        "ransac_inliers": n_inliers,
        "ransac_trials": len(tentative_matches),
        "transform_center": (cx, cy),
    }


def align_using_best_triplets(
    original_minutiae: list[dict[str, Any]],
    partial_minutiae: list[dict[str, Any]],
    tentative_matches: list[dict[str, Any]],
    image_shape: tuple[int, ...],
    *,
    max_trials: int = 100,
) -> dict[str, Any] | None:
    """
    Entry point: RANSAC on all tentative matches (replaces random triplet loop;
    OpenCV RANSAC already samples minimal sets internally).
    """
    del max_trials  # reserved for future explicit triplet sampling
    return refine_alignment_ransac(
        original_minutiae,
        partial_minutiae,
        tentative_matches,
        image_shape,
    )
