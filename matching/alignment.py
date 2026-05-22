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


def find_core_point(
    minutiae: list[dict[str, Any]],
    *,
    cores: list[dict[str, Any]] | None = None,
    image_shape: tuple[int, ...] | None = None,
) -> tuple[float, float] | None:
    """
    Core reference: Poincaré cores if provided, else ridge-density centroid of minutiae.
    """
    if cores:
        return float(cores[0]["x"]), float(cores[0]["y"])

    if not minutiae:
        if image_shape:
            h, w = image_shape[:2]
            return w / 2.0, h / 2.0
        return None

    xs = np.array([float(m["x"]) for m in minutiae], dtype=np.float64)
    ys = np.array([float(m["y"]) for m in minutiae], dtype=np.float64)
    if image_shape:
        h, w = image_shape[:2]
        cell = max(16, min(w, h) // 12)
        best_count = -1
        best_xy = (float(np.median(xs)), float(np.median(ys)))
        for cy in range(cell, h - cell, cell):
            for cx in range(cell, w - cell, cell):
                mask = (np.abs(xs - cx) < cell) & (np.abs(ys - cy) < cell)
                cnt = int(mask.sum())
                if cnt > best_count:
                    best_count = cnt
                    best_xy = (float(cx), float(cy))
        return best_xy

    return float(np.median(xs)), float(np.median(ys))


def align_by_core_point(
    minutiae_ref: list[dict[str, Any]],
    minutiae_qry: list[dict[str, Any]],
    *,
    cores_ref: list[dict[str, Any]] | None = None,
    cores_qry: list[dict[str, Any]] | None = None,
    image_shape: tuple[int, ...] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """
    Coarse translation: align query core onto reference core (query stays fixed in ref frame).
    Returns (ref unchanged, transformed query, alignment meta).
    """
    c_ref = find_core_point(minutiae_ref, cores=cores_ref, image_shape=image_shape)
    c_qry = find_core_point(minutiae_qry, cores=cores_qry, image_shape=image_shape)
    if c_ref is None or c_qry is None:
        return minutiae_ref, minutiae_qry, {"method": "core_skip"}

    dx = c_ref[0] - c_qry[0]
    dy = c_ref[1] - c_qry[1]
    aligned_qry = []
    for m in minutiae_qry:
        aligned_qry.append({**m, "x": float(m["x"]) + dx, "y": float(m["y"]) + dy})

    return (
        minutiae_ref,
        aligned_qry,
        {
            "method": "core_point",
            "dx": int(round(dx)),
            "dy": int(round(dy)),
            "core_ref": c_ref,
            "core_qry": c_qry,
        },
    )


def use_core_alignment() -> bool:
    import os

    return (os.getenv("USE_CORE_ALIGNMENT") or "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def use_triplet_alignment() -> bool:
    import os

    return (os.getenv("USE_TRIPLET_ALIGNMENT") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def align_by_triplets(
    minutiae_ref: list[dict[str, Any]],
    minutiae_qry: list[dict[str, Any]],
    image_shape: tuple[int, ...],
    *,
    max_iter: int = 50,
    match_radius: float = 15.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """
    Random triplet sampling: pick best translation on query minutiae (coarse).
    Falls back to core alignment when too few points.
    """
    if len(minutiae_ref) < 3 or len(minutiae_qry) < 3:
        _, aligned, meta = align_by_core_point(
            minutiae_ref, minutiae_qry, image_shape=image_shape
        )
        meta["method"] = "triplet_fallback_core"
        return minutiae_ref, aligned, meta

    pts_ref = np.array([[m["x"], m["y"]] for m in minutiae_ref], dtype=np.float64)
    pts_qry = np.array([[m["x"], m["y"]] for m in minutiae_qry], dtype=np.float64)
    rng = np.random.default_rng(42)

    best_dx, best_dy = 0.0, 0.0
    best_inliers = -1

    for _ in range(max_iter):
        idx_r = rng.choice(len(pts_ref), 3, replace=False)
        idx_q = rng.choice(len(pts_qry), 3, replace=False)
        dx = float(np.mean(pts_ref[idx_r, 0] - pts_qry[idx_q, 0]))
        dy = float(np.mean(pts_ref[idx_r, 1] - pts_qry[idx_q, 1]))
        shifted = pts_qry + np.array([dx, dy])
        inliers = 0
        for p in shifted:
            dists = np.hypot(pts_ref[:, 0] - p[0], pts_ref[:, 1] - p[1])
            if float(dists.min()) < match_radius:
                inliers += 1
        if inliers > best_inliers:
            best_inliers = inliers
            best_dx, best_dy = dx, dy

    aligned_qry = []
    for m in minutiae_qry:
        aligned_qry.append(
            {**m, "x": float(m["x"]) + best_dx, "y": float(m["y"]) + best_dy}
        )
    return (
        minutiae_ref,
        aligned_qry,
        {
            "method": "triplet",
            "dx": int(round(best_dx)),
            "dy": int(round(best_dy)),
            "triplet_inliers": int(best_inliers),
            "triplet_trials": int(max_iter),
        },
    )


def apply_core_prealignment(
    original_minutiae: list[dict[str, Any]],
    partial_minutiae: list[dict[str, Any]],
    image_shape: tuple[int, ...],
    *,
    cores_ref: list[dict[str, Any]] | None = None,
    cores_qry: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    if use_triplet_alignment():
        _, aligned_qry, meta = align_by_triplets(
            original_minutiae,
            partial_minutiae,
            image_shape,
        )
        return original_minutiae, aligned_qry, meta

    if not use_core_alignment():
        return original_minutiae, partial_minutiae, None
    _, aligned_qry, meta = align_by_core_point(
        original_minutiae,
        partial_minutiae,
        cores_ref=cores_ref,
        cores_qry=cores_qry,
        image_shape=image_shape,
    )
    return original_minutiae, aligned_qry, meta


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
