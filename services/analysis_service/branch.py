"""Per-image preprocessing branch: ridges, skeleton, minutiae."""

from __future__ import annotations

from typing import Any, Iterator

import cv2
import numpy as np

from features.extractor import enhance_minutiae_from_image
from features.minutiae_taxonomy import count_by_type
from utils.image_processing import (
    assess_fingerprint_quality,
    detect_edges,
    detect_singular_points,
    enhance_image,
    preprocess_image,
)
from utils.image_utils import _img_data_uri
from utils.minutiae_extractor import (
    extract_minutiae,
    visualize_minutiae,
    visualize_singular_points,
)


def _process_branch(
    gray: np.ndarray,
    denoise_method: str,
    fast_denoise_h: int,
    gauss_ksize: int,
    border_margin: int,
    min_distance: int,
    min_contrast: int,
    min_angle_diff: int,
):
    out = {}
    proc = preprocess_image(
        gray,
        denoise_method=denoise_method,
        fast_denoise_h=fast_denoise_h,
        gauss_ksize=gauss_ksize,
    )
    if proc is None:
        out["error"] = "فشلت المعالجة المسبقة"
        return out
    out["processed"] = proc
    out["white_pre"] = int(np.sum(proc == 255))

    quality_score, _qm = assess_fingerprint_quality(proc)
    out["quality_score"] = quality_score

    ridges, omap = detect_edges(proc)
    if ridges is None:
        out["error"] = "فشل استخراج التموجات"
        return out
    out["ridges"] = ridges
    out["white_ridges"] = int(np.sum(ridges == 255))

    cores, deltas = detect_singular_points(omap, proc > 127)
    out["cores"] = cores
    out["deltas"] = deltas

    skel = enhance_image(ridges)
    if skel is None:
        out["error"] = "فشلت الهيكلة"
        return out
    out["skeleton"] = skel
    out["white_skel"] = int(np.sum(skel == 255))

    minutiae = extract_minutiae(
        skel,
        border_margin=border_margin,
        min_distance=min_distance,
        original_image=proc,
        min_contrast=min_contrast,
        min_angle_diff=min_angle_diff,
        ridge_image=ridges,
    )
    minutiae, ext_note = enhance_minutiae_from_image(
        proc, minutiae, min_distance=float(min_distance)
    )
    out["minutiae_extraction"] = ext_note
    out["minutiae"] = minutiae
    out["minutiae_count"] = len(minutiae)
    out["minutiae_by_type"] = count_by_type(minutiae)
    if minutiae:
        vis = visualize_minutiae(skel, minutiae)
        out["vis_minutiae"] = vis
    return out


def _iter_branch_live(
    gray: np.ndarray,
    denoise_method: str,
    fast_denoise_h: int,
    gauss_ksize: int,
    border_margin: int,
    min_distance: int,
    min_contrast: int,
    min_angle_diff: int,
    branch_key: str,
    holder: list,
) -> Iterator[dict[str, Any]]:
    """يُصدِر أحداث صورة تلو الأخرى ثم يضع الناتج الكامل في holder[0]."""
    holder[0] = None
    proc = preprocess_image(
        gray,
        denoise_method=denoise_method,
        fast_denoise_h=fast_denoise_h,
        gauss_ksize=gauss_ksize,
    )
    if proc is None:
        yield {"type": "error", "branch": branch_key, "message": "فشلت المعالجة المسبقة"}
        return
    yield {
        "type": "image",
        "branch": branch_key,
        "stage": "processed",
        "src": _img_data_uri(proc),
        "white": int(np.sum(proc > 127)),
    }

    quality_score, q_map = assess_fingerprint_quality(proc)
    q_map_color = cv2.applyColorMap(q_map, cv2.COLORMAP_JET)
    yield {
        "type": "image",
        "branch": branch_key,
        "stage": "quality_map",
        "src": _img_data_uri(q_map_color),
        "quality_score": round(quality_score, 1),
    }

    ridges, omap = detect_edges(proc)
    if ridges is None:
        yield {"type": "error", "branch": branch_key, "message": "فشل استخراج التموجات"}
        return
    yield {
        "type": "image",
        "branch": branch_key,
        "stage": "ridges",
        "src": _img_data_uri(ridges),
        "white": int(np.sum(ridges > 127)),
    }

    cores, deltas = detect_singular_points(omap, proc > 127)

    skel = enhance_image(ridges)
    if skel is None:
        yield {"type": "error", "branch": branch_key, "message": "فشلت الهيكلة"}
        return
    yield {
        "type": "image",
        "branch": branch_key,
        "stage": "skeleton",
        "src": _img_data_uri(skel),
        "white": int(np.sum(skel > 0)),
    }

    minutiae = extract_minutiae(
        skel,
        border_margin=border_margin,
        min_distance=min_distance,
        original_image=proc,
        min_contrast=min_contrast,
        min_angle_diff=min_angle_diff,
        ridge_image=ridges,
    )
    minutiae, ext_note = enhance_minutiae_from_image(
        proc, minutiae, min_distance=float(min_distance)
    )
    vis = None
    if minutiae:
        vis = visualize_minutiae(skel, minutiae)
    if vis is not None:
        yield {
            "type": "image",
            "branch": branch_key,
            "stage": "minutiae_vis",
            "src": _img_data_uri(vis),
            "n_min": len(minutiae),
        }

    if cores or deltas:
        vis_sp = visualize_singular_points(proc, cores, deltas)
        yield {
            "type": "image",
            "branch": branch_key,
            "stage": "singular_vis",
            "src": _img_data_uri(vis_sp),
        }

    out = {
        "processed": proc,
        "ridges": ridges,
        "skeleton": skel,
        "minutiae": minutiae,
        "minutiae_count": len(minutiae),
        "minutiae_extraction": ext_note,
        "minutiae_by_type": count_by_type(minutiae),
        "quality_score": quality_score,
        "quality_map": q_map_color,
        "cores": cores,
        "deltas": deltas,
        "white_pre": int(np.sum(proc > 127)),
        "white_ridges": int(np.sum(ridges > 127)),
        "white_skel": int(np.sum(skel > 0)),
        "vis_minutiae": vis,
    }
    holder[0] = out
