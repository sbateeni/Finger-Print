"""Crop regions from fingerprint images (grid sectors or normalized rectangle)."""

from __future__ import annotations

from typing import Any

import numpy as np

FULL_REGION = (0.0, 0.0, 1.0, 1.0)


def parse_norm_region(s: str | None) -> tuple[float, float, float, float]:
    if not s or not str(s).strip():
        return FULL_REGION
    parts = [p.strip() for p in str(s).split(",")]
    if len(parts) != 4:
        return FULL_REGION
    try:
        x, y, w, h = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
    except ValueError:
        return FULL_REGION
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.02, min(1.0 - x, w))
    h = max(0.02, min(1.0 - y, h))
    return x, y, w, h


def format_norm_region(x: float, y: float, w: float, h: float) -> str:
    return f"{x:.4f},{y:.4f},{w:.4f},{h:.4f}"


def is_full_region(x: float, y: float, w: float, h: float) -> bool:
    return x <= 0.001 and y <= 0.001 and w >= 0.999 and h >= 0.999


def crop_norm_region(img: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    if img is None or img.size == 0 or is_full_region(x, y, w, h):
        return img
    hh, ww = img.shape[:2]
    if hh < 8 or ww < 8:
        return img
    x0 = int(x * ww)
    y0 = int(y * hh)
    x1 = int(min(ww, (x + w) * ww))
    y1 = int(min(hh, (y + h) * hh))
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return img
    return np.ascontiguousarray(img[y0:y1, x0:x1])


def normalize_ref_grid(divisions: int, cell: int) -> tuple[int, int]:
    d = int(divisions or 1)
    if d not in (1, 4, 6):
        d = 1
    if d == 1:
        return 1, 0
    c = max(0, min(int(cell or 0), d - 1))
    return d, c


def grid_layout(divisions: int) -> tuple[int, int]:
    if divisions == 4:
        return 2, 2
    if divisions == 6:
        return 3, 2
    return 1, 1


def resolve_grid_cells_for_crop(
    grid_cells: str | None,
    *,
    grid_divisions: int = 1,
    grid_cell: int = 0,
    region_norm: str | None = None,
) -> str:
    """
    Cells string passed to apply_fingerprint_region.

    Empty ref_grid_cells must NOT fall back to grid_cell=0 (that cropped ~25% of
    the reference and looked like unwanted zoom). Grid crop runs only when the
    user explicitly selected sector(s) in grid mode.
    """
    cells_s = (grid_cells or "").strip()
    if cells_s:
        return cells_s
    x, y, w, h = parse_norm_region(region_norm)
    if not is_full_region(x, y, w, h):
        return ""
    return ""


def parse_grid_cells(cells: str | None) -> list[int]:
    if not cells or not str(cells).strip():
        return []
    out: list[int] = []
    for part in str(cells).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def union_grid_cells_norm(divisions: int, cells: list[int]) -> tuple[float, float, float, float]:
    d = int(divisions)
    if d not in (4, 6) or not cells:
        return FULL_REGION
    cols, rows = grid_layout(d)
    xs: list[float] = []
    ys: list[float] = []
    x2s: list[float] = []
    y2s: list[float] = []
    for c in cells:
        if c < 0 or c >= d:
            continue
        col = c % cols
        row = c // cols
        x0 = col / cols
        y0 = row / rows
        x1 = (col + 1) / cols
        y1 = (row + 1) / rows
        xs.append(x0)
        ys.append(y0)
        x2s.append(x1)
        y2s.append(y1)
    if not xs:
        return FULL_REGION
    x = min(xs)
    y = min(ys)
    w = max(x2s) - x
    h = max(y2s) - y
    return x, y, w, h


def cell_label(divisions: int, cell: int, lang: str = "ar") -> str:
    d, c = normalize_ref_grid(divisions, cell)
    if d == 1:
        return "كامل" if lang == "ar" else "Full"
    cols, rows = grid_layout(d)
    row = c // cols + 1
    col = c % cols + 1
    if lang == "ar":
        return f"قطاع {c + 1}/{d} (صف {row}، عمود {col})"
    return f"Sector {c + 1}/{d} (row {row}, col {col})"


def region_label(
    region_norm: str | None,
    *,
    grid_divisions: int = 1,
    grid_cells: str | None = None,
    lang: str = "ar",
    which: str = "ref",
) -> str:
    x, y, w, h = parse_norm_region(region_norm)
    if is_full_region(x, y, w, h):
        if lang == "ar":
            return "البصمة المرجعية كاملة" if which == "ref" else "البصمة المقارنة كاملة"
        return "Full reference" if which == "ref" else "Full query"
    pct = int(round(w * h * 100))
    if lang == "ar":
        base = "منطقة مرجعية" if which == "ref" else "منطقة مقارنة"
        return f"{base} ({pct}% من الصورة)"
    base = "Reference region" if which == "ref" else "Query region"
    return f"{base} ({pct}% of image)"


def apply_fingerprint_region(
    img: np.ndarray,
    region_norm: str | None,
    *,
    grid_divisions: int = 1,
    grid_cells: str | None = None,
) -> np.ndarray:
    """Crop by normalized rect; if full and grid cells set, use union of cells."""
    x, y, w, h = parse_norm_region(region_norm)
    if not is_full_region(x, y, w, h):
        return crop_norm_region(img, x, y, w, h)
    cells = parse_grid_cells(grid_cells)
    if int(grid_divisions) > 1 and cells:
        ux, uy, uw, uh = union_grid_cells_norm(int(grid_divisions), cells)
        return crop_norm_region(img, ux, uy, uw, uh)
    return img


def region_audit_fields(
    *,
    ref_region: str | None = None,
    partial_region: str | None = None,
    ref_grid_divisions: int = 1,
    ref_grid_cells: str | None = None,
    partial_grid_divisions: int = 1,
    partial_grid_cells: str | None = None,
    lang: str = "ar",
) -> dict[str, Any]:
    rd, _ = normalize_ref_grid(ref_grid_divisions, 0)
    pd, _ = normalize_ref_grid(partial_grid_divisions, 0)
    return {
        "ref_region": format_norm_region(*parse_norm_region(ref_region)),
        "partial_region": format_norm_region(*parse_norm_region(partial_region)),
        "ref_grid_divisions": rd,
        "ref_grid_cells": (ref_grid_cells or "").strip(),
        "partial_grid_divisions": pd,
        "partial_grid_cells": (partial_grid_cells or "").strip(),
        "ref_region_label": region_label(
            ref_region, grid_divisions=rd, grid_cells=ref_grid_cells, lang=lang, which="ref"
        ),
        "partial_region_label": region_label(
            partial_region,
            grid_divisions=pd,
            grid_cells=partial_grid_cells,
            lang=lang,
            which="partial",
        ),
    }


# Backward-compatible aliases
def crop_reference_region(img: np.ndarray, divisions: int, cell: int) -> np.ndarray:
    cells = str(cell) if int(divisions) > 1 else ""
    return apply_fingerprint_region(
        img, None, grid_divisions=divisions, grid_cells=cells if cells else str(cell)
    )


def ref_grid_audit_fields(divisions: int, cell: int, lang: str = "ar") -> dict[str, Any]:
    d, c = normalize_ref_grid(divisions, cell)
    return region_audit_fields(
        ref_region=None,
        ref_grid_divisions=d,
        ref_grid_cells=str(c) if d > 1 else "",
        lang=lang,
    )
