import numpy as np

from services.analysis_service.ref_grid import (
    apply_fingerprint_region,
    crop_norm_region,
    normalize_ref_grid,
    parse_grid_cells,
    resolve_grid_cells_for_crop,
    union_grid_cells_norm,
)
from services.analysis_service.transforms import _apply_manual_transform, effective_preview_transform


def test_normalize_ref_grid():
    assert normalize_ref_grid(1, 99) == (1, 0)
    assert normalize_ref_grid(4, 5) == (4, 3)
    assert normalize_ref_grid(6, 10) == (6, 5)


def test_full_region_empty_cells_no_grid_crop():
    """Regression: empty cells + divisions=4 must not crop to cell 0 (false zoom)."""
    img = np.ones((120, 180), dtype=np.uint8) * 255
    cells = resolve_grid_cells_for_crop(
        "",
        grid_divisions=4,
        grid_cell=0,
        region_norm="0,0,1,1",
    )
    assert cells == ""
    out = apply_fingerprint_region(img, "0,0,1,1", grid_divisions=4, grid_cells=cells)
    assert out.shape == img.shape


def test_crop_and_union():
    img = np.ones((120, 180), dtype=np.uint8) * 255
    c4 = apply_fingerprint_region(img, None, grid_divisions=4, grid_cells="0")
    assert c4.shape[0] == 60 and c4.shape[1] == 90
    u = union_grid_cells_norm(4, [0, 3])
    c2 = crop_norm_region(img, *u)
    assert c2.shape[0] == 120


def test_parse_cells():
    assert parse_grid_cells("0,3,5") == [0, 3, 5]


def test_effective_preview_transform_ignores_zoom_when_region_set():
    z, sx, sy, ignored = effective_preview_transform(180, 40, -20, "0.1,0.2,0.4,0.5", True)
    assert ignored is True
    assert z == 100 and sx == 0 and sy == 0
    z2, _, _, ignored2 = effective_preview_transform(150, 10, 0, "0,0,1,1", True)
    assert ignored2 is False
    assert z2 == 150


def test_region_before_zoom_uses_original_pixels():
    """Crop must use pre-zoom coordinates (top-left quadrant of source image)."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:50, :50] = 200
    img[50:, 50:] = 40
    region = "0,0,0.5,0.5"
    crop_then_zoom = _apply_manual_transform(
        apply_fingerprint_region(img, region), 150, 0, 0, True
    )
    zoom_then_crop = apply_fingerprint_region(
        _apply_manual_transform(img, 150, 0, 0, True), region
    )
    assert crop_then_zoom.shape[0] == zoom_then_crop.shape[0]
    assert crop_then_zoom.shape[1] == zoom_then_crop.shape[1]
    assert int(crop_then_zoom.mean()) > int(zoom_then_crop.mean())
