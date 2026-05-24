"""
Remove white background from Palestinian Police logo PNG/JPEG.

Place the official logo as:
  static/branding/palestinian-police-logo-source.png
Then run:
  python scripts/prepare_police_logo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BRAND = ROOT / "static" / "branding"
SOURCE_CANDIDATES = (
    BRAND / "palestinian-police-logo-source.png",
    BRAND / "palestinian-police-logo-source.jpg",
    BRAND / "palestinian-police-logo-source.jpeg",
)


def _find_source() -> Path:
    for p in SOURCE_CANDIDATES:
        if p.is_file():
            return p
    return SOURCE_CANDIDATES[0]
OUT = BRAND / "palestinian-police-logo.png"
WM = BRAND / "palestinian-police-logo-wm.png"


def remove_white_background(img: Image.Image) -> Image.Image:
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    white = (r > 235) & (g > 235) & (b > 235)
    arr[white, 3] = 0
    fringe = (r > 200) & (g > 200) & (b > 200) & (~white)
    arr[fringe, 3] = np.clip(arr[fringe, 3].astype(np.int32) - 140, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main() -> None:
    BRAND.mkdir(parents=True, exist_ok=True)
    source = _find_source()
    if not source.is_file():
        raise SystemExit(
            f"Missing source image.\n"
            f"Save the police logo as:\n  {SOURCE_CANDIDATES[0]}\n"
            f"Then run this script again."
        )
    img = remove_white_background(Image.open(source))
    img.save(OUT, optimize=True)
    w, h = img.size
    wm = img.resize((max(1, int(w * 0.55)), max(1, int(h * 0.55))), Image.Resampling.LANCZOS)
    wm.save(WM, optimize=True)
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")
    print(f"Wrote {WM.relative_to(ROOT)} ({WM.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
