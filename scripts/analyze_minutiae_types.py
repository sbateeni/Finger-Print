#!/usr/bin/env python3
"""Print minutiae counts by type for a fingerprint image (poster / training validation)."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from config import (
    DEFAULT_BORDER_MARGIN,
    DEFAULT_MIN_ANGLE_DIFF,
    DEFAULT_MIN_CONTRAST,
    DEFAULT_MIN_DISTANCE,
)
from features.minutiae_taxonomy import POSTER_36_SUMMARY, count_by_type
from services.analysis_service import _process_branch
from utils.image_utils import _decode_upload_type


def main() -> None:
    p = argparse.ArgumentParser(description="Count minutiae types on one image")
    p.add_argument("image", type=Path, help="Fingerprint image path")
    args = p.parse_args()
    raw = args.image.read_bytes()
    gray = _decode_upload_type(raw)
    branch = _process_branch(
        gray,
        "fastNlMeans",
        10,
        5,
        DEFAULT_BORDER_MARGIN,
        DEFAULT_MIN_DISTANCE,
        DEFAULT_MIN_CONTRAST,
        DEFAULT_MIN_ANGLE_DIFF,
    )
    if branch.get("error"):
        print("Error:", branch["error"])
        sys.exit(1)
    counts = count_by_type(branch.get("minutiae") or [])
    total = sum(counts.values())
    print(f"Image: {args.image}")
    print(f"Total minutiae: {total}")
    print("By type:")
    for t, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {n}")
    print("\nPoster reference (educational 36-point key):")
    for t, n in POSTER_36_SUMMARY.items():
        print(f"  {t}: {n}")


if __name__ == "__main__":
    main()
