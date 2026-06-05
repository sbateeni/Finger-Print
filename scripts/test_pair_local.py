#!/usr/bin/env python3
"""Quick local pair test: two fingerprint images → match stats."""

from __future__ import annotations

import sys
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
from features.minutiae_taxonomy import count_by_type
from services.analysis_service import _process_branch
from utils.fusion import apply_fusion_to_match
from utils.image_utils import _decode_upload_type
from utils.matcher import match_fingerprints_with_partial_alignment


def analyze_image(path: Path, label: str) -> dict:
    gray = _decode_upload_type(path.read_bytes())
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
        return {"label": label, "path": str(path), "error": branch["error"]}
    return {
        "label": label,
        "path": str(path),
        "branch": branch,
        "counts": count_by_type(branch.get("minutiae") or []),
        "n": branch.get("minutiae_count", 0),
        "quality": branch.get("quality_score"),
        "cores": len(branch.get("cores") or []),
        "deltas": len(branch.get("deltas") or []),
    }


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python scripts/test_pair_local.py <ref.jpg> <query.jpg>")
        sys.exit(1)
    ref_p = Path(sys.argv[1])
    qry_p = Path(sys.argv[2])

    ref = analyze_image(ref_p, "مرجعية")
    qry = analyze_image(qry_p, "مقارنة")

    for r in (ref, qry):
        print(f"\n=== {r['label']}: {r['path']} ===")
        if r.get("error"):
            print("خطأ:", r["error"])
            continue
        print(f"جودة (0-1): {r.get('quality')}")
        print(f"Core: {r['cores']}  Delta: {r['deltas']}")
        print(f"عدد النقاط: {r['n']}")
        print("حسب النوع:", r["counts"])

    if ref.get("error") or qry.get("error"):
        sys.exit(1)

    ro, rp = ref["branch"], qry["branch"]
    sk = ro["skeleton"]
    mr = match_fingerprints_with_partial_alignment(
        ro["minutiae"],
        rp["minutiae"],
        sk.shape,
        cores_ref=ro.get("cores"),
        cores_qry=rp.get("cores"),
    )
    mr = apply_fusion_to_match(mr, ro, rp)

    print("\n=== نتيجة المطابقة ===")
    print(f"match_score: {mr.get('match_score', 0):.2f}%")
    print(f"mcc_score: {mr.get('mcc_score', 0):.2f}%")
    print(f"fused_score: {mr.get('fused_score', 0):.2f}%")
    print(f"decision_status: {mr.get('decision_status')}")
    print(f"matched_points: {mr.get('matched_points')} / partial {mr.get('total_partial')}")
    print(f"alignment_gain_matches: {mr.get('alignment_gain_matches')}")
    print(f"alignment_method: {mr.get('alignment_method')}")
    if mr.get("core_prealignment"):
        print(f"core_prealignment: {mr.get('core_prealignment')}")


if __name__ == "__main__":
    main()
