from __future__ import annotations

"""
Offline baseline: scores each row using Finger-Print pipeline (no Flask, no SIFT replacement).
Adds EER / FAR / FRR and ROC export (from evaluation/metrics.py).
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Repo root on sys.path (run as: python evaluation/run_baseline.py)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import compute_eer, roc_points
from config import (
    DEFAULT_BORDER_MARGIN,
    DEFAULT_MIN_ANGLE_DIFF,
    DEFAULT_MIN_CONTRAST,
    DEFAULT_MIN_DISTANCE,
)
from services.analysis_service import process_form_analysis, run_matching_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline EER/ROC for Finger-Print pipeline")
    p.add_argument("--pairs", required=True, help="CSV: img1,img2,label (genuine|impostor)")
    p.add_argument(
        "--score",
        choices=("fused", "match", "mcc"),
        default="fused",
        help="Score field from match result (default: fused_score after ORB+fusion when available)",
    )
    p.add_argument("--out-json", default="evaluation/baseline_metrics.json")
    p.add_argument("--out-csv", default="evaluation/roc_points.csv")
    # Optional: align with typical UI defaults
    p.add_argument("--original-zoom", type=int, default=100)
    p.add_argument("--partial-zoom", type=int, default=100)
    p.add_argument("--partial-shift-x", type=int, default=0)
    p.add_argument("--partial-shift-y", type=int, default=0)
    p.add_argument("--apply-preview-scale", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--auto-scale-normalization", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--denoise-method", default="fastNlMeans")
    p.add_argument("--fast-denoise-h", type=int, default=10)
    p.add_argument("--gauss-ksize", type=int, default=3)
    p.add_argument(
        "--respect-quality-gate",
        action="store_true",
        help="Apply the same quality gate as the web UI (default: off for small demo pairs)",
    )
    return p.parse_args()


def load_pairs(path: Path, root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img1 = (root / row["img1"]).resolve()
            img2 = (root / row["img2"]).resolve()
            label = row["label"].strip().lower()
            rows.append({"img1": str(img1), "img2": str(img2), "label": label})
    return rows


def extract_score(match_result: Dict[str, Any], score_key: str) -> float:
    if score_key == "fused":
        return float(match_result.get("fused_score") or 0.0)
    if score_key == "match":
        return float(match_result.get("match_score") or 0.0)
    if score_key == "mcc":
        return float(match_result.get("mcc_score") or 0.0)
    raise ValueError(score_key)


def main() -> int:
    args = parse_args()
    root = ROOT
    pair_path = (root / args.pairs).resolve()
    rows = load_pairs(pair_path, root)

    genuine_scores: List[float] = []
    impostor_scores: List[float] = []
    per_row_log: List[Dict[str, Any]] = []

    out_json = (root / args.out_json).resolve()
    out_csv = (root / args.out_csv).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    border_margin = DEFAULT_BORDER_MARGIN
    min_distance = DEFAULT_MIN_DISTANCE
    min_contrast = DEFAULT_MIN_CONTRAST
    min_angle_diff = DEFAULT_MIN_ANGLE_DIFF

    for row in rows:
        p1 = Path(row["img1"])
        p2 = Path(row["img2"])
        label = row["label"]
        if label not in ("genuine", "impostor"):
            print(f"skip unknown label={label} for {p1.name} vs {p2.name}")
            continue
        try:
            o_raw = p1.read_bytes()
            p_raw = p2.read_bytes()
        except OSError as e:
            print(f"read failed {p1} / {p2}: {e}")
            score_v = 0.0
            if label == "genuine":
                genuine_scores.append(score_v)
            else:
                impostor_scores.append(score_v)
            per_row_log.append(
                {"img1": row["img1"], "img2": row["img2"], "label": label, "error": str(e)}
            )
            continue

        try:
            same_file, sha_o, sha_p, ro, rp, dm = process_form_analysis(
                o_raw,
                p_raw,
                border_margin,
                min_distance,
                min_contrast,
                min_angle_diff,
                args.denoise_method,
                args.fast_denoise_h,
                args.gauss_ksize,
                args.original_zoom,
                args.partial_zoom,
                args.partial_shift_x,
                args.partial_shift_y,
                args.apply_preview_scale,
                args.auto_scale_normalization,
                operator_name="baseline",
                case_reference="evaluation",
            )
        except Exception as e:
            print(f"process_form_analysis failed for {p1.name} vs {p2.name}: {e}")
            score_v = 0.0
            if label == "genuine":
                genuine_scores.append(score_v)
            else:
                impostor_scores.append(score_v)
            per_row_log.append(
                {"img1": row["img1"], "img2": row["img2"], "label": label, "error": str(e)}
            )
            continue

        form_ctx = {
            "border_margin": border_margin,
            "min_distance": min_distance,
            "min_contrast": min_contrast,
            "min_angle_diff": min_angle_diff,
            "denoise_method": dm,
            "fast_denoise_h": args.fast_denoise_h,
            "gauss_ksize": args.gauss_ksize,
            "original_zoom": args.original_zoom,
            "partial_zoom": args.partial_zoom,
            "partial_shift_x": args.partial_shift_x,
            "partial_shift_y": args.partial_shift_y,
            "apply_preview_scale": args.apply_preview_scale,
            "auto_scale_normalization": args.auto_scale_normalization,
            "auto_scale_factor_applied": round(float(ro.get("auto_scale_factor_applied", 1.0)), 4),
        }

        score_v = 0.0
        row_entry: Dict[str, Any] = {"img1": row["img1"], "img2": row["img2"], "label": label}

        if ro.get("error") or rp.get("error"):
            row_entry["error"] = ro.get("error") or rp.get("error")
        else:
            try:
                mr, _, _report_rel, _audit, _warn = run_matching_pipeline(
                    ro,
                    rp,
                    sha_o,
                    sha_p,
                    dm,
                    form_ctx,
                    "baseline",
                    "evaluation",
                    border_margin,
                    min_distance,
                    min_contrast,
                    min_angle_diff,
                    args.fast_denoise_h,
                    args.gauss_ksize,
                    write_report_and_audit=False,
                    quality_gate_enabled=args.respect_quality_gate,
                )
                score_v = extract_score(mr, args.score)
                row_entry.update(
                    {
                        "score_field": args.score,
                        "score_value": score_v,
                        "status": mr.get("status"),
                        "same_file_bytes": bool(same_file),
                    }
                )
            except Exception as e:
                row_entry["error"] = str(e)

        row_entry.setdefault("score_value", 0.0)
        score_v = float(row_entry["score_value"])
        per_row_log.append(row_entry)

        if label == "genuine":
            genuine_scores.append(score_v)
        else:
            impostor_scores.append(score_v)

        print(
            f"{label:8} score({args.score})={score_v:.2f}  {Path(row['img1']).name} vs {Path(row['img2']).name}"
        )

    thresholds = [float(t) for t in range(0, 101, 2)]
    metrics = compute_eer(genuine_scores, impostor_scores, thresholds)
    roc = roc_points(genuine_scores, impostor_scores, thresholds)

    payload = {
        "pipeline": "Finger-Print process_form_analysis + run_matching_pipeline",
        "score_used": args.score,
        "genuine_count": len(genuine_scores),
        "impostor_count": len(impostor_scores),
        "eer": metrics.eer,
        "threshold_at_eer": metrics.threshold,
        "far_at_eer": metrics.far,
        "frr_at_eer": metrics.frr,
        "confusion_at_eer": {"tp": metrics.tp, "tn": metrics.tn, "fp": metrics.fp, "fn": metrics.fn},
        "per_row": per_row_log,
    }

    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["fpr", "tpr"])
        for fpr, tpr in roc:
            w.writerow([fpr, tpr])

    print(json.dumps({k: v for k, v in payload.items() if k != "per_row"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
