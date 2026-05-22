"""
Calibration utility for fused fingerprint scores.

Input file format (CSV or JSON array), each row/object must include:
  - label: 1 for genuine, 0 for impostor (accepts true/false, genuine/impostor)
  - minutiae_score: 0..100
  - mcc_score: 0..100
  - orb_score: 0..100

Example CSV header:
label,minutiae_score,mcc_score,orb_score
1,78.2,51.0,64.0
0,21.4,12.2,18.0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Sample:
    label: int
    minutiae_score: float
    mcc_score: float
    orb_score: float


def _as_label(v: Any) -> int:
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return 1 if int(v) == 1 else 0
    s = str(v).strip().lower()
    if s in {"1", "true", "genuine", "match", "positive"}:
        return 1
    if s in {"0", "false", "impostor", "non-match", "negative"}:
        return 0
    raise ValueError(f"Unsupported label value: {v!r}")


def _clip(x: float) -> float:
    return max(0.0, min(100.0, float(x)))


def _label_from_status(v: Any) -> int | None:
    s = str(v or "").strip().upper()
    if not s:
        return None
    if "HIGH MATCH" in s or "MEDIUM MATCH" in s:
        return 1
    if "LOW MATCH" in s or "NO MATCH" in s or "INCONCLUSIVE" in s:
        return 0
    return None


def load_samples(path: Path, infer_label_from_status: bool = False) -> list[Sample]:
    if path.suffix.lower() == ".csv":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, list):
            raise ValueError("JSON input must be an array of objects")
        rows = obj
    else:
        raise ValueError("Input must be .csv or .json")

    out: list[Sample] = []
    skipped_unlabeled = 0
    for r in rows:
        raw_label = r.get("label")
        if raw_label is None or str(raw_label).strip() == "":
            if infer_label_from_status:
                inferred = _label_from_status(r.get("status"))
                if inferred is None:
                    skipped_unlabeled += 1
                    continue
                raw_label = inferred
            else:
                skipped_unlabeled += 1
                continue
        out.append(
            Sample(
                label=_as_label(raw_label),
                minutiae_score=_clip(float(r["minutiae_score"])),
                mcc_score=_clip(float(r["mcc_score"])),
                orb_score=_clip(float(r["orb_score"])),
            )
        )
    if not out:
        raise ValueError(
            "No labeled samples found. Fill 'label' with 1/0 for at least some rows."
        )
    if skipped_unlabeled:
        print(f"Skipped unlabeled rows: {skipped_unlabeled}")
    return out


def fused_score(s: Sample, w_min: float, w_mcc: float, w_orb: float) -> float:
    w_sum = w_min + w_mcc + w_orb
    if w_sum <= 0:
        return 0.0
    return (
        s.minutiae_score * w_min + s.mcc_score * w_mcc + s.orb_score * w_orb
    ) / w_sum


def _roc_points(scores_labels: list[tuple[float, int]]) -> list[dict[str, float]]:
    # Threshold sweep from high to low (accept if score >= threshold)
    thresholds = sorted({round(sc, 6) for sc, _ in scores_labels}, reverse=True)
    thresholds = [101.0] + thresholds + [-1.0]

    pos = sum(1 for _, y in scores_labels if y == 1)
    neg = sum(1 for _, y in scores_labels if y == 0)
    if pos == 0 or neg == 0:
        raise ValueError("Calibration requires both positive and negative samples")

    pts: list[dict[str, float]] = []
    for t in thresholds:
        tp = fp = tn = fn = 0
        for s, y in scores_labels:
            pred = 1 if s >= t else 0
            if pred == 1 and y == 1:
                tp += 1
            elif pred == 1 and y == 0:
                fp += 1
            elif pred == 0 and y == 0:
                tn += 1
            else:
                fn += 1
        tpr = tp / pos
        fpr = fp / neg
        fnr = fn / pos
        pts.append(
            {
                "threshold": t,
                "tpr": tpr,
                "fpr": fpr,
                "fnr": fnr,
                "precision": (tp / (tp + fp)) if (tp + fp) else 0.0,
                "recall": tpr,
            }
        )
    return pts


def _auc_from_roc(points: list[dict[str, float]]) -> float:
    pts = sorted(points, key=lambda p: p["fpr"])
    auc = 0.0
    for i in range(1, len(pts)):
        x1, y1 = pts[i - 1]["fpr"], pts[i - 1]["tpr"]
        x2, y2 = pts[i]["fpr"], pts[i]["tpr"]
        auc += (x2 - x1) * (y1 + y2) * 0.5
    return max(0.0, min(1.0, auc))


def _eer(points: list[dict[str, float]]) -> tuple[float, float]:
    best = min(points, key=lambda p: abs(p["fpr"] - p["fnr"]))
    eer = (best["fpr"] + best["fnr"]) * 0.5
    return float(eer), float(best["threshold"])


def _threshold_for_far(points: list[dict[str, float]], target_far: float) -> float:
    valid = [p for p in points if p["fpr"] <= target_far]
    if not valid:
        return max(p["threshold"] for p in points)
    # maximize recall under FAR constraint
    pick = max(valid, key=lambda p: (p["recall"], p["threshold"]))
    return float(pick["threshold"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate fused fingerprint thresholds from labeled scores.")
    parser.add_argument("--input", required=True, help="CSV/JSON labeled scores file")
    parser.add_argument("--out", default="output/calibration_report.json", help="Output report json path")
    parser.add_argument("--w-min", type=float, default=0.6)
    parser.add_argument("--w-mcc", type=float, default=0.3)
    parser.add_argument("--w-orb", type=float, default=0.1)
    parser.add_argument("--target-far-high", type=float, default=0.01)
    parser.add_argument("--target-far-medium", type=float, default=0.05)
    parser.add_argument("--target-far-low", type=float, default=0.10)
    parser.add_argument(
        "--infer-label-from-status",
        action="store_true",
        help="Infer missing label from status (HIGH/MEDIUM=>1, LOW/NO MATCH/INCONCLUSIVE=>0).",
    )
    args = parser.parse_args()

    samples = load_samples(
        Path(args.input),
        infer_label_from_status=args.infer_label_from_status,
    )
    sl = [(fused_score(s, args.w_min, args.w_mcc, args.w_orb), s.label) for s in samples]
    roc = _roc_points(sl)
    auc = _auc_from_roc(roc)
    eer, eer_thr = _eer(roc)

    thr_high = _threshold_for_far(roc, args.target_far_high)
    thr_med = _threshold_for_far(roc, args.target_far_medium)
    thr_low = _threshold_for_far(roc, args.target_far_low)

    # enforce monotonic descending HIGH >= MEDIUM >= LOW
    thr_high = max(thr_high, thr_med, thr_low)
    thr_med = min(thr_high, max(thr_med, thr_low))
    thr_low = min(thr_med, thr_low)

    report = {
        "summary": {
            "n_samples": len(samples),
            "n_genuine": sum(1 for s in samples if s.label == 1),
            "n_impostor": sum(1 for s in samples if s.label == 0),
            "weights": {
                "FUSION_W_MINUTIAE": args.w_min,
                "FUSION_W_MCC": args.w_mcc,
                "FUSION_W_ORB": args.w_orb,
            },
            "auc": round(auc, 6),
            "eer": round(eer, 6),
            "eer_threshold": round(eer_thr, 4),
        },
        "recommended_thresholds": {
            "FUSED_THRESHOLD_HIGH": round(thr_high, 2),
            "FUSED_THRESHOLD_MEDIUM": round(thr_med, 2),
            "FUSED_THRESHOLD_LOW": round(thr_low, 2),
        },
        "target_far": {
            "high": args.target_far_high,
            "medium": args.target_far_medium,
            "low": args.target_far_low,
        },
        "config_snippet": {
            "FUSION_W_MINUTIAE": args.w_min,
            "FUSION_W_MCC": args.w_mcc,
            "FUSION_W_ORB": args.w_orb,
            "FUSED_THRESHOLD_HIGH": round(thr_high, 2),
            "FUSED_THRESHOLD_MEDIUM": round(thr_med, 2),
            "FUSED_THRESHOLD_LOW": round(thr_low, 2),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Calibration complete.")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("Recommended thresholds:", report["recommended_thresholds"])


if __name__ == "__main__":
    main()
