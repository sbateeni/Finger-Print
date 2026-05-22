"""
Build a calibration dataset template from output/audit_log.jsonl.

Usage:
  python build_calibration_dataset.py --out output/calibration_dataset_template.csv

Result:
  CSV with score columns ready for `calibrate_thresholds.py`.
  The `label` column is intentionally empty for manual annotation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _orb_conf_to_score(conf: str | None) -> float:
    c = (conf or "").upper()
    if c == "HIGH":
        return 80.0
    if c == "MEDIUM":
        return 55.0
    if c == "LOW":
        return 30.0
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Create calibration template CSV from audit log.")
    parser.add_argument("--audit", default="output/audit_log.jsonl")
    parser.add_argument("--out", default="output/calibration_dataset_template.csv")
    args = parser.parse_args()

    audit_path = Path(args.audit)
    if not audit_path.exists():
        raise FileNotFoundError(f"Audit log not found: {audit_path}")

    rows: list[dict[str, Any]] = []
    with audit_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            params = rec.get("params") or {}
            orb_score = _to_float(rec.get("orb_score"), _orb_conf_to_score(rec.get("orb_confidence")))
            rows.append(
                {
                    "label": "",  # annotate manually: 1 genuine / 0 impostor
                    "sha256_original": rec.get("sha256_original", ""),
                    "sha256_partial": rec.get("sha256_partial", ""),
                    "report_filename": rec.get("report_filename", ""),
                    "status": rec.get("status", ""),
                    "match_score": _to_float(rec.get("match_score")),
                    "minutiae_score": _to_float(rec.get("match_score")),
                    "mcc_score": _to_float(rec.get("mcc_score")),
                    "orb_score": orb_score,
                    "fused_score": _to_float(rec.get("fused_score")),
                    "quality_gate_failed": bool(rec.get("quality_gate_failed", False)),
                    "quality_gate_reason": rec.get("quality_gate_reason", ""),
                    "match_distance_threshold": params.get("MATCH_DISTANCE_THRESHOLD", ""),
                    "match_angle_threshold_deg": params.get("MATCH_ANGLE_THRESHOLD_DEG", ""),
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "sha256_original",
        "sha256_partial",
        "report_filename",
        "status",
        "match_score",
        "minutiae_score",
        "mcc_score",
        "orb_score",
        "fused_score",
        "quality_gate_failed",
        "quality_gate_reason",
        "match_distance_threshold",
        "match_angle_threshold_deg",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Template dataset written: {out_path}")
    print(f"Rows: {len(rows)}")
    print("Next: fill 'label' column (1/0), then run calibrate_thresholds.py directly on this file.")


if __name__ == "__main__":
    main()
