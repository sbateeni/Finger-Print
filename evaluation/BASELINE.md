# Offline baseline evaluation (Finger-Print)

This folder adds **metrics only** (`FAR` / `FRR` / `EER`, ROC points) borrowed from **Finger-print-pro**, wired to **this repo’s pipeline** (`process_form_analysis` + `run_matching_pipeline`) instead of Flask/SIFT. It **does not replace** the forensic web matcher.

## 1. Pair manifest CSV

Columns: `img1`, `img2`, `label`

- `img1` = reference (same role as «البصمة الأصلية» in the UI).
- `img2` = query / partial (same role as «البصمة المقارنة»).
- `label` = `genuine` (same identity) or `impostor` (different identities).

Paths are resolved relative to the **repository root**.

Example: `evaluation/pairs.csv`.

## 2. Run baseline

From the repo root:

```bash
python evaluation/run_baseline.py --pairs evaluation/pairs.csv
```

Options:

| Flag | Meaning |
|------|---------|
| `--score {fused,match,mcc}` | Which numeric score feeds EER/ROC (`fused_score` recommended). |
| `--out-json` | Default `evaluation/baseline_metrics.json` |
| `--out-csv` | Default `evaluation/roc_points.csv` |

By default the baseline script **disables the runtime quality gate** so small demo pairs still produce scores. Use `--respect-quality-gate` to match the web UI exactly.

## 3. Interpret

- Lower **EER** is better when you have enough pairs.
- With only 1 genuine + 1 impostor row, metrics are **illustrative** only.
- Prefer tracking **relative EER/FAR** before vs after matcher or threshold changes on a fixed CSV.

## 4. Optional tests

```bash
pip install pytest
pytest tests/test_evaluation_metrics.py
```
