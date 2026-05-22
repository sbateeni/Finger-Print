# Matching pipeline

## Modules

| Module | Role |
|--------|------|
| `features/extractor.py` | Skeleton CN + optional **pyfing** (`USE_PYFING_EXTRACTION=1`) |
| `preprocessing/quality.py` | **QualityChecker** — heuristic or **NFIQ2 CLI** (`QUALITY_BACKEND`, `NFIQ2_CLI_PATH`) |
| `matching/alignment.py` | Core pre-align + RANSAC refinement |
| `matching/compare_engine.py` | **FingerprintMatcher** — grid/RANSAC + MCC (`MATCH_ENGINE_THRESHOLD=40`) |
| `matching/matcher.py` | Public `Matcher` facade |
| `utils/fusion.py` | Minutiae + MCC fusion (ORB off by default) |

## Pip packages that do **not** exist

- `pip install nfiq2` — use NIST NFIQ2 binary + `NFIQ2_CLI_PATH`
- `pip install fingerprint-matcher` — use `matching/compare_engine.py` instead
- `pip install sourceafis` — not available for Python

## Optional ML

```bash
pip install -r requirements-ml.txt
# then in .env: USE_PYFING_EXTRACTION=1
```
