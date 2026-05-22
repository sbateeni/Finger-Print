# Matching upgrades

## Implemented

- **Minutiae filter** (`features/minutiae_filter.py`): border + isolated-ridge cleanup for partial prints.
- **RANSAC alignment** (`matching/alignment.py`): refines translation/rotation/scale after grid search (`utils/matcher.py`).
- **Fusion** (`utils/fusion.py`): Minutiae + MCC by default; ORB only if `USE_ORB_FUSION=1`.
- **Quality gate** (`utils/quality_gate.py`): heuristic pre-check (NFIQ2-style threshold via `QUALITY_GATE_MIN_SCORE`).

## SourceAFIS / Bozorth3

There is no maintained `pip install sourceafis` for Python. For certified Bozorth3/NFIQ2 you would need:

- NIST NBIS (C) via subprocess, or
- SourceAFIS (.NET/Java) via CLI/JPype.

Current stack targets the same goals with RANSAC + MCC + improved minutiae filtering without a JVM.
