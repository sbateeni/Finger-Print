# ترقية عالمية (Global Upgrade) — من فرع Kali

مرجع التنفيذ: `dev/global-upgrade` (مدمج جزئياً في `main`).

## الحالة

| مكوّن | الحالة |
|--------|--------|
| `preprocessing/enhancer.py` (Gabor) | ✅ |
| `matching/bozorth_matcher.py` | ✅ |
| `utils/bozorth_bridge.py` | ✅ في `utils/fusion.py` |
| `evaluation/quality.py` (NFIQ2 هيكل) | ✅ |
| `tests/test_global_upgrade.py` | ✅ |
| `requirements-global.txt` | ✅ |

## متغيرات `.env`

```env
USE_GABOR_ENHANCER=1
USE_SKIMAGE_SKELETON=1
USE_BOZORTH_MATCHER=1
BOZORTH_MATCH_THRESHOLD=25
FVC2004_PATH=   # اختياري لـ test_on_fvc
```

## اختبار

```bash
pip install -r requirements.txt
pip install -r requirements-global.txt   # اختياري
pytest tests/test_global_upgrade.py tests/test_bozorth_fusion.py -q
pytest tests/test_on_fvc.py -v            # يتطلب FVC2004_PATH
```

## تشغيل

```bash
./run_dev.sh
# أو Windows: .\run_dev.ps1
```

انظر أيضاً [ROADMAP.md](ROADMAP.md) و [PHASE_6_INTEGRATION.md](PHASE_6_INTEGRATION.md) (Sprint 6.5).
