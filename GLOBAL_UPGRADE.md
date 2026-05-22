# خطة التطوير العالمي — Finger-Print

الفرع: `dev/global-upgrade`  
المستودع: https://github.com/sbateeni/Finger-Print

## التشغيل (Kali / Linux / Windows)

```bash
git clone https://github.com/sbateeni/Finger-Print.git
cd Finger-Print
git checkout dev/global-upgrade   # بعد دمج الفرع
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # TELEGRAM_BOT_TOKEN
python scripts/init_db.py
./run_dev.sh                      # web + Telegram
```

## ما تم تنفيذه

| المرحلة | الحالة | الملفات |
|---------|--------|---------|
| 1 — تحسين Gabor + CN | ✅ | `preprocessing/enhancer.py`, `utils/image_processing.py`, `utils/minutiae_extractor.py` |
| 2 — محاذاة + Bozorth | ✅ (اختياري) | `matching/alignment.py`, `matching/bozorth_matcher.py` |
| 3 — جودة NFIQ2 + FVC | ✅ هيكل | `evaluation/quality.py`, `tests/test_on_fvc.py` |
| 4 — Telegram | ✅ موجود | `bot/telegram_bot.py` — مقارنتان + PDF؛ أوامر إضافية لاحقاً |

## متغيرات البيئة

| المتغير | الافتراضي | الوصف |
|---------|-----------|--------|
| `USE_GABOR_ENHANCER` | `1` | تفعيل محسّن Gabor قبل الثنائية |
| `USE_SKIMAGE_SKELETON` | `1` | هيكلة عبر scikit-image |
| `USE_BOZORTH_MATCHER` | `1` | دمج Bozorth في القرار النهائي |
| `BOZORTH_MATCH_THRESHOLD` | `25` | عتبة Bozorth |
| `FVC2004_PATH` | — | مسار قاعدة FVC للاختبار |

## مكتبات اختيارية

```bash
pip install -r requirements-global.txt
# أو يدوياً:
# pip install git+https://github.com/Utkarsh-Deshmukh/Fingerprint-Enhancement-Python
```

## الاختبارات

```bash
pytest tests/test_global_upgrade.py tests/test_evaluation_metrics.py -q
pytest tests/test_on_fvc.py -v   # يتطلب FVC2004_PATH
```

## ملاحظات

- المسار الإنتاجي الحالي: `services/analysis_service.py` + `utils/orb_matcher.py` (MCC + ORB + minutiae).
- Bozorth **مفعّل افتراضياً** في `combined_verdict` (وزن ~15–20%).
- Telegram: `/register` `/match` `/templates` + المسار القديم (صورتان + PDF).

## Pull Request

```bash
git push -u origin dev/global-upgrade
# ثم PR → main على GitHub
```
