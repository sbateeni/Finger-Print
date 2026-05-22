# لا تستبدل المشروع بأكواد الدردشة الجاهزة

## أوامر pip التي **لا تعمل**

```bash
pip install nfiq2              # ❌ غير موجود على PyPI
pip install fingerprint-matcher # ❌ غير موجود على PyPI
```

## ما هو مُنفَّذ بالفعل في Finger-Print

| المقترح في الدردشة | التنفيذ الفعلي في المشروع |
|---------------------|---------------------------|
| `pf.extract_minutiae()` | `pf.minutiae_extraction()` في `features/extractor.py` |
| `FingerprintMatcher` pip | `matching/compare_engine.py` + `utils/matcher.py` (RANSAC + MCC) |
| `import nfiq2` | `preprocessing/quality.py` — heuristic + **NFIQ2 CLI** اختياري |
| محاذاة Core | `matching/alignment.py` — `align_by_core_point` |
| محاذاة Triplets | `align_by_triplets` — `USE_TRIPLET_ALIGNMENT=1` |
| بوت register/match | `services/telegram_templates.py` + `/register` `/match` |

## التشغيل الصحيح

```powershell
# Windows
.\run_dev.ps1

# أو بدون إعادة تحميل تلقائي (أقل مشاكل)
$env:LIVE_RELOAD="0"; python run_app.py
```

```bash
# Linux/Kali
./run_dev.sh
```

## pyfing (اختياري)

```bash
pip install -r requirements-ml.txt
```

في `.env`:

```env
USE_PYFING_EXTRACTION=1
```

## لماذا التقرير (7) = (5) بنفس 32%؟

غالباً لأنك تشغّل **نسخة قديمة** قبل `git pull`، أو لم تُعد تحليل نفس الزوج بعد التحديث.

بعد السحب، أعد رفع **نفس الصورتين** من الويب أو تيليجرام وتحقق من الحقول:

- `core_prealignment`
- `alignment_gain_matches`
- `minutiae_extraction`

## NFIQ2 الحقيقي

ثبّت binary من NIST ثم:

```env
NFIQ2_CLI_PATH=C:/path/to/nfiq2.exe
QUALITY_BACKEND=nfiq2_cli
```
