# خطة تطوير جوهر البرنامج

> الحالة: [ROADMAP.md](ROADMAP.md) · التكامل: [PHASE_6_INTEGRATION.md](PHASE_6_INTEGRATION.md)

## ترتيب تنفيذ الخطة (الأهمية الأولى)

| # | البند | الحالة |
|---|--------|--------|
| 1 | Segmentation | ✅ `segment_fingerprint` في `utils/image_processing.py` |
| 2 | تقليل أخطاء minutiae | ✅ `features/minutiae_filter.py` + مرشحات |
| 3 | قاعدة البيانات | 🟡 `database/` + CRUD؛ حفظ من `/analyze` ⬜ |
| 4 | دمج ORB / Bozorth | ✅ `utils/fusion.py`؛ ORB افتراضيًا off |
| 5 | اختبارات وحدة | ✅ `tests/` (12+ ملف) |
| 6 | Batch Matching | ⬜ |
| 7 | FVC2004 | 🟡 `tests/test_on_fvc.py` |

---

## الخطة التفصيلية

### المرحلة 1: تحسين المعالجة المسبقة للصور

#### 1.1 تحسين استخراج منطقة البصمة (Segmentation) ✅
- **الملف**: `utils/image_processing.py` — `segment_fingerprint()` + `preprocess_image()`
- **اختبار مقترح**: `tests/test_preprocess.py`

#### 1.2 Auto-tuning للمعالجة ⬜
- **الملفات**: `preprocessing/quality.py`
- **الخطوات**:
  1. `auto_tune_parameters(img)` → `fast_denoise_h`, `gauss_ksize`
  2. خيار «Auto» في `templates/index.html`

---

### المرحلة 2: تحسين استخراج النقاط الدقيقة

#### 2.1 تقليل الأخطاء (False Minutiae) ✅
- `features/minutiae_filter.py` + `utils/minutiae_extractor.py`
- اختبارات: `tests/test_minutiae_filter.py`

#### 2.2 إضافة أنواع نقاط جديدة 🟡
- `features/minutiae_taxonomy.py` — dot, lake, bridge, …
- اختبارات: `tests/test_extended_minutiae.py`, `tests/test_minutiae_taxonomy.py`

---

### المرحلة 3: تحسين المطابقة

#### 3.1 تحسين دمج ORB / MCC / Bozorth ✅
- `utils/fusion.py`, `utils/orb_matcher.py`, `config/config.py`
- `USE_ORB_FUSION=0` افتراضي؛ `USE_BOZORTH_MATCHER=1`
- معايرة: `scripts/calibrate_thresholds.py`

#### 3.2 Batch Matching ⬜
- `services/analysis_service/batch.py` (جديد)
- `POST /batch-analyze` — ZIP مرجع + عدة queries

---

### المرحلة 4: تنفيذ قاعدة البيانات

#### 4.1 إعداد SQLAlchemy ✅
- `database/__init__.py`, `database/models.py` (User, Fingerprint, Match, Review)
- `fingerprint.db` في جذر المشروع

#### 4.2 دوال CRUD ✅ / حفظ من التحليل ⬜
- `database/crud.py` موجود
- **متبقٍ:** استدعاء من `pipeline.py` بعد كل تحليل — Sprint 6.4

---

### المرحلة 5: اختبارات الأداء والتصحيح

#### 5.1 اختبارات وحدة ✅
- `tests/test_matching_pipeline.py`, `test_ref_grid.py`, `test_report_pdf.py`, …
- تشغيل: `python -m pytest tests/ -q`

#### 5.2 اختبارات دقة المطابقة 🟡
- `evaluation/run_baseline.py`, `tests/test_on_fvc.py`
- **متبقٍ:** `scripts/run_fvc_evaluation.py` + توثيق `FVC2004_PATH` في `.env.example`
- مخرجات: تحديث `evaluation/baseline_metrics.json`, `evaluation/GAP_REPORT.md`

---

## إضافات منفذة خارج هذه الخطة الأصلية

| الميزة | الملفات |
|--------|---------|
| بث SSE مباشر | `services/analysis_service/streaming.py` |
| منطقة كاملة/مربع/شبكة | `static/region_select.js`, `ref_grid.py` |
| تيليجرام inbox | `services/telegram_inbox.py` |
| تقرير عربي HTML | `utils/report_pdf.py` |
| Gabor + Bozorth (Kali) | `preprocessing/enhancer.py`, `matching/bozorth_matcher.py` |
