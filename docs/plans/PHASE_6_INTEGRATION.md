# المرحلة 6 — التكامل وإغلاق الخطة

بعد اكتمال **كتابة** المراحل 1–5 (ملفات منفصلة)، هذه المرحلة توصلها بمسار العمل اليومي: الويب، التقارير، وقاعدة البيانات.

---

## الحالة الحالية

| المكوّن | الكود | مربوط بـ `/analyze` | ملاحظة |
|---------|-------|---------------------|--------|
| `features/fingerprint_classifier.py` | ✅ | ❌ | يعمل في `matching/compare_engine` و Telegram templates |
| `features/minutiae_landmarks.py` | ✅ | ❌ | لا يُستدعى من `branch.py` |
| `matching/landmark_matcher.py` | ✅ | ❌ | غير في `utils/fusion.py` |
| `matching/compare_engine.py` | ✅ | ❌ | الويب: `utils/matcher` + `apply_fusion_to_match` |
| `services/manual_annotation_service.py` | ✅ | ❌ | API جاهز |
| `routers/editor.py` | ✅ | ❌ | **غير مُضمَّن في `server/server.py`** |
| `templates/manual_editor.html` | ✅ | ❌ | لا يوجد route `/editor` |
| `database/models.py` (حقول Phase) | ✅ | 🟡 | CRUD موجود؛ الحفظ من التحليل نادر |

---

## Sprint 6.1 — محرر يدوي (يوم 1–2)

### المهام

- [ ] في `server/server.py`:
  ```python
  from routers import pages, report, analysis, editor
  app.include_router(editor.router)
  ```
- [ ] في `routers/pages.py` (أو `editor`): `GET /editor` → `manual_editor.html`
- [ ] ربط `static/js/manual_editor.js` عبر `/static/js/...`
- [ ] اختبار: `POST /api/editor/add-minutia` مع `fingerprint_id` من DB
- [ ] رابط من `templates/index.html`: «مراجعة يدوية» بعد التحليل (اختياري: يفتح editor بمعرّف case)

### معايير القبول

- فتح `http://127.0.0.1:8000/editor` يعرض الواجهة
- إضافة/حذف نقطة تُحدَّث `minutiae_data` في SQLite

---

## Sprint 6.2 — تصنيف + رفض مبكر (يوم 2–4)

### المهام

- [ ] دالة `apply_classification_gate(ro, rp, form_ctx)` في `services/analysis_service/pipeline.py`
- [ ] استدعاء `classify_fingerprint` لكل فرع (صورة `processed` أو gray قبل resize)
- [ ] إذا `not compatible` → نتيجة `INCONCLUSIVE` أو `NO MATCH` مع `classification_check_reason`
- [ ] إظهار الحقول في `forensic_report.html` / `report_generator.py`
- [ ] اختبار: بصمتان بـ metadata مختلف (`finger_type`) → رفض قبل MCC

### الملفات

- `services/analysis_service/pipeline.py`
- `utils/report_generator.py`
- `tests/test_classification_gate.py` (جديد)

---

## Sprint 6.3 — العلامات التشريحية في المسار (يوم 4–6)

### المهام

- [ ] في `services/analysis_service/branch.py` بعد `extract_minutiae`:
  ```python
  from features.minutiae_landmarks import extract_landmarks
  minutiae, _ = extract_landmarks(minutiae, image=proc, ridge_image=ridges)
  ```
- [ ] في `utils/fusion.py` أو `orb_matcher.combined_verdict`:
  - استدعاء `compare_landmarks(mo, mp)` 
  - وزن اختياري `FUSION_W_LANDMARK` في `config/config.py`
- [ ] تقرير: جدول توزيع العلامات (8 أنواع)
- [ ] اختبار: `tests/test_landmark_fusion.py`

---

## Sprint 6.4 — حفظ في قاعدة البيانات (يوم 6–8)

### المهام

- [ ] بعد `run_matching_pipeline`: `database/crud.create_fingerprint` × 2 + `create_match`
- [ ] تخزين: `fingerprint_classification`, `landmarks`, `match_details` (JSON كامل)
- [ ] `case_reference` + `operator_name` من النموذج
- [ ] مسار اختياري: `--no-db` في التطوير عبر `WRITE_TO_DB=0`
- [ ] `scripts/init_db.py` في `run_dev` (اختياري)

### معايير القبول

- بعد تحليل واحد: صف في `matches` + صفّان في `fingerprints`
- التقرير يبقى في `output/` كما هو

---

## Sprint 6.5 — دمج global-upgrade (يوم 8–10)

مرجع: فرع `dev/global-upgrade` / ملفات مدمجة جزئياً.

- [x] Bozorth في `utils/fusion.py` + `config`
- [x] `preprocessing/enhancer.py` + `USE_GABOR_ENHANCER`
- [x] توثيق في `docs/plans/GLOBAL_UPGRADE.md`
- [ ] `pip install -r requirements-global.txt` اختياري في README
- [ ] `pytest tests/test_global_upgrade.py tests/test_bozorth_fusion.py`

---

## Sprint 6.6 — إكمال DEVELOPMENT_PLAN (متفرّق)

| البند | الحالة | الإجراء |
|-------|--------|---------|
| `segment_fingerprint` | ✅ | — |
| `auto_tune_parameters` | ⬜ | دالة في `preprocessing/quality.py` + خيار UI |
| Batch `/batch-analyze` | ⬜ | `routers/analysis.py` + ZIP |
| FVC2004 | 🟡 | `scripts/run_fvc_evaluation.py` يستدعي `test_on_fvc` |
| Unit tests image_processing | 🟡 | إضافة `tests/test_preprocess.py` |

---

## قائمة تحقق «الخطة مكتملة»

عند اكتمال كل ما يلي، تُعتبر `docs/plans` **مغلقة للإصدار 2.4**:

### جوهر (CORE)
- [x] baseline منفصل عن post-alignment
- [x] fused_score + quality gate
- [x] معايرة thresholds (سكربت)
- [ ] معايرة مُعتمدة في `config` من تقرير حقيقي

### مراحب 1–5 (منتج)
- [ ] تصنيف في `/analyze`
- [ ] landmarks في pipeline + تقرير
- [ ] `/editor` يعمل
- [ ] DB يحفظ النتائج

### بنية (DEV)
- [ ] Batch أو قرار «مؤجّل» موثّق في ROADMAP
- [ ] FVC تقرير EER مُسجّل في `evaluation/`

### تشغيل
- [x] `docs/plans/README.md` + ROADMAP
- [x] إصلاح زوم المرجعية (ref_grid_cells)
- [ ] CI: `pytest` على PR

---

## أوامر تحقق

```powershell
cd C:\Users\HP\Documents\GitHub\Finger-Print
python -m pytest tests/ -q --tb=no
.\run_dev.ps1
```

```bash
python -m pytest tests/ -q
./run_dev.sh
python scripts/calibrate_thresholds.py --help
```

---

## ترتيب التنفيذ الموصى به

```
6.1 Editor → 6.2 Classification → 6.3 Landmarks → 6.4 DB → 6.5 Global → 6.6 Batch/FVC
```

يمكن تنفيذ 6.2 و 6.3 بالتوازي بعد 6.1.
