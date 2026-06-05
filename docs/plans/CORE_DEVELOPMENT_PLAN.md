# خطة تطوير الجوهر (Core) — قبل التنفيذ

## 1) خط الأساس الحالي (Baseline من آخر تشغيل)

### مخرجات المطابقة (الحالة الحالية)
- `total_original`: 154
- `total_partial`: 67
- `matched_points`: 50
- `match_score`: 74.63%
- `dice_score`: 45.25%
- `status`: HIGH MATCH
- `orb_matches`: 47
- `orb_confidence`: MEDIUM
- `mcc_similarity`: 35.72%
- `mcc_pairs`: 67

### ملاحظات فورية على جودة المنطق الحالي
- النتيجة قوية ظاهرياً، لكن هناك إشارات اتساق تحتاج تدقيق:
  - `baseline_matched = 0` مع `baseline_match_score = 74.63%` غير منطقية غالباً (لازم مراجعة حساب/عرض baseline قبل المحاذاة).
  - يجب التأكد أن أي تحسين بعد المحاذاة يظهر كزيادة حقيقية عن baseline وليس إعادة استخدام نفس القيمة النهائية.

### ملاحظات عرض/تقرير
- ظهور `????` في النص العربي يعني مشكلة عرض/ترميز في بعض المسارات (خصوصاً عند التصدير/الطباعة).
- ملف HTML يحتوي `meta charset="UTF-8"`، لكن يلزم توحيد الترميز والخطوط العربية عبر كل قنوات التوليد (HTML + print/PDF).

---

## 2) المطلوب تطويره (الجوهر فقط)

الهدف: تحسين **دقة القرار** و**ثبات النتيجة**، وليس الشكل أو الواجهة.

### الأولوية A: ثبات المطابقة الأساسية
1. **تصحيح منطق baseline قبل المحاذاة**
   - حفظ baseline الحقيقي قبل أي alignment.
   - فصل واضح بين:
     - `baseline_matched`, `baseline_match_score`
     - `matched_points`, `match_score` بعد alignment.
2. **توثيق شفاف للتحول**
   - إظهار `alignment_gain_matches` و`alignment_gain_score` بدقة.

### الأولوية B: تقوية الدليل متعدد القنوات
1. **MCC + Geometric + ORB Fusion**
   - الإبقاء على كل قناة كـ sub-score مستقل.
   - الحكم النهائي من `fused_score` موزون، وليس من قناة واحدة.
2. **قواعد جودة قبل الحكم**
   - إذا جودة الصورة/النقاط منخفضة: النتيجة `INCONCLUSIVE` بدل قرار حاد.

### الأولوية C: المعايرة الإحصائية
1. **استبدال العتبات اليدوية بعتبات معايرة**
   - بناء dataset (genuine/impostor).
   - حساب ROC/EER وتحديد thresholds مدعومة رقمياً.
2. **تثبيت عتبات لكل قناة + fused**
   - عتبات منفصلة لـ minutiae وorb وmcc وfused.

---

## 3) تعديلات الكود المطلوبة (تضاف للخطة قبل التنفيذ)

## A) تصحيح baseline/alignment (إجباري أولاً)
- `utils/matcher.py`
  - مراجعة إنشاء baseline وتأكيد أنه يُحسب قبل أي transform.
  - إضافة حقل: `alignment_gain_score`.
  - اختبار وحدات بسيط لحالة baseline=0 للتأكد أن baseline_score يكون 0 أيضاً.

- `utils/forensic.py`
  - تحديث نص `alignment_summary_ar` ليعرض gain بالنقاط والنسبة.
  - إضافة تحذير اتساق إذا البيانات متضاربة.

- `server.py` و`templates/index.html` و`static/live_analyze.js`
  - عرض baseline/final من حقول منفصلة فقط (بدون إعادة تدوير قيمة واحدة).

## B) صلابة التقرير والترميز (مهم قبل أي مرحلة جديدة)
- `utils/report_generator.py`
  - فرض الكتابة بترميز UTF-8 بشكل موحد.
  - التأكد أن كل النصوص العربية لا تمر بسلسلة تحويل تكسر الأحرف.
  - توحيد CSS fonts بخطوط عربية fallback مناسبة للطباعة.

- `server.py` (download route)
  - التأكد من `media_type="text/html; charset=utf-8"` عند تقديم التقرير.

## C) تحسين قرار الجوهر (بعد إصلاح A وB)
- `utils/mcc.py`, `utils/matcher.py`, `utils/orb_matcher.py`
  - إنشاء `fused_score` رسمي.
  - توثيق أوزان الدمج في `config.py`.

- `config.py`
  - فصل:
    - thresholds تشغيل مؤقتة
    - thresholds معايرة (لاحقاً من سكربت المعايرة).

---

## 4) خطة التنفيذ المختصرة (Sprint-ready)

1. **Sprint 0 (تصحيح إلزامي)**  
   baseline/alignment consistency + report encoding.

2. **Sprint 1 (Core Scoring)**  
   fused score (Minutiae + MCC + ORB) مع output واضح لكل قناة.

3. **Sprint 2 (Calibration)**  
   سكربت معايرة + تحديث thresholds من بيانات حقيقية.

4. **Sprint 3 (Quality Gate)**  
   إدخال `INCONCLUSIVE` ورفض القرار عند quality منخفضة.

---

## 5) معايير قبول كل مرحلة

### بعد Sprint 0
- لا يوجد تناقض baseline/final في أي تقرير.
- لا يظهر `????` في HTML الناتج من المسار المعتاد.

### بعد Sprint 1
- كل نتيجة تعرض:
  - `score_minutiae`
  - `score_mcc`
  - `score_orb`
  - `score_fused`
- الحكم النهائي مبني فقط على `score_fused`.

### بعد Sprint 2
- تقرير معايرة يتضمن ROC/EER وعتبات معتمدة.
- انخفاض الأخطاء مقارنة بالعتبات اليدوية.

### بعد Sprint 3
- حالات الجودة الضعيفة تصنف `INCONCLUSIVE` بدلاً من قرارات مضللة.

---

## 6) قرار البدء

Sprint 0–3 **منجزة** للمسار الرئيسي `/analyze`.  
الخطوة التالية: **[PHASE_6_INTEGRATION.md](PHASE_6_INTEGRATION.md)** (ربط المراحل 1–5 + editor + DB).

---

## تحديث التنفيذ (منجز)

| Sprint | الحالة | ملاحظات |
|--------|--------|---------|
| 0 baseline + ترميز | ✅ | `utils/matcher.py`, `report_generator` |
| 1 fused_score | ✅ | Minutiae + MCC + ORB(opt) + Bozorth |
| 2 معايرة | 🟡 | `scripts/calibrate_thresholds.py` — يحتاج dataset موسوم |
| 3 quality gate | ✅ | `INCONCLUSIVE` في pipeline |

### Sprint 4 (مقترح — بعد Phase 6)

- [ ] ربط `classification` + `landmarks` في pipeline
- [ ] اعتماد عتبات من `calibration_report.json` في `config`
- [ ] تقرير ROC/EER في `evaluation/`

### إصلاحات حديثة 🔧

- قصّ المرجعية التلقائي (قطاع شبكة 0): `resolve_grid_cells_for_crop` في `ref_grid.py`
- عربي PDF → HTML: `resolve_report_download`
- تيليجرام: حفظ صور في `output/telegram_inbox/`

---

## روابط

- [ROADMAP.md](ROADMAP.md)
- [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)
- [PHASE_6_INTEGRATION.md](PHASE_6_INTEGRATION.md)
