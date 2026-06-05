# خارطة الطريق الموحّدة

## الحالة العامة (2026-06-05)

| المحور | التقدّم | ملاحظة |
|--------|---------|--------|
| A — جوهر المطابقة (CORE) | ~85% | Sprint 0–3 منجزة؛ معايرة تحتاج بيانات FVC/حقيقية |
| B — بنية ومعالجة (DEV) | ~70% | segmentation + tests؛ batch و FVC جزئي |
| C — مراحل 1–5 (PHASES) | ~60% | ملفات جاهزة؛ **التكامل مع `/analyze` ناقص** |
| منصة التشغيل | ~90% | FastAPI، SSE، Telegram inbox، تقارير AR→HTML |
| تكامل Kali (`dev/global-upgrade`) | 🟡 | Bozorth، Gabor enhancer، `GLOBAL_UPGRADE` — دمج جزئي |

---

## ما يعمل اليوم في الإنتاج ✅

- واجهة `/` — رفع مرجعية + مقارنة، فحص سريع/عميق، منطقة (كامل/مربع/شبكة)
- بث مباشر (SSE) — مراحل المعالجة + مطابقة + تقرير
- `fused_score` — Minutiae + MCC + Bozorth (+ ORB اختياري عبر `USE_ORB_FUSION`)
- Quality gate — `INCONCLUSIVE` عند جودة/نقاط منخفضة
- تقرير عربي — تحميل HTML (PDF العربي معطّل عمداً)
- تيليجرام — رفع صور → `output/telegram_inbox/` + `/analyze` اختياري
- اختبارات — `pytest tests/` (matcher، ref_grid، report، telegram، …)
- معايرة — `scripts/calibrate_thresholds.py` + قالب CSV

---

## فجوات حرجة (أولوية التنفيذ)

راجع [PHASE_6_INTEGRATION.md](PHASE_6_INTEGRATION.md) للتفاصيل.

1. **ربط `compare_engine` + التصنيف** بمسار `run_matching_pipeline` (الويب يستخدم `utils/matcher` فقط).
2. **دمج `landmark_matcher`** في `fused_score` أو عرض منفصل في التقرير.
3. **تفعيل محرر يدوي** — `routers/editor.py` غير مُضمَّن في `server/server.py`.
4. **حفظ نتائج التحليل في SQL** — الجداول موجودة؛ المسار الحالي يعتمد `audit_log.jsonl` أكثر.
5. **Batch analyze** — غير منفّذ.
6. **FVC2004 تلقائي** — `tests/test_on_fvc.py` موجود؛ يحتاج `FVC2004_PATH` وبيانات.

---

## الجدول الزمني المقترح

### المرحلة 6 — تكامل (2–3 أسابيع)

| الأسبوع | المهام |
|---------|--------|
| 1 | Editor router + صفحة `/editor`؛ تصنيف في pipeline؛ إصلاحات منطقة/زوم (✅ ref_grid cells) |
| 2 | Landmarks في التقرير والمطابقة؛ persist Match/Fingerprint من التحليل |
| 3 | اختبارات تكامل E2E؛ توثيق ENV؛ دمج `dev/global-upgrade` في main |

### المرحلة 7 — قياس وجودة (2 أسابيع)

| المهمة | الملفات |
|--------|---------|
| تشغيل baseline على `evaluation/` | `evaluation/run_baseline.py` |
| معايرة من `pairs.csv` حقيقية | `scripts/calibrate_thresholds.py` |
| FVC EER/AUC | `tests/test_on_fvc.py`, `scripts/test_fvc.py` (جديد) |
| تقرير GAP | `evaluation/GAP_REPORT.md` |

### المرحلة 8 — توسعات (اختياري)

- Batch ZIP analyze
- Auto-tune denoise من الجودة
- ML لتصنيف الإصبع
- PDF عربي (WeasyPrint أو بديل)

---

## خريطة الملفات ↔ الخطة

```
docs/plans/
├── ROADMAP.md              ← أنت هنا
├── CORE_DEVELOPMENT_PLAN.md ← Sprint 0–3 (منجز)
├── DEVELOPMENT_PLAN.md      ← بنية + batch + FVC
├── PHASE_1_2 … PHASE_3_4_5  ← تفاصيل الميزات
├── COMPLETE_SUMMARY.md      ← ملخص (يُحدَّث مع الحالة الحقيقية)
├── PHASE_6_INTEGRATION.md   ← قائمة مهام التكامل
└── GLOBAL_UPGRADE.md        ← Bozorth / Gabor (Kali)
```

---

## قرارات معتمدة

| القرار | السبب |
|--------|--------|
| عربي → HTML فقط | PDF لا يعرض RTL بشكل صحيح |
| ORB افتراضيًا معطّل | ضعيف على البصمات الحقيقية |
| منطقة قبل zoom على الخادم | مربع التحديد على الصورة الأصلية |
| `ref_grid_cells` فارغ = بصمة كاملة | تجنّب قصّ القطاع 0 تلقائياً |
| تيليجرام: حفظ ثم كود | `TELEGRAM_AUTO_ANALYZE=0` افتراضي |

---

## تعريف «مكتمل»

- **كود مكتمل:** الملفات والدوال موجودة ومُختبرة جزئياً.
- **منتج مكتمل:** الميزة تعمل من الواجهة/API بدون خطوات يدوية للمطوّر.
- **خطة مكتملة:** كل بنود ROADMAP + PHASE_6 في حالة ✅ منتج.

**الهدف الحالي:** رفع «منتج مكتمل» للمحور C وإغلاق فجوات المرحلة 6.
