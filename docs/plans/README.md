# خطط التطوير — Finger-Print

دليل موحّد لجميع خطط المشروع. اقرأ **ROADMAP.md** أولاً للحالة الحالية، ثم **PHASE_6_INTEGRATION.md** للمهام المتبقية.

---

## الملفات

| الملف | المحتوى | الجمهور |
|--------|---------|---------|
| [ROADMAP.md](ROADMAP.md) | خارطة طريق موحّدة + مصفوفة الحالة | الجميع |
| [CORE_DEVELOPMENT_PLAN.md](CORE_DEVELOPMENT_PLAN.md) | جوهر المطابقة: baseline، fusion، معايرة، quality gate | مطوّر الخوارزمية |
| [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) | معالجة صور، DB، batch، FVC | مطوّر البنية |
| [PHASE_1_2_IMPLEMENTATION.md](PHASE_1_2_IMPLEMENTATION.md) | تصنيف البصمة + 8 علامات تشريحية | تفاصيل المرحلة 1–2 |
| [PHASE_3_4_5_IMPLEMENTATION.md](PHASE_3_4_5_IMPLEMENTATION.md) | محرر يدوي + landmark matcher + DB | تفاصيل المرحلة 3–5 |
| [COMPLETE_SUMMARY.md](COMPLETE_SUMMARY.md) | ملخص المراحل 1–5 + أمثلة | مراجعة سريعة |
| [PHASE_6_INTEGRATION.md](PHASE_6_INTEGRATION.md) | **التكامل والإغلاق** — ما يُنفَّذ الآن | التنفيذ القادم |
| [GLOBAL_UPGRADE.md](GLOBAL_UPGRADE.md) | Gabor، Bozorth، NFIQ2 — فرع Kali | DevOps / خوارزمية |

---

## مسارات التطوير (ثلاثة محاور)

```
محور A — الجوهر (CORE)     baseline + fused_score + معايرة + INCONCLUSIVE
محور B — البنية (DEV)      segmentation + tests + batch + FVC
محور C — الميزات (PHASES)  تصنيف + landmarks + محرر يدوي + DB
```

المحور C: **الكود موجود** لكن جزء كبير **غير موصول** بمسار الويب الرئيسي (`/analyze`).

---

## رموز الحالة

| الرمز | المعنى |
|--------|--------|
| ✅ | منجز ومُستخدم في الإنتاج |
| 🟡 | منجز في الكود، يحتاج تكامل أو اختبار |
| ⬜ | مخطط، لم يُنفَّذ |
| 🔧 | إصلاح/انحدار (مثل زوم المرجعية 2026-06) |

---

## تشغيل سريع

```powershell
.\run_dev.ps1
# http://127.0.0.1:8000
```

```bash
./run_dev.sh
# http://0.0.0.0:8000
```

هيكل المشروع: [docs/PROJECT_LAYOUT.md](../PROJECT_LAYOUT.md)

---

**آخر تحديث للخطة:** 2026-06-05
