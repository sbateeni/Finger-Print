# تصنيف النقاط الدقيقة — ملصق 36 نقطة + PDF الموصل

## 1) ملصق التعرف (36 نقطة)

القائمة التي أرسلتها **مطابقة** لمفتاح الملصق (Crimescene poster):

| النوع (EN) | العدد | نوع المحرك |
|------------|------:|------------|
| Ending Ridge | 12 | `endpoint` |
| Bifurcation | 23 | `bifurcation` |
| Island | 1 | `island` |
| **المجموع** | **36** | |

**ملاحظة:** المحرك **لا يثبت 36 إحداثية ثابتة** من الملصق — يستخرج النقاط تلقائياً من صورتك بعد Gabor + thinning. العدد الفعلي يختلف حسب جودة الصورة.

## 2) ملف PDF `132-137-1-PB.pdf`

بحث أكاديمي (جامعة الموصل) عن الحكومة الإلكترونية + بصمة الإصبع. يذكر **سبعة** سمات للخطوط (أوسع من الملصق):

| PDF (عربي) | PDF (EN) | في المحرك |
|------------|----------|-----------|
| نهاية الخط | ending | ✅ `endpoint` |
| تفرع | bifurcation | ✅ `bifurcation` |
| بحيرة | lake | ✅ `lake` |
| خط قصير | short ridge | ✅ `island` |
| نقطة | dot | ✅ `dot` |
| حافة بارزة | divergence | ✅ `divergence` |
| معبر / جسر | bridge | ✅ `bridge` |

معمارية النظام من PDF (تسجيل قالب + مطابقة 1:1 / 1:N) موجودة في:

- الويب: تحليل زوج صور + تقرير PDF
- تيليجرام: `/register` + `/match` + تحليل عميق لزوجين

## 3) ما تم دمجه في الكود

| المكون | الملف |
|--------|-------|
| استخراج CN + island | `utils/minutiae_extractor.py` |
| فلترة حواف/معزولة | `features/minutiae_filter.py` |
| جدول الملصق 36 | `features/minutiae_taxonomy.py` |
| مطابقة حسب النوع | `utils/matcher.py` |
| محاذاة Core + RANSAC | `matching/alignment.py` |
| جودة صورة | `preprocessing/quality.py` |

## 4) التحقق على صورة الملصق (النموذج)

```bash
python scripts/analyze_minutiae_types.py path/to/poster_fingerprint.png
```

قارن `minutiae_by_type` في التقرير أو مخرج السكربت مع جدول الـ36 (تقريبي — الاستخراج تلقائي).

إعداد `.env`:

```env
ENABLE_EXTENDED_MINUTIAE=1
```

## 5) التحقق من التقرير

في تقرير PDF الجنائي ابحث عن:

- `minutiae_extraction`
- عدد `total_original` / `total_partial`
- توزيع الأنواع (إن وُجد في السجل)

للمقارنة مع الملصق التعليمي: استخدم صورة **نظيفة عالية التباين** — الصور الجزئية أو منخفضة الجودة تعطي عدداً أقل من 36.
