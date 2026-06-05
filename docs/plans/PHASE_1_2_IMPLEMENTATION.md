# خطة التطوير - المرحلة 1 و 2 مكتملة

## المتطلبات الأصلية

المستخدم طلب إضافة:
1. **تصنيف أنواع البصمات** - لمنع مطابقة بصمات مختلفة تماماً
2. **العلامات التشريحية الثمانية** - لتحسين دقة المطابقة
3. **واجهة تحكم يدوي** - لتصحيح الأخطاء والإضافة اليدوية

مع التأكيد على تطبيق التصنيف على **كلا البصمتين** (البصمة المرجعية والبصمة الاستعلام) وليس فقط على الأصل.

---

## المرحلة 1: تصنيف أنواع البصمات ✅ (مكتملة)

### الملفات المُنشأة والمعدّلة:

#### 1. `features/fingerprint_classifier.py` (جديد)
**الوظيفة**: محرك تصنيف أنواع البصمات

**الفئات الرئيسية**:
- `FingerType`: enum للأصابع (إبهام، سبابة، وسطى، بنصر، خنصر)
- `FingerprintRegion`: enum للمناطق (طرف الإصبع، جذور الأصابع، تحت السبابة)
- `FingerprintClassification`: كائن يحتوي على بيانات التصنيف
- `FingerprintClassifier`: محرك التصنيف الرئيسي

**الميزات**:
```python
# تصنيف البصمة
classification = classify_fingerprint(
    image=img1,
    minutiae=points1,
    metadata={"finger_type": "thumb"}
)

# التحقق من توافق بصمتين
is_compatible, reason = check_fingerprints_compatible(
    classification_ref,
    classification_qry
)

# إذا لم تكن متوافقة:
# → رفض المطابقة فوراً
# → السبب: "Different finger types: thumb vs index"
```

**الرفع التلقائي للكود**:
- فحص نوع الأصبع على كلا البصمتين ✓
- فحص منطقة البصمة على كلا البصمتين ✓
- فحص حجم البصمة (تحمل اختلاف 50%) ✓
- التحقق من الثقة (warning إذا كانت < 50%) ✓

#### 2. `database/models.py` (معدّل)
**التحديثات**:

في جدول `Fingerprint`:
```python
# التصنيف
fingerprint_type = Column(String(50))  # thumb, index, middle, etc.
fingerprint_region = Column(String(50))  # fingertip, palm_root, sub_index
fingerprint_classification = Column(JSON)  # كل بيانات التصنيف

# العلامات التشريحية (للمرحلة 2)
landmarks = Column(JSON)

# المراجعة اليدوية (للمرحلة 3)
is_manually_reviewed = Column(Integer, default=0)
manual_review_timestamp = Column(DateTime)
manual_review_by = Column(Integer, ForeignKey('users.id'))
manual_review_notes = Column(Text)
```

في جدول `Match`:
```python
# فحص التوافق
classification_compatible = Column(Integer, default=1)
classification_check_reason = Column(Text)
```

#### 3. `matching/compare_engine.py` (معدّل)
**التحديثات**:

الآن `FingerprintMatcher.compare_fingerprints()` يفعل:
1. **يصنف البصمة المرجعية** (إذا لم تُعطَ تصنيف مسبقاً)
2. **يصنف البصمة الاستعلام** (إذا لم تُعطَ تصنيف مسبقاً)
3. **يفحص التوافق بين التصنيفات**
4. إذا كانت **غير متوافقة** → **رفض فوري** (score=0, is_match=False)
5. إذا كانت **متوافقة** → تابع المطابقة التفصيلية

```python
# مثال الاستخدام
similarity, is_match, result = matcher.compare_fingerprints(
    minutiae_ref=points_ref,
    minutiae_qry=points_qry,
    image_shape=(384, 384),
    image_ref=img_ref,  # للتصنيف التلقائي
    image_qry=img_qry,
    metadata_ref={"finger_type": "thumb"},
    metadata_qry={"finger_type": "thumb"},
)

# النتائج تحتوي الآن على:
result["classification_check"] = {
    "performed": True,
    "ref_classification": {...},
    "qry_classification": {...},
    "compatible": True/False,
    "reason": "Different finger types: thumb vs index"
}
```

---

## المرحلة 2: العلامات التشريحية الثمانية ✅ (مكتملة)

### الملفات المُنشأة والمعدّلة:

#### 1. `features/minutiae_landmarks.py` (جديد)
**الوظيفة**: استخراج وتصنيف العلامات التشريحية

**الفئات**:
- `LandmarkAnalyzer`: محرك تحليل العلامات

**الميزات**:
```python
# استخراج العلامات
enhanced = extract_landmarks(
    minutiae=points,
    image=img,
    ridge_image=ridge_img
)

# كل نقطة توجد الآن:
# {
#     "x": 100, "y": 150,
#     "type": "bifurcation",
#     "landmark_type": "bifurcation",  # ← جديد
#     "landmark_info": {...},  # معلومات العلامة
#     "landmark_features": {...},  # خصائص محلية
# }

# إحصائيات
stats = landmark_statistics(enhanced)
# {
#     "total_landmarks": 45,
#     "landmark_counts": {"termination": 12, "bifurcation": 23, ...},
#     "high_importance_count": 35,
#     ...
# }
```

#### 2. `features/minutiae_taxonomy.py` (معدّل)
**التحديثات**:

أضيفت قاعدة البيانات الشاملة `ANATOMICAL_LANDMARKS`:

```python
ANATOMICAL_LANDMARKS = {
    "termination": {
        "name_en": "Termination / Ridge Ending",
        "name_ar": "نهاية الخط",
        "symbol": "◇",
        "icon": "ending",
        "description_en": "Point where a ridge ends abruptly",
        "description_ar": "نقطة تنتهي فيها خطوط البصمة",
        "connectivity_number": 1,
        "forensic_importance": "High",
        ...
    },
    "bifurcation": {
        "name_en": "Bifurcation / Ridge Split",
        "name_ar": "التفرع / التشعب",
        "symbol": "⊢",
        "icon": "bifurcation",
        "connectivity_number": 3,
        "forensic_importance": "High",
        ...
    },
    # وهكذا للـ 8 علامات كاملة
}
```

**دوال جديدة**:
```python
get_landmark_by_name(name)  # الحصول على تفاصيل العلامة
get_all_landmarks()  # الحصول على كل العلامات
landmark_names(lang="en")  # أسماء العلامات بلغة معينة
is_high_importance_landmark(name)  # فحص الأهمية الجنائية
```

---

## الـ 8 العلامات التشريحية - معلومات شاملة

| # | العربية | الإنجليزية | الرمز | الأهمية | CN |
|---|---------|-----------|------|--------|-----|
| 1 | **نهاية الخط** | Termination | ◇ | عالية | 1 |
| 2 | **التفرع** | Bifurcation | ⊢ | عالية | 3 |
| 3 | **الجزيرة** | Island | ⊗ | متوسطة | 2 |
| 4 | **الشرطة** | Ridge | ─ | متوسطة | 0 |
| 5 | **العين** | Loop/Eye | ◯ | عالية | 0 |
| 6 | **الجسر** | Bridge | ⌢ | متوسطة | 4 |
| 7 | **البحيرة** | Lake | ◈ | متوسطة | 2 |
| 8 | **النقطة** | Dot | • | منخفضة | 0 |

---

## البنية الحالية للمشروع

```
features/
├── fingerprint_classifier.py    ← مصنف النوع (جديد)
├── minutiae_landmarks.py        ← محرر العلامات (جديد)
├── minutiae_taxonomy.py         ← قاعدة بيانات العلامات (معدّل)
├── extractor.py
├── ...

matching/
├── compare_engine.py            ← محرك المطابقة مع التصنيف (معدّل)
├── matcher.py
├── ...

database/
├── models.py                    ← النماذج مع حقول جديدة (معدّل)
├── crud.py
├── ...
```

---

## الخطوات التالية

### المرحلة 3: واجهة التحكم اليدوي (المرحلة القادمة)

**الملفات المطلوبة**:
- `templates/manual_editor.html` - واجهة المحرر
- `static/js/manual_editor.js` - منطق التفاعل
- `routers/editor.py` - المسارات الخادمية
- `services/manual_annotation_service.py` - خدمة حفظ

**الميزات المطلوبة**:
1. عرض البصمة مع كل النقاط
2. حذف النقاط الخاطئة (نقر + X)
3. إضافة نقاط ناقصة (اختيار نوع + نقر على الموقع)
4. تحديث النتيجة تلقائياً
5. حفظ التعديلات

---

## كيفية الاستخدام - أمثلة عملية

### مثال 1: المطابقة الأساسية مع التصنيف التلقائي

```python
from matching.compare_engine import FingerprintMatcher
from features.fingerprint_classifier import classify_fingerprint

# تحضير البيانات
matcher = FingerprintMatcher(threshold=40)

# المطابقة (مع التصنيف التلقائي)
similarity, is_match, result = matcher.compare_fingerprints(
    minutiae_ref=original_points,
    minutiae_qry=query_points,
    image_shape=(384, 384),
    image_ref=original_image,
    image_qry=query_image,
)

print(f"Classification Check: {result['classification_check']}")
if result['classification_check']['compatible']:
    print(f"Match Score: {similarity}")
else:
    print(f"Rejected: {result['rejection_reason']}")
```

### مثال 2: المطابقة مع تصنيف محدد مسبقاً

```python
from features.fingerprint_classifier import classify_fingerprint

# التصنيف المسبق
class_ref = classify_fingerprint(
    metadata={"finger_type": "thumb", "region": "fingertip"}
)
class_qry = classify_fingerprint(
    metadata={"finger_type": "thumb", "region": "fingertip"}
)

# المطابقة بدون إعادة تصنيف
similarity, is_match, result = matcher.compare_fingerprints(
    minutiae_ref=original_points,
    minutiae_qry=query_points,
    image_shape=(384, 384),
    classification_ref=class_ref,
    classification_qry=class_qry,
)
```

### مثال 3: استخراج العلامات التشريحية

```python
from features.minutiae_landmarks import extract_landmarks, landmark_statistics

# استخراج العلامات
enhanced = extract_landmarks(
    minutiae=points,
    image=img,
    ridge_image=ridge_img
)

# الإحصائيات
stats = landmark_statistics(enhanced)
print(f"Landmark Distribution: {stats['landmark_counts']}")
print(f"High Importance: {stats['high_importance_count']}")
```

---

## ملاحظات مهمة

1. **التطبيق على كلا البصمتين**: ✅ مُطبّق
   - النظام يصنف البصمة المرجعية والاستعلام بشكل مستقل
   - يفحص التوافق قبل المطابقة

2. **الفشل المبكر** (Early Rejection):
   - إذا كانت البصمات من أنواع مختلفة → رفض فوري
   - توفير الموارد وتحسين الأداء

3. **قابلية التوسع**:
   - يمكن إضافة معلومات تصنيف من المستخدم
   - يمكن تدريب نموذج ML للتصنيف التلقائي لاحقاً

4. **التوثيق**:
   - كل دالة موثقة مع أمثلة
   - أسماء عربية وإنجليزية للعلامات

---

## الخطوة التالية

الكود جاهز. التكامل مع `/analyze`:

→ [PHASE_6_INTEGRATION.md](PHASE_6_INTEGRATION.md) (Sprint 6.2 تصنيف · 6.3 landmarks)
