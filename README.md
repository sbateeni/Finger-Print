# نظام تحليل البصمات الجنائي

نظام متقدم لتحليل ومطابقة البصمات باستخدام تقنيات الذكاء الاصطناعي.

## المتطلبات الأساسية

- Python 3.8 أو أحدث
- pip (مدير حزم Python)

## التثبيت

1. قم بنسخ المستودع:
```bash
git clone https://github.com/yourusername/Finger-Print.git
cd Finger-Print
```

2. قم بإنشاء بيئة افتراضية وتفعيلها:
```bash
python -m venv venv
source venv/bin/activate  # على Linux/Mac
venv\Scripts\activate     # على Windows
```

3. قم بتثبيت المتطلبات:
```bash
pip install -r requirements.txt
```

## كيفية الاستخدام

### النسخة المحلية

لتشغيل النسخة المحلية (مع دعم `use_container_width`):
```bash
streamlit run streamlit_app_local.py
```

### النسخة الإلكترونية

لتشغيل النسخة الإلكترونية (بدون `use_container_width`):
```bash
streamlit run streamlit_app_online.py
```

## الميزات

- تحليل البصمات باستخدام تقنيات الذكاء الاصطناعي
- استخراج النقاط المميزة من البصمات
- مطابقة البصمات الجزئية مع البصمات الكاملة
- عرض نتائج مفصلة للمطابقة
- حفظ نتائج المطابقة في ملفات نصية

## هيكل المشروع

```
Finger-Print/
├── streamlit_app_local.py    # النسخة المحلية
├── streamlit_app_online.py   # النسخة الإلكترونية
├── requirements.txt          # المتطلبات
├── preprocessing/            # معالجة الصور
├── features/                 # استخراج الميزات
├── matching/                 # مطابقة البصمات
└── results/                  # نتائج المطابقة
```

## المساهمة

نرحب بمساهماتكم! يرجى اتباع الخطوات التالية:

1. قم بعمل Fork للمشروع
2. قم بإنشاء فرع جديد (`git checkout -b feature/AmazingFeature`)
3. قم بعمل Commit للتغييرات (`git commit -m 'Add some AmazingFeature'`)
4. قم بعمل Push للفرع (`git push origin feature/AmazingFeature`)
5. قم بفتح Pull Request

## الترخيص

هذا المشروع مرخص تحت رخصة MIT - انظر ملف [LICENSE](LICENSE) للمزيد من التفاصيل. 