# نظام مطابقة البصمات الجنائي

نظام متكامل لمطابقة البصمات باستخدام تقنيات الذكاء الاصطناعي وتحليل الصور. يستخدم النظام في مجال الطب الشرعي والتحقيقات الجنائية.

## المميزات

- تحليل النقاط الدقيقة في البصمات (Minutiae Points)
- تحليل أنماط الخطوط (Ridge Patterns)
- تقييم جودة الصور
- مطابقة جزئية للبصمات
- واجهة مراجعة بشرية
- تخزين وتتبع نتائج المطابقة

## المتطلبات

- Python 3.8+
- OpenCV
- NumPy
- SciPy
- scikit-image
- Flask
- SQLAlchemy

## التثبيت

1. استنساخ المستودع:
```bash
git clone https://github.com/yourusername/fingerprint-matching.git
cd fingerprint-matching
```

2. إنشاء بيئة افتراضية:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. تثبيت المتطلبات:
```bash
pip install -r requirements.txt
```

4. إعداد ملف البيئة:
```bash
cp .env.example .env
# تعديل المتغيرات في ملف .env
```

5. تهيئة قاعدة البيانات:
```bash
flask db init
flask db migrate
flask db upgrade
```

## التشغيل

1. تشغيل الخادم المحلي:
```bash
python app.py
```

2. فتح المتصفح على العنوان:
```
http://localhost:5000
```

## النشر على Render.com

1. إنشاء حساب على Render.com
2. ربط المستودع مع Render
3. إنشاء خدمة Web Service جديدة
4. تكوين المتغيرات البيئية
5. نشر التطبيق

## هيكل المشروع

```
fingerprint-matching/
├── app.py                 # التطبيق الرئيسي
├── requirements.txt       # متطلبات المشروع
├── Procfile              # ملف تكوين Render
├── .env                  # متغيرات البيئة
├── .gitignore           # ملفات Git المستثناة
├── database/            # نماذج قاعدة البيانات
├── features/            # استخراج الميزات
├── matching/            # خوارزميات المطابقة
├── preprocessing/       # معالجة الصور
├── static/             # الملفات الثابتة
├── templates/          # قوالب HTML
└── uploads/            # مجلد التحميلات
```

## المساهمة

1. Fork المشروع
2. إنشاء فرع جديد (`git checkout -b feature/amazing-feature`)
3. Commit التغييرات (`git commit -m 'Add amazing feature'`)
4. Push إلى الفرع (`git push origin feature/amazing-feature`)
5. فتح Pull Request

## الترخيص

هذا المشروع مرخص تحت رخصة MIT - انظر ملف [LICENSE](LICENSE) للمزيد من التفاصيل.

## الاتصال

- البريد الإلكتروني: your.email@example.com
- الموقع: https://your-website.com 