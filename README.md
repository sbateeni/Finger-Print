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
cp config/environment.example .env
# أو: cp .env.example .env
# تعديل المتغيرات في ملف .env
```

5. تهيئة قاعدة البيانات (اختياري):
```bash
python scripts/init_db.py
```

## التشغيل

Windows:
```powershell
.\run_dev.ps1
```

Linux / Kali:
```bash
./run_dev.sh
```

أو:
```bash
python run.py
```

الواجهة: `http://127.0.0.1:8000` (Windows) أو `http://0.0.0.0:8000` (Linux)

انظر [docs/PROJECT_LAYOUT.md](docs/PROJECT_LAYOUT.md) و [docs/CROSS_PLATFORM.md](docs/CROSS_PLATFORM.md).

## النشر على Render.com

1. إنشاء حساب على Render.com
2. ربط المستودع مع Render
3. إنشاء خدمة Web Service جديدة
4. تكوين المتغيرات البيئية
5. نشر التطبيق

## هيكل المشروع

```
Finger-Print/
├── run.py, run_dev.*       # واجهات تشغيل → scripts/run/
├── requirements*.txt       # → requirements/
├── config/environment.example
├── deploy/                 # Dockerfile, Procfile
├── docs/plans/             # خطط التطوير
├── scripts/run/            # run_app.py, run_telegram.py
├── server/                 # FastAPI
├── bot/                    # Telegram
├── services/               # التحليل والتقارير
├── matching/               # المطابقة
├── templates/              # HTML
└── static/                 # CSS/JS
```

التفاصيل: [docs/PROJECT_LAYOUT.md](docs/PROJECT_LAYOUT.md)

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