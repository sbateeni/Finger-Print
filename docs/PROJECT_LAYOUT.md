# هيكل المشروع

ملفات الجذر (للتوافق مع GitHub / Render / pip):

| الملف | الدور |
|--------|--------|
| `README.md` | وصف المشروع |
| `LICENSE` | الترخيص |
| `.gitignore` | استثناءات Git |
| `.env` | أسرار التشغيل (محلي — لا يُرفع) |
| `.env.example` | إرشاد — انسخ من `config/environment.example` |
| `run.py` / `run_dev.*` | واجهات تشغيل سريعة → `scripts/run/` |
| `requirements*.txt` | إحالات → `requirements/` |
| `Dockerfile` / `Procfile` | نشر — النسخة المرجعية في `deploy/` |

## المجلدات

```
Finger-Print/
├── scripts/run/          # تشغيل: run_app.py, run_telegram.py, run_dev.*
├── requirements/         # main.txt, dev.txt, ml.txt
├── deploy/               # Dockerfile, Procfile (مرجع)
├── config/
│   ├── environment.example   # نموذج .env
│   └── config.py             # إعدادات التطبيق
├── docs/
│   ├── plans/            # DEVELOPMENT_PLAN, CORE_DEVELOPMENT_PLAN
│   ├── notes/            # ملاحظات مؤقتة / scratchpad
│   └── CROSS_PLATFORM.md
├── server/               # FastAPI
├── bot/                  # Telegram
├── services/             # منطق التحليل
└── ...
```

## أوامر التشغيل (كما السابق)

```powershell
.\run_dev.ps1          # Windows
```

```bash
./run_dev.sh           # Linux / Kali
```

```bash
python run.py
pip install -r requirements.txt
cp config/environment.example .env
```

```bash
docker build -f deploy/Dockerfile .
```
