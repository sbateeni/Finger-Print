# التشغيل على Kali و Windows

## تشغيل سريع

| النظام | الأمر | الواجهة |
|--------|--------|---------|
| Kali / Linux | `./run_dev.sh` | `http://0.0.0.0:8000` |
| Windows (PowerShell) | `.\run_dev.ps1` | `http://127.0.0.1:8000` |
| Windows (cmd) | `run_dev.bat` | نفس العنوان |

الإعدادات الافتراضية: `LIVE_RELOAD=0` و `TELEGRAM_EMBEDDED=1` — عملية واحدة للويب والبوت (تجنّب خطأ Telegram 409).

## Telegram

- **توكن واحد = جهاز واحد في نفس الوقت.** لا تشغّل نفس `TELEGRAM_BOT_TOKEN` على Kali و Windows معاً.
- عند 409 Conflict على Windows:
  ```powershell
  .\scripts\stop_telegram_bot.ps1
  .\run_dev.ps1
  ```
- على Kali:
  ```bash
  bash scripts/stop_telegram_bot.sh
  ./run_dev.sh
  ```

## إعادة تحميل الكود أثناء التطوير

```bash
# Linux
LIVE_RELOAD=1 ./run_dev.sh
```

```powershell
# Windows
$env:LIVE_RELOAD='1'
.\run_dev.ps1
```

مع `LIVE_RELOAD=1` يُشغَّل البوت في عملية منفصلة عن uvicorn (لتجنّب إعادة تشغيل المراقب مرتين).

## SSL على Windows

إذا فشل اتصال Telegram بسبب الشهادات:

```
TELEGRAM_SSL_VERIFY=0
```

(للتطوير المحلي فقط)

## مكتبات Kali (اختياري)

إذا فشل `pip install` (cairo / PDF):

```bash
sudo apt install -y python3-dev build-essential libjpeg-dev zlib1g-dev \
  pkg-config libcairo2-dev libgirepository-2.0-dev git
```

## اختبار بدون Telegram

```bash
python run_app.py --no-telegram
```
