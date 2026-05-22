# تشغيل موحّد: web + Telegram
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:LIVE_RELOAD = "1"
python run_app.py @args
