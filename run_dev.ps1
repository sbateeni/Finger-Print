# تشغيل موحّد: web + Telegram
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& "$PSScriptRoot\scripts\stop_telegram_bot.ps1"
$env:LIVE_RELOAD = "1"
python run_app.py @args
