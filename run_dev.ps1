# تشغيل موحّد: web + Telegram
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not $env:AUTO_SYNC_ENV -or $env:AUTO_SYNC_ENV -ne "0") {
  python "$PSScriptRoot\scripts\sync_env_from_example.py"
}
& "$PSScriptRoot\scripts\stop_telegram_bot.ps1"
$env:LIVE_RELOAD = "1"
python run_app.py @args
