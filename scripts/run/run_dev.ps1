# Windows — تشغيل موحّد: web + Telegram (مثل run_dev.sh على Kali)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

if (-not $env:HOST) { $env:HOST = "127.0.0.1" }
if (-not $env:PORT) { $env:PORT = "8000" }
# 0 = stable (one Telegram poller, embedded bot). 1 = reload + separate bot process
if (-not $env:LIVE_RELOAD) { $env:LIVE_RELOAD = "0" }
if (-not $env:TELEGRAM_EMBEDDED) { $env:TELEGRAM_EMBEDDED = "1" }

if (-not $env:AUTO_SYNC_ENV -or $env:AUTO_SYNC_ENV -ne "0") {
  python "$Root\scripts\sync_env_from_example.py"
}

$logoSrc = Join-Path $Root "static\branding\palestinian-police-logo-source.png"
if (Test-Path $logoSrc) {
  python "$Root\scripts\prepare_police_logo.py" 2>$null
}

& "$Root\scripts\stop_telegram_bot.ps1"
Write-Host "=== Fingerprint workstation (Windows) ==="
Write-Host "Web:  http://$($env:HOST):$($env:PORT)"
Write-Host "Dev reload: `$env:LIVE_RELOAD='1'; .\run_dev.ps1"
Write-Host "Note: use one TELEGRAM_BOT_TOKEN per machine (not Kali + Windows at once)"
python "$PSScriptRoot\run_app.py" @args
