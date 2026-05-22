# Stop orphaned Telegram bot processes (fixes 409 Conflict)
$ErrorActionPreference = "SilentlyContinue"
Set-Location $PSScriptRoot\..

if (Test-Path ".telegram_bot.pid") {
    $pid = Get-Content ".telegram_bot.pid" -Raw
    if ($pid -match '^\d+$') {
        Write-Host "Stopping bot PID $pid"
        taskkill /PID $pid /T /F 2>$null
    }
    Remove-Item ".telegram_bot.pid" -Force
}

Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'run_telegram|-m bot' } |
    ForEach-Object {
        Write-Host "Stopping $($_.ProcessId): $($_.CommandLine)"
        taskkill /PID $_.ProcessId /T /F 2>$null
    }

Write-Host "Done. Start again with: .\run_dev.ps1"
