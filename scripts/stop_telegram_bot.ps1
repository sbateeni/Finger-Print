# Stop duplicate Telegram / uvicorn workers for this project (fixes 409 Conflict)
$ErrorActionPreference = "SilentlyContinue"
Set-Location $PSScriptRoot\..

$lockDir = Join-Path $PWD "config\output"
if (-not (Test-Path $lockDir)) { New-Item -ItemType Directory -Path $lockDir -Force | Out-Null }
$lock = Join-Path $lockDir ".telegram_polling.lock"
if (Test-Path $lock) { Remove-Item $lock -Force }

$root = (Get-Location).Path.ToLower()
$killed = 0
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object {
    $cmd = ($_.CommandLine + "").ToLower()
    if ($cmd -notlike "*$($root.Replace('\','\\'))*" -and $cmd -notlike "*finger-print*") { return }
    if ($cmd -notmatch "bot|uvicorn|run_app|dev_server") { return }
    if ($_.ProcessId -eq $PID) { return }
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    $killed++
}
Write-Host "Stopped $killed stale Python worker(s). Lock file cleared."
