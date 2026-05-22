# تشغيل التطوير: dev_server.py يقتل شجرة عمليات uvicorn (--reload) عند Ctrl+C على Windows
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:LIVE_RELOAD = "1"
python server\dev_server.py @args
