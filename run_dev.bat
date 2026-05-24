@echo off
cd /d "%~dp0"
if not defined HOST set HOST=127.0.0.1
if not defined PORT set PORT=8000
if not defined LIVE_RELOAD set LIVE_RELOAD=0
if not defined TELEGRAM_EMBEDDED set TELEGRAM_EMBEDDED=1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop_telegram_bot.ps1"
python run_app.py %*
