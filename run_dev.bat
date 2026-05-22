@echo off
cd /d "%~dp0"
set LIVE_RELOAD=1
python server\dev_server.py %*
