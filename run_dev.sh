#!/usr/bin/env bash
# Kali / Linux — تشغيل الواجهة + (اختياري) بوت Telegram في طرفين منفصلين
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install -q -r requirements.txt

export LIVE_RELOAD="${LIVE_RELOAD:-1}"
echo "Web UI: http://127.0.0.1:8000"
echo "Telegram (optional, other terminal): python run_telegram.py"
exec python server/dev_server.py --host 0.0.0.0 --port 8000
