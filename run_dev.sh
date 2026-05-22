#!/usr/bin/env bash
# Kali / Linux — تشغيل موحّد: الواجهة web + بوت Telegram
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install -q -r requirements.txt

if [ ! -f .env ] && [ -f .env.example ]; then
  echo "Copy .env.example to .env and set TELEGRAM_BOT_TOKEN"
fi

export LIVE_RELOAD="${LIVE_RELOAD:-1}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

echo "Unified launcher: Web UI + Telegram (single process tree)"
exec python run_app.py --host "$HOST" --port "$PORT" "$@"
