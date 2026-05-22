#!/usr/bin/env bash
# Kali / Linux — تشغيل موحّد: الواجهة web + بوت Telegram
set -euo pipefail
cd "$(dirname "$0")"

# Build headers (Pillow/OpenCV wheels usually prebuilt; needed if pip falls back to source)
if command -v apt-get >/dev/null 2>&1; then
  missing=""
  for pkg in python3-dev build-essential libjpeg-dev zlib1g-dev; do
    dpkg -s "$pkg" >/dev/null 2>&1 || missing="$missing $pkg"
  done
  if [ -n "$missing" ]; then
    echo "Optional (for pip source builds): sudo apt install -y$missing"
  fi
fi

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
