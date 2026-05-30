#!/usr/bin/env bash
# Kali / Linux — تشغيل موحّد: سحب GitHub + الواجهة + Telegram
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export FP_PLATFORM=linux
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
# 0 = stable on Kali (Telegram + web without uvicorn killing the bot on file changes)
export LIVE_RELOAD="${LIVE_RELOAD:-0}"
export AUTO_GIT_UPDATE="${AUTO_GIT_UPDATE:-1}"

echo "=== Fingerprint workstation (Linux/Kali) ==="
uname -a 2>/dev/null || true

# Build/PDF system libs on Debian/Kali (only if pip fails on pycairo/cairo)
KALI_APT_PACKAGES="pkg-config libcairo2-dev libgirepository-2.0-dev build-essential libjpeg-dev zlib1g-dev"
if command -v apt-get >/dev/null 2>&1; then
  missing=""
  for pkg in python3-dev build-essential libjpeg-dev zlib1g-dev git; do
    dpkg -s "$pkg" >/dev/null 2>&1 || missing="$missing $pkg"
  done
  if [ -n "$missing" ]; then
    echo "If pip/git fails, run: sudo apt install -y$missing $KALI_APT_PACKAGES"
  fi
fi

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is required for auto-update. sudo apt install -y git"
  exit 1
fi

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

# Pull latest from GitHub before installing deps / starting bot
if [ -f scripts/git_pull_startup.py ]; then
  echo "--- GitHub ---"
  python scripts/git_pull_startup.py || echo "Git: update step failed (continuing)"
fi

pip install -q -r requirements.txt

if [ "${AUTO_SYNC_ENV:-1}" != "0" ] && [ -f scripts/sync_env_from_example.py ]; then
  python scripts/sync_env_from_example.py || true
fi

if [ ! -f .env ] && [ -f config/environment.example ]; then
  echo "Copy config/environment.example to .env and set TELEGRAM_BOT_TOKEN"
fi

if [ -f scripts/stop_telegram_bot.sh ]; then
  bash scripts/stop_telegram_bot.sh || true
fi

echo "--- Start ---"
echo "Web:  http://${HOST}:${PORT}"
echo "Telegram: deep analysis on this Linux host (do not run same token on Windows)"
echo "Dev reload: LIVE_RELOAD=1 ./run_dev.sh"
exec python scripts/run/run_app.py --host "$HOST" --port "$PORT" "$@"
