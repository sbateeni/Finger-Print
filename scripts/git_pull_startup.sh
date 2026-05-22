#!/usr/bin/env bash
# Fetch + fast-forward pull from GitHub (uses .env AUTO_GIT_UPDATE*)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .git ]; then
  echo "Git: not a repository — skip update"
  exit 0
fi

if [ ! -d venv ]; then
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
[ -f .env ] && set -a && . ./.env && set +a || true

exec python scripts/git_pull_startup.py
