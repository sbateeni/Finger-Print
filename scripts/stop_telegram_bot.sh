#!/usr/bin/env bash
# Stop duplicate Telegram / uvicorn workers (fixes 409 Conflict)
set -euo pipefail
cd "$(dirname "$0")/.."

LOCK_DIR="config/output"
mkdir -p "$LOCK_DIR"
rm -f "$LOCK_DIR/.telegram_polling.lock"

killed=0
if command -v pgrep >/dev/null 2>&1; then
  while read -r pid cmd; do
    [ -z "${pid:-}" ] && continue
    case "$cmd" in
      *Finger-Print*|*finger-print*)
        case "$cmd" in
          *"-m bot"*|*uvicorn*|*run_app*|*dev_server*)
            kill -TERM "$pid" 2>/dev/null || true
            killed=$((killed + 1))
            ;;
        esac
        ;;
    esac
  done < <(pgrep -af python 2>/dev/null || true)
fi

echo "Stopped $killed stale Python worker(s). Lock file cleared."
