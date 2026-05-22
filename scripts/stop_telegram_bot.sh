#!/usr/bin/env bash
# Stop duplicate Telegram / uvicorn workers on Linux/Kali (fixes 409 Conflict)
set -euo pipefail
cd "$(dirname "$0")/.."

LOCK_DIR="config/output"
mkdir -p "$LOCK_DIR"
rm -f "$LOCK_DIR/.telegram_polling.lock"

ROOT="$(pwd)"
killed=0

_stop_pid() {
  local pid="$1"
  [ -z "$pid" ] || [ "$pid" = "$$" ] && return
  kill -TERM "$pid" 2>/dev/null || true
  killed=$((killed + 1))
}

if command -v pgrep >/dev/null 2>&1; then
  while read -r line; do
    case "$line" in
      *run_app*|*dev_server*|*uvicorn*|*"-m bot"*|*telegram_bot*)
        pid="${line%% *}"
        _stop_pid "$pid"
        ;;
    esac
  done < <(pgrep -af python 2>/dev/null || true)
else
  while read -r pid cmd; do
    case "$cmd" in
      *Finger-Print*|*finger-print*)
        case "$cmd" in
          *run_app*|*dev_server*|*uvicorn*|*"-m bot"*) _stop_pid "$pid" ;;
        esac
        ;;
    esac
  done < <(ps -eo pid,args 2>/dev/null | tail -n +2 || true)
fi

sleep 0.5
echo "Stopped $killed stale Python worker(s). Lock file cleared."
