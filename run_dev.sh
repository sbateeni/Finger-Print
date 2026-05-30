#!/usr/bin/env bash
# Wrapper — التشغيل الفعلي في scripts/run/run_dev.sh
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/scripts/run/run_dev.sh" "$@"
