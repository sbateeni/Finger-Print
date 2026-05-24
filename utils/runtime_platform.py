"""Runtime OS detection — Linux/Kali vs Windows defaults."""

from __future__ import annotations

import os
import platform
import sys


def system_name() -> str:
    return platform.system()


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def is_windows() -> bool:
    return sys.platform == "win32"


def default_bind_host() -> str:
    """Kali/Linux: listen on all interfaces; Windows dev: localhost."""
    env = (os.getenv("HOST") or "").strip()
    if env:
        return env
    return "0.0.0.0" if is_linux() else "127.0.0.1"


def telegram_stop_script_hint() -> str:
    if is_windows():
        return ".\\scripts\\stop_telegram_bot.ps1"
    return "bash scripts/stop_telegram_bot.sh"


def live_reload_enabled() -> bool:
    """Stable default off (Kali + Windows). Set LIVE_RELOAD=1 for dev auto-reload."""
    raw = (os.getenv("LIVE_RELOAD") or "").strip().lower()
    if not raw:
        return False
    return raw not in ("0", "false", "no", "off")


def dev_run_hint() -> str:
    if is_windows():
        return ".\\run_dev.ps1"
    return "./run_dev.sh"
