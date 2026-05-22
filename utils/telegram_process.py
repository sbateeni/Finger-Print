"""
Single Telegram poller guard — prevents 409 Conflict from duplicate getUpdates.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PID_FILE = ROOT / ".telegram_bot.pid"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if platform.system() == "Windows":
        run_kw: dict = {"capture_output": True, "text": True, "stdin": subprocess.DEVNULL}
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            run_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        proc = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], **run_kw)
        out = proc.stdout or ""
        return str(pid) in out and "python" in out.lower()
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _kill_pid(pid: int) -> None:
    if platform.system() == "Windows":
        run_kw: dict = {"capture_output": True, "stdin": subprocess.DEVNULL}
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            run_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], **run_kw)
    else:
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass


def stop_stale_telegram_bot() -> None:
    """Stop previous bot instance recorded in PID file (if still running)."""
    if not PID_FILE.is_file():
        return
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        PID_FILE.unlink(missing_ok=True)
        return
    if pid != os.getpid() and _pid_alive(pid):
        logger.warning("Stopping previous Telegram bot process (PID %s)", pid)
        _kill_pid(pid)
    PID_FILE.unlink(missing_ok=True)


def claim_telegram_bot_pid(pid: int | None = None) -> None:
    stop_stale_telegram_bot()
    PID_FILE.write_text(str(pid or os.getpid()), encoding="utf-8")


def release_telegram_bot_pid() -> None:
    try:
        if PID_FILE.is_file():
            stored = int(PID_FILE.read_text(encoding="utf-8").strip())
            if stored == os.getpid():
                PID_FILE.unlink(missing_ok=True)
    except (ValueError, OSError):
        PID_FILE.unlink(missing_ok=True)
