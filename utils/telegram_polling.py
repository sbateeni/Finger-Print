"""
Ensure a single Telegram long-polling instance (avoids 409 Conflict).
"""

from __future__ import annotations

import logging
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_LOCK_NAME = ".telegram_polling.lock"


def _lock_path() -> Path:
    from config import OUTPUT_DIR

    p = Path(OUTPUT_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p / _LOCK_NAME


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return str(pid) in (out.stdout or "")
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_polling_lock() -> bool:
    """Return True if this process may start polling."""
    lock = _lock_path()
    my_pid = os.getpid()
    if lock.is_file():
        try:
            other = int(lock.read_text(encoding="utf-8").strip())
        except ValueError:
            other = 0
        if other and other != my_pid and _pid_alive(other):
            logger.warning(
                "Telegram polling lock held by PID %s — skip second poller (409 prevention)",
                other,
            )
            return False
    lock.write_text(str(my_pid), encoding="utf-8")
    return True


def release_polling_lock() -> None:
    lock = _lock_path()
    if not lock.is_file():
        return
    try:
        if int(lock.read_text(encoding="utf-8").strip()) == os.getpid():
            lock.unlink(missing_ok=True)
    except (ValueError, OSError):
        pass


async def prepare_bot_session(bot) -> None:
    """Drop webhook / pending updates so polling can attach cleanly."""
    try:
        await bot.delete_webhook(drop_pending_updates=True)
    except Exception as e:
        logger.warning("delete_webhook: %s", e)


def kill_stale_local_bot_processes(project_root: Path | None = None) -> int:
    """
    Best-effort: stop other python processes in this repo running bot/uvicorn.
    Returns number of processes signaled.
    """
    root = (project_root or Path(__file__).resolve().parent.parent).resolve()
    root_s = str(root).lower()
    killed = 0
    my_pid = os.getpid()

    if platform.system() == "Windows":
        try:
            rows = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                    "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            import json

            raw = (rows.stdout or "").strip()
            if not raw:
                return 0
            data = json.loads(raw)
            if isinstance(data, dict):
                data = [data]
            for row in data:
                pid = int(row.get("ProcessId") or 0)
                cmd = (row.get("CommandLine") or "").lower()
                if pid == my_pid or pid <= 0:
                    continue
                if root_s not in cmd:
                    continue
                if not any(
                    x in cmd
                    for x in ("-m bot", "uvicorn", "run_app", "dev_server", "telegram")
                ):
                    continue
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                killed += 1
        except Exception as e:
            logger.warning("kill_stale_local_bot_processes: %s", e)
        return killed

    # Linux/Kali: pgrep -f is more reliable than truncated ps args
    try:
        proc = subprocess.run(
            ["pgrep", "-af", "python"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        for line in (proc.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            cmd = parts[1].lower()
            if pid == my_pid or root_s not in cmd:
                continue
            if not any(x in cmd for x in ("-m bot", "uvicorn", "run_app", "dev_server")):
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                killed += 1
            except OSError:
                pass
    except FileNotFoundError:
        try:
            out = subprocess.run(
                ["ps", "-eo", "pid,args"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            for line in (out.stdout or "").splitlines()[1:]:
                parts = line.strip().split(None, 1)
                if len(parts) < 2:
                    continue
                pid = int(parts[0])
                cmd = parts[1].lower()
                if pid == my_pid or root_s not in cmd:
                    continue
                if not any(x in cmd for x in ("-m bot", "uvicorn", "run_app", "dev_server")):
                    continue
                try:
                    os.kill(pid, signal.SIGTERM)
                    killed += 1
                except OSError:
                    pass
        except Exception as e:
            logger.warning("kill_stale_local_bot_processes (ps): %s", e)
    except Exception as e:
        logger.warning("kill_stale_local_bot_processes (pgrep): %s", e)
    return killed
