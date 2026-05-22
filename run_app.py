"""
تشغيل موحّد: الواجهة web + بوت Telegram من أمر واحد.

  python run.py
  python run.py --host 0.0.0.0 --port 8000

Windows (PowerShell):
  .\\run_dev.ps1

Kali / Linux:
  ./run_dev.sh
"""

from __future__ import annotations

import argparse
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from utils.git_updater import run_startup_auto_update, start_periodic_auto_update


def _kill_process_tree(pid: int) -> None:
    if platform.system() == "Windows":
        run_kw: dict = {
            "cwd": ROOT,
            "capture_output": True,
            "stdin": subprocess.DEVNULL,
        }
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            run_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], **run_kw)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (AttributeError, ProcessLookupError, OSError):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def _popen(cmd: list[str]) -> subprocess.Popen:
    popen_kw: dict = {"cwd": ROOT}
    if platform.system() != "Windows":
        popen_kw["start_new_session"] = True
    return subprocess.Popen(cmd, **popen_kw)


def _telegram_enabled(explicit_no: bool) -> bool:
    if explicit_no:
        return False
    if (os.getenv("TELEGRAM_ENABLED") or "").strip().lower() in ("0", "false", "no", "off"):
        return False
    return bool((os.getenv("TELEGRAM_BOT_TOKEN") or "").strip())


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="تشغيل نظام البصمات (web + Telegram).")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", default=os.getenv("PORT", "8000"))
    parser.add_argument("--no-reload", action="store_true", help="بدون إعادة تحميل تلقائية للويب")
    parser.add_argument("--no-telegram", action="store_true", help="تشغيل الواجهة فقط")
    args = parser.parse_args()

    run_startup_auto_update()
    start_periodic_auto_update()

    children: list[subprocess.Popen] = []
    use_reload = not args.no_reload and os.getenv("LIVE_RELOAD", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )

    embedded = (os.getenv("TELEGRAM_EMBEDDED") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if _telegram_enabled(args.no_telegram):
        if embedded:
            print("Telegram bot: embedded in web server (طابور انتظار موحّد).")
        else:
            print("Telegram bot: separate process (TELEGRAM_EMBEDDED=0)…")
            children.append(_popen([sys.executable, "-m", "bot"]))
    elif not args.no_telegram:
        print("Telegram bot: skipped (set TELEGRAM_BOT_TOKEN in .env to enable).")

    if use_reload:
        web_cmd = [
            sys.executable,
            str(ROOT / "server" / "dev_server.py"),
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
    else:
        web_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "server.server:app",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]

    print(f"Web UI: http://{args.host}:{args.port}")
    children.append(_popen(web_cmd))

    shutting_down = False

    def shutdown(exit_code: int = 0) -> None:
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        for proc in children:
            if proc.poll() is None:
                _kill_process_tree(proc.pid)
        time.sleep(0.3)
        for proc in children:
            if proc.poll() is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
        sys.exit(exit_code)

    def handle_sig(*_a: object) -> None:
        shutdown(130)

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        while True:
            for proc in children:
                rc = proc.poll()
                if rc is not None:
                    print(f"Process exited with code {rc}: {' '.join(proc.args)}")
                    shutdown(rc if rc is not None else 1)
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown(130)


if __name__ == "__main__":
    main()
