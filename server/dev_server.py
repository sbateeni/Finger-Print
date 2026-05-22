"""
تشغيل uvicorn للتطوير مع إيقاف موثوق عند Ctrl+C على Windows.

مع --reload يُنشئ uvicorn مراقبًا وعملية فرعية؛ إيقاف الشجرة يتطلب taskkill /T
أو إرسال إشارة لمجموعة عمليات على Unix.
"""

from __future__ import annotations

import argparse
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


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


def main() -> None:
    parser = argparse.ArgumentParser(description="تشغيل خادم التطوير مع إيقاف شجرة العمليات.")
    parser.add_argument("--no-reload", action="store_true", help="بدون إعادة تحميل (Ctrl+C أبسط)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="8000")
    args = parser.parse_args()

    cmd: list[str] = [
        sys.executable,
        "-m",
        "uvicorn",
        "server.server:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if not args.no_reload:
        cmd.append("--reload")
        cmd.extend(["--reload-dir", str(ROOT)])

    popen_kw: dict = {"cwd": ROOT}
    if platform.system() != "Windows":
        popen_kw["start_new_session"] = True

    p = subprocess.Popen(cmd, **popen_kw)

    def handle_sig(*_a: object) -> None:
        if p.poll() is not None:
            sys.exit(130)
        _kill_process_tree(p.pid)
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except ProcessLookupError:
                pass
        sys.exit(130)

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        rc = p.wait()
    except KeyboardInterrupt:
        handle_sig()
    sys.exit(rc if rc is not None else 0)


if __name__ == "__main__":
    main()
