"""
Optional auto-update from GitHub on startup (and periodic checks).

Environment:
  AUTO_GIT_UPDATE=1              Enable fetch + pull when behind (default: 1)
  AUTO_GIT_UPDATE_INTERVAL_SEC   Periodic check interval; 0 = startup only (default: 0)
  AUTO_GIT_UPDATE_ALLOW_DIRTY=0  Skip pull if working tree has local changes (default)
  AUTO_GIT_REMOTE=origin         Remote name (default: origin)
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip().isdigit():
        return default
    return int(raw)


def is_auto_update_enabled() -> bool:
    return _env_bool("AUTO_GIT_UPDATE", True)


def update_interval_sec() -> int:
    return max(0, _env_int("AUTO_GIT_UPDATE_INTERVAL_SEC", 0))


def _run_git(args: list[str], cwd: Path = ROOT) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except FileNotFoundError:
        return 127, "", "git not found in PATH"
    except subprocess.TimeoutExpired:
        return -2, "", "git command timed out"
    except Exception as e:
        return -1, "", str(e)


def _is_git_repo() -> bool:
    code, out, _ = _run_git(["rev-parse", "--is-inside-work-tree"])
    return code == 0 and out == "true"


def _has_local_changes() -> bool:
    # -uno: tracked files only — local .env (gitignored) does not block pull on Kali
    code, out, _ = _run_git(["status", "--porcelain", "-uno"])
    return code == 0 and bool(out.strip())


def _upstream_ref(remote: str) -> Optional[str]:
    code, out, _ = _run_git(["rev-parse", "--abbrev-ref", "@{u}"])
    if code == 0 and out:
        return out
    code, branch, _ = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    if code != 0 or not branch:
        return None
    fallback = f"{remote}/{branch}"
    code2, _, _ = _run_git(["rev-parse", "--verify", fallback])
    return fallback if code2 == 0 else None


def _commits_behind(upstream: str) -> int:
    code, out, err = _run_git(["rev-list", "--count", f"HEAD..{upstream}"])
    if code != 0:
        logger.debug("commits_behind failed: %s", err)
        return 0
    try:
        return int(out)
    except ValueError:
        return 0


def check_and_pull_updates(force: bool = False) -> dict[str, Any]:
    """
    Fetch from remote; fast-forward pull if local branch is behind.
    Returns a result dict (never raises).
    """
    result: dict[str, Any] = {
        "ok": True,
        "updated": False,
        "skipped": True,
        "reason": "",
        "commits_pulled": 0,
        "behind_before": 0,
    }

    if not force and not is_auto_update_enabled():
        result["reason"] = "AUTO_GIT_UPDATE disabled"
        return result

    if not _is_git_repo():
        result["reason"] = "not a git repository"
        return result

    remote = (os.getenv("AUTO_GIT_REMOTE") or "origin").strip() or "origin"
    allow_dirty = _env_bool("AUTO_GIT_UPDATE_ALLOW_DIRTY", False)

    if _has_local_changes() and not allow_dirty:
        result["reason"] = "local changes present — pull skipped (commit/stash or set AUTO_GIT_UPDATE_ALLOW_DIRTY=1)"
        logger.warning("Auto-update skipped: uncommitted local changes")
        return result

    code, _, err = _run_git(["fetch", remote, "--prune"])
    if code != 0:
        result["ok"] = False
        result["reason"] = f"git fetch failed: {err or code}"
        logger.warning("Auto-update fetch failed: %s", err)
        return result

    upstream = _upstream_ref(remote)
    if not upstream:
        result["reason"] = "no upstream branch configured"
        return result

    behind = _commits_behind(upstream)
    result["behind_before"] = behind
    if behind == 0:
        result["reason"] = "already up to date"
        logger.info("Auto-update: already up to date (%s)", upstream)
        return result

    result["skipped"] = False
    code, out, err = _run_git(["merge", "--ff-only", upstream])
    if code != 0:
        result["ok"] = False
        result["reason"] = f"fast-forward pull failed: {err or out or code}"
        logger.warning("Auto-update pull failed: %s", err or out)
        return result

    result["updated"] = True
    result["commits_pulled"] = behind
    result["reason"] = f"pulled {behind} commit(s) from {upstream}"
    logger.info("Auto-update: %s", result["reason"])
    return result


def format_update_message(result: dict[str, Any]) -> str:
    if result.get("skipped") and result.get("reason") == "AUTO_GIT_UPDATE disabled":
        return "Git: auto-update disabled (AUTO_GIT_UPDATE=0)"
    if not result.get("ok"):
        return f"Git: failed — {result.get('reason', 'unknown')}"
    if result.get("updated"):
        return f"Git: pulled {result.get('commits_pulled', 0)} commit(s) — {result.get('reason', '')}"
    return f"Git: {result.get('reason', 'ok')}"


def run_startup_auto_update(*, echo: bool = False) -> dict[str, Any]:
    """Run once at process start. Set echo=True to print status (Kali launcher)."""
    if not is_auto_update_enabled():
        result = {"skipped": True, "reason": "AUTO_GIT_UPDATE disabled"}
        if echo:
            print(format_update_message(result))
        return result
    logger.info("Checking GitHub for updates…")
    result = check_and_pull_updates()
    msg = format_update_message(result)
    if echo:
        print(msg)
    elif result.get("updated"):
        print(msg)
    return result


def start_periodic_auto_update(stop_event: Optional[threading.Event] = None) -> Optional[threading.Thread]:
    """Background thread for repeated checks. Returns None if interval is 0."""
    interval = update_interval_sec()
    if interval <= 0 or not is_auto_update_enabled():
        return None

    def _loop() -> None:
        while stop_event is None or not stop_event.is_set():
            if stop_event is not None:
                if stop_event.wait(timeout=interval):
                    break
            else:
                time.sleep(interval)
            if stop_event is not None and stop_event.is_set():
                break
            try:
                check_and_pull_updates()
            except Exception:
                logger.exception("Periodic auto-update failed")

    thread = threading.Thread(target=_loop, name="git-auto-update", daemon=True)
    thread.start()
    logger.info("Periodic auto-update every %s seconds", interval)
    return thread
