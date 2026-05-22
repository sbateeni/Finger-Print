"""
Merge missing keys from .env.example into .env (never overwrite existing values).
Run automatically from run_dev.ps1 / run_dev.sh when AUTO_SYNC_ENV=1 (default).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / ".env.example"
TARGET = ROOT / ".env"

KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _parse_keys(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = KEY_RE.match(s)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def sync_env(*, dry_run: bool = False) -> list[str]:
    if not EXAMPLE.is_file():
        return []
    example_keys = _parse_keys(EXAMPLE)
    target_keys = _parse_keys(TARGET) if TARGET.is_file() else {}
    missing = [k for k in example_keys if k not in target_keys]
    if not missing:
        return []

    lines: list[str] = []
    if TARGET.is_file():
        lines = TARGET.read_text(encoding="utf-8").splitlines()
    else:
        lines = ["# Auto-created from .env.example — add your secrets below", ""]

    block = ["", "# --- added by scripts/sync_env_from_example.py ---"]
    for key in missing:
        block.append(f"{key}={example_keys[key]}")
    lines.extend(block)
    if not lines[-1].endswith("\n"):
        lines.append("")

    if not dry_run:
        TARGET.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return missing


def main() -> None:
    import sys

    dry = "--dry-run" in sys.argv
    added = sync_env(dry_run=dry)
    if added:
        msg = "Added keys: " + ", ".join(added)
        print(msg if not dry else f"[dry-run] {msg}")
    else:
        print(".env is up to date with .env.example")


if __name__ == "__main__":
    main()
