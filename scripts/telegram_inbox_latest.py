#!/usr/bin/env python3
"""Print latest Telegram inbox paths (reference + query) for scripts / IDE."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from services.telegram_inbox import INBOX_ROOT, get_inbox_status


def _default_chat_id() -> int | None:
    raw = (os.getenv("TELEGRAM_ALLOWED_CHAT_IDS") or "").strip()
    if not raw:
        return None
    first = raw.split(",")[0].strip()
    try:
        return int(first)
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Latest Telegram inbox image paths")
    parser.add_argument("--chat-id", type=int, help="Telegram chat id (default: first allowed id)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    chat_id = args.chat_id if args.chat_id is not None else _default_chat_id()
    if chat_id is None:
        print("Set TELEGRAM_ALLOWED_CHAT_IDS in .env or pass --chat-id", file=sys.stderr)
        return 1

    st = get_inbox_status(chat_id)
    if args.json:
        print(json.dumps(st, ensure_ascii=False, indent=2))
        return 0

    print(f"inbox root: {INBOX_ROOT}")
    print(f"chat_id: {chat_id}")
    print(f"reference: {st['reference_path'] or '(missing)'}")
    print(f"query:     {st['query_path'] or '(missing)'}")
    print(f"manifest:  {st['manifest_path']}")
    if st["reference_path"] and st["query_path"]:
        print()
        print(
            f'python scripts/test_pair_local.py "{st["reference_path"]}" "{st["query_path"]}"'
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
