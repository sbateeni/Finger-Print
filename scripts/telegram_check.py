#!/usr/bin/env python3
"""Check Telegram bot token / webhook / polling conflict (409)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
if not token:
    print("TELEGRAM_BOT_TOKEN missing in .env")
    sys.exit(1)

import httpx

base = f"https://api.telegram.org/bot{token}"
with httpx.Client(timeout=30.0) as client:
    me = client.get(f"{base}/getMe").json()
    if not me.get("ok"):
        print("getMe failed:", me)
        sys.exit(1)
    bot = me["result"]
    print(f"OK bot: @{bot.get('username')} ({bot.get('first_name')})")

    wh = client.get(f"{base}/getWebhookInfo").json()
    url = (wh.get("result") or {}).get("url") or ""
    print(f"Webhook: {url or '(none — polling mode OK)'}")

    # One-shot getUpdates: 409 means another client is already polling
    r = client.post(f"{base}/getUpdates", json={"timeout": 0, "limit": 1})
    if r.status_code == 409:
        print("\n409 CONFLICT — جهاز/عملية أخرى تستخدم نفس التوكن الآن.")
        print("  • أوقف run_dev على Windows إن كان يعمل")
        print("  • على Kali: bash scripts/stop_telegram_bot.sh && ./run_dev.sh")
        sys.exit(2)
    data = r.json()
    if data.get("ok"):
        print("getUpdates: OK (no conflict — يمكن تشغيل البوت)")
    else:
        print("getUpdates:", data)
