"""
تشغيل بوت Telegram فقط (اختياري — للتشخيص).

التشغيل العادي: python run.py  (web + Telegram معًا)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from bot.telegram_bot import main

if __name__ == "__main__":
    main()
