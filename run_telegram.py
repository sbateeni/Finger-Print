"""
تشغيل بوت Telegram (بجانب الواجهة web — لا يستبدلها).

  python run_telegram.py

يحتاج TELEGRAM_BOT_TOKEN في ملف .env
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from bot.telegram_bot import main
from utils.git_updater import run_startup_auto_update, start_periodic_auto_update

if __name__ == "__main__":
    run_startup_auto_update()
    start_periodic_auto_update()
    main()
