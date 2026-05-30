"""
تشغيل موحّد: الواجهة web + بوت Telegram
(يُوجّه إلى scripts/run/run_app.py)
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "run" / "run_app.py"
    runpy.run_path(str(target), run_name="__main__")
