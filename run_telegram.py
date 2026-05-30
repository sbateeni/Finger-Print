"""
تشغيل بوت Telegram فقط (اختياري — للتشخيص).
(يُوجّه إلى scripts/run/run_telegram.py)
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "run" / "run_telegram.py"
    runpy.run_path(str(target), run_name="__main__")
