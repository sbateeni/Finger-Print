#!/usr/bin/env python3
"""Pull latest from GitHub before starting on Kali/Linux (called from run_dev.sh)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from utils.git_updater import run_startup_auto_update

if __name__ == "__main__":
    run_startup_auto_update(echo=True)
