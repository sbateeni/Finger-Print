"""
تهيئة قاعدة البيانات وإنشاء الجداول
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from database import engine, Base
from database.models import User, Fingerprint, Match, Review

print("Creating database tables...")
Base.metadata.create_all(bind=engine)
print("Database tables created successfully!")
