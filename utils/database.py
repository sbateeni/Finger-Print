import json
import sqlite3
from datetime import datetime
from pathlib import Path

from config import OUTPUT_DIR

DB_PATH = (Path(OUTPUT_DIR) / "fingerprint_sys.db").resolve()

def init_db():
    """تهيئة قاعدة البيانات وجداول النتائج والتدقيق."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # جدول السجلات الجنائية (Audit Log)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            case_reference TEXT,
            operator_name TEXT,
            sha256_original TEXT,
            sha256_partial TEXT,
            match_score REAL,
            status TEXT,
            full_data TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def save_audit_record(record: dict):
    """حفظ سجل تدقيق في قاعدة البيانات."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_logs (
                timestamp, case_reference, operator_name, 
                sha256_original, sha256_partial, 
                match_score, status, full_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.get("timestamp", datetime.now().isoformat()),
            record.get("case_reference", ""),
            record.get("operator_name", ""),
            record.get("sha256_original", ""),
            record.get("sha256_partial", ""),
            record.get("match_score", 0.0),
            record.get("status", ""),
            json.dumps(record, ensure_ascii=False)
        ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving to DB: {e}")

# تهيئة قاعدة البيانات عند استيراد الملف
init_db()
