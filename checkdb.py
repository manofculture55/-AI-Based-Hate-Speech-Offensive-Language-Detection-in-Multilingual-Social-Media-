#!/usr/bin/env python3
"""
KRIXION Hate Speech DB Inspector - Lists ALL tables, columns, row counts & top 5 rows
Save as: check_db.py  |  Run: python check_db.py
Path: data/app.db (KTPL standard)
"""

import sqlite3
import os
from pathlib import Path
import pandas as pd

# KRIXION Project Paths [file:2]
DATADIR = "data"
DBPATH = os.path.join(DATADIR, "app.db")

def inspect_database(db_path):
    """Complete SQLite inspection: tables → columns → counts → sample rows"""
    if not os.path.exists(db_path):
        print(f"❌ ERROR: {db_path} not found!")
        print("💡 Run: mkdir -p data && touch data/app.db")
        return
    
    print(f"🔍 INSPECTING: {db_path}")
    print("=" * 80)
    
    conn = sqlite3.connect(db_path)
    
    # 1. LIST ALL TABLES
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table';", 
        conn
    )['name'].tolist()
    
    print(f"📊 FOUND {len(tables)} TABLES:")
    print("-" * 40)
    
    for table in tables:
        # 2. GET COLUMN INFO
        cols = pd.read_sql_query(
            f"PRAGMA table_info({table});", 
            conn
        )
        col_names = cols['name'].tolist()
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        
        print(f"\n📋 TABLE: `{table}` ({row_count} rows)")
        print("   COLUMNS:", ", ".join(col_names))
        print("   SCHEMA:", cols[['name', 'type', 'notnull', 'pk']].to_string(index=False))
        
        # 3. TOP 5 ROWS (pretty format)
        if row_count > 0:
            try:
                sample = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5;", conn)
                print(f"   🧪 TOP 5 ROWS:\n{sample.to_string(index=False, max_colwidth=30)}")
            except:
                print("   ⚠️  Sample failed (binary/complex data)")
        else:
            print("   ℹ️  (empty table)")
        print()
    
    conn.close()
    print("✅ INSPECTION COMPLETE!")

if __name__ == "__main__":
    inspect_database(DBPATH)
