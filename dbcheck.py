import sqlite3
import os
import pandas as pd

DB_PATH = 'data/app.db'

if not os.path.exists(DB_PATH):
    print(f"Database {DB_PATH} does not exist.")
    exit(1)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get number of tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]
print(f"Number of tables: {len(tables)}")
print(f"Tables: {', '.join(tables)}")

# For each table, get columns and sample data
for table in tables:
    print(f"\n--- Table: {table} ---")
    
    # Get columns
    cursor.execute(f"PRAGMA table_info({table});")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"Columns: {', '.join(columns)}")
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table};")
    row_count = cursor.fetchone()[0]
    print(f"Row count: {row_count}")
    
    # Sample data (first 5 rows)
    if row_count > 0:
        df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5;", conn)
        print("Sample data:")
        print(df.to_string(index=False))
    else:
        print("No data in table.")

conn.close()
