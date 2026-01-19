import sqlite3
import os

# Define the path to the database file in the data folder
DB_PATH = os.path.join("data", "app.db")

def init_db():
    """
    Initializes the SQLite database with the mandatory schema 
    defined in KRIXION Project Brief Section 4.
    """
    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Table: predictions (Stores user inputs and model results) [9]
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            lang TEXT CHECK(lang IN ('hi','en','hi-en')),
            predicted_label INTEGER CHECK(predicted_label IN (0,1,2)),
            score REAL NOT NULL,
            model_name TEXT NOT NULL,
            latency_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 2. Table: runs (Stores training metrics and model performance) [10]
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            macro_f1 REAL,
            accuracy REAL,
            precision REAL,
            recall REAL,
            latency_p95_ms REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 3. Table: annotations (For manual labeling/correction) [10]
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            lang TEXT,
            truelabel INTEGER,
            source TEXT
        )
    ''')

    conn.commit()
    conn.close()
    print(f"✅ Database initialized successfully at: {DB_PATH}")

if __name__ == "__main__":
    init_db()