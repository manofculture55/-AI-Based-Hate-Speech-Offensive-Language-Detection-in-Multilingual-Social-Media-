import pandas as pd
import os
import sqlite3
import warnings

# --- IMPORT MODULAR FUNCTIONS ---
# This satisfies "Clean Modular Code" requirement [Section 8]
from src.data.preprocess import clean_text
from src.data.langid import detect_language_strict

warnings.filterwarnings("ignore")

# Define Paths
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "app.db")
FINAL_CSV = os.path.join(DATA_DIR, "clean_data.csv")

def save_to_db(df):
    """Saves processed data to SQLite 'annotations' table [Section 4]."""
    conn = sqlite3.connect(DB_PATH)
    df.to_sql('annotations', conn, if_exists='replace', index=False) 
    conn.close()
    print("✅ Data saved to SQLite 'annotations' table.")

# --- DATASET PROCESSORS ---

def process_hasoc_english():
    path = os.path.join(DATA_DIR, "hasoc2019_en_test-2919.tsv")
    if not os.path.exists(path): return None
    
    df = pd.read_csv(path, sep='\t', quoting=3)
    
    def get_label(row):
        t1, t2 = str(row.get('task_1','')), str(row.get('task_2',''))
        if t1 == 'NOT': return 0
        if t2 == 'HATE': return 2
        return 1 

    df['text'] = df['text'].apply(clean_text)
    df['label'] = df.apply(get_label, axis=1)
    df['lang'] = df['text'].apply(lambda x: detect_language_strict(x, "english"))
    return df[['text', 'label', 'lang']]

def process_hasoc_hindi(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path): return None
    
    try:
        df = pd.read_csv(path, sep='\t', quoting=3, on_bad_lines='skip')
    except:
        df = pd.read_csv(path, sep='\t', on_bad_lines='skip')

    def get_label(row):
        t1, t2 = str(row.get('task_1','')), str(row.get('task_2',''))
        if t1 == 'NOT': return 0
        if t2 == 'HATE': return 2
        return 1

    df['text'] = df['text'].apply(clean_text)
    df['label'] = df.apply(get_label, axis=1)
    df['lang'] = df['text'].apply(lambda x: detect_language_strict(x, "hindi_mixed"))
    return df[['text', 'label', 'lang']]

def process_mdpi():
    path = os.path.join(DATA_DIR, "MDPI2025_Dataset.csv")
    if not os.path.exists(path): return None
    
    df = pd.read_csv(path, encoding='utf-8')
    label_map = {0: 0, 1: 1, 2: 2, 3: 1}
    
    df['text'] = df['comment'].apply(clean_text)
    df['label'] = df['label'].map(label_map).fillna(0).astype(int)
    df['lang'] = df['text'].apply(lambda x: detect_language_strict(x, "english"))
    return df[['text', 'label', 'lang']]

def process_indo_hate():
    path = os.path.join(DATA_DIR, "prime_Indo_HateSpeech_Dataset.xlsx")
    if not os.path.exists(path): return None
    
    df = pd.read_excel(path)
    
    def map_indo_label(l):
        l = str(l).strip().replace("'", "")
        if 'HS0' in l: return 0
        if 'HS1' in l: return 1
        if 'HSN' in l or 'HS2' in l: return 2
        return 0

    df['text'] = df['Comment'].apply(clean_text)
    df['label'] = df['Label'].apply(map_indo_label)
    df['lang'] = df['text'].apply(lambda x: detect_language_strict(x, "indo_mixed"))
    return df[['text', 'label', 'lang']]

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print("🚀 KRIXION Data Normalization...")
    
    # Process datasets
    dfs = []
    d1 = process_hasoc_english()
    if d1 is not None: dfs.append(d1)
    d2 = process_hasoc_hindi("hasoc2019_hi_test_gold_2919.tsv")
    if d2 is not None: dfs.append(d2)
    d3 = process_hasoc_hindi("hindi_dataset.csv")
    if d3 is not None: dfs.append(d3)
    d4 = process_mdpi()
    if d4 is not None: dfs.append(d4)
    d5 = process_indo_hate()
    if d5 is not None: dfs.append(d5)
    
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.drop_duplicates(subset=['text'], inplace=True)
        final_df = final_df[final_df['text'].str.len() > 2]
        
        # 🔥 CRITICAL: Fix column names to match db.py schema EXACTLY
        final_df = final_df.rename(columns={'label': 'truelabel'})  # label → truelabel
        final_df['source'] = 'krixion_datasets'  # Add required source column
        
        # Save CSV for training
        final_df[['text', 'lang', 'truelabel']].to_csv("data/clean_data.csv", index=False)
        
        # Save to SQLite annotations table (Section 4)
        final_df.to_sql('annotations', sqlite3.connect(DB_PATH), if_exists='replace', index=False)
        print("✅ Data saved to SQLite 'annotations' table.")

        
        print(f"✅ SAVED {len(final_df)} rows to annotations table")
        print(final_df[['lang', 'truelabel']].value_counts())
    else:
        print("❌ No datasets found")
