import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- IMPORT MODELS ---
from src.models.baseline import BaselineModel
from src.models.bilstm import DeepModel
from src.models.transformer import TransformerModel  # <--- NEW IMPORT

# Configuration
DATA_PATH = "data/clean_data.csv"
REPORT_DIR = "reports"
BASELINE_DIR = "models/baseline"
DEEP_DIR = "models/deep"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(BASELINE_DIR, exist_ok=True)
os.makedirs(DEEP_DIR, exist_ok=True)

def load_data():
    print("  [Train] Loading Data...")
    df = pd.read_csv(DATA_PATH).dropna(subset=['text', 'label'])
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()
    
    # Stratified Split
    X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.30, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_metrics(y_true, y_pred, model_name, results_dict):
    acc = accuracy_score(y_true, y_pred)
    print(f"  ✅ {model_name.upper()} Accuracy: {acc:.4f}")
    
    report = classification_report(y_true, y_pred, output_dict=True)
    results_dict[model_name] = {"accuracy": acc, "report": report}
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.title(f"Confusion Matrix - {model_name.upper()}")
    plt.savefig(os.path.join(REPORT_DIR, f"confusion_matrix_{model_name}.png"))
    plt.close()

def run_training():
    X_train_txt, y_train, X_val_txt, y_val, X_test_txt, y_test = load_data()
    results = {}

    # --- STAGE 1: BASELINE (SKIPPED) ---
    print("\n⏩ SKIPPING STAGE 1 (Already Trained)")
    # for algo in ['lr', 'nb', 'svm']:
    #     model = BaselineModel(algorithm=algo)
    #     model.train(X_train_txt, y_train)
    #     # ... saving code ...

    # --- STAGE 2: DEEP LEARNING (SKIPPED) ---
    print("\n⏩ SKIPPING STAGE 2 (Already Trained)")
    # for arch in ['bilstm', 'cnn']:
    #     # ... training code ...

    # --- STAGE 3: TRANSFORMER (DistilBERT) ---
    print("\n🚀 STAGE 3: TRANSFORMER (DistilBERT)")
    print("  [Note] First run will download model (~260MB). Please wait...")
    
    # Initialize and Train
    transformer = TransformerModel()
    transformer.train(X_train_txt, y_train)
    transformer.save()
    
    # Evaluate
    print("  [Evaluate] DistilBERT...")
    preds = transformer.predict(X_test_txt)
    save_metrics(y_test, preds, "distilbert", results)

    # Final Report
    with open(os.path.join(REPORT_DIR, "training_report_stage3.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✨ Stage 3 Complete! Check {REPORT_DIR}/")

if __name__ == "__main__":
    run_training()