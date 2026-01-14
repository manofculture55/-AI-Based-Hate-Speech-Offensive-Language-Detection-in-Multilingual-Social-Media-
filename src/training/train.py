import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- IMPORT MODELS ---
# We import both the Baseline (Day 2) and Deep (Day 3) classes
from src.models.baseline import BaselineModel
from src.models.bilstm import DeepModel

# Configuration
DATA_PATH = "data/clean_data.csv"
REPORT_DIR = "reports"
BASELINE_DIR = "models/baseline"
DEEP_DIR = "models/deep"

# Ensure directories exist
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(BASELINE_DIR, exist_ok=True)
os.makedirs(DEEP_DIR, exist_ok=True)

def load_data():
    """Load and Split Data (70/15/15) [Source 17]."""
    print("  [Train] Loading Data...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['text', 'label'])
    
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()
    
    # Stratified Split
    X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.30, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_metrics(y_true, y_pred, model_name, results_dict):
    """Helper to save reports and plots."""
    acc = accuracy_score(y_true, y_pred)
    print(f"  ✅ {model_name.upper()} Accuracy: {acc:.4f}")
    
    # Save Report
    report = classification_report(y_true, y_pred, output_dict=True)
    results_dict[model_name] = {"accuracy": acc, "report": report}
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name.upper()}")
    plt.savefig(os.path.join(REPORT_DIR, f"confusion_matrix_{model_name}.png"))
    plt.close()

def run_training():
    # 1. Load Data
    X_train_txt, y_train, X_val_txt, y_val, X_test_txt, y_test = load_data()
    results = {}

    # --- STAGE 1: BASELINE MODELS (Day 2) ---
    print("\n🚀 STARTING STAGE 1: BASELINE MODELS (CPU)")
    baseline_algos = ['lr', 'nb', 'svm']
    
    for algo in baseline_algos:
        print(f"  [Baseline] Training {algo.upper()}...")
        model = BaselineModel(algorithm=algo)
        model.train(X_train_txt, y_train)
        
        # Save Model
        model.save(os.path.join(BASELINE_DIR, f"{algo}_model.pkl"))
        
        # Evaluate
        preds = model.predict(X_test_txt)
        save_metrics(y_test, preds, algo, results)

    # --- STAGE 2: DEEP LEARNING MODELS (Day 3) ---
    print("\n🚀 STARTING STAGE 2: DEEP LEARNING MODELS (TensorFlow)")
    
    # Prepare Tokenizer once using helper
    helper = DeepModel()
    helper.prepare_tokenizer(X_train_txt)
    
    # Vectorize Data
    X_train_seq = helper.preprocess(X_train_txt)
    X_val_seq = helper.preprocess(X_val_txt)
    X_test_seq = helper.preprocess(X_test_txt)
    
    # Convert labels to numpy arrays
    y_train_np = np.array(y_train)
    y_val_np = np.array(y_val)
    y_test_np = np.array(y_test)

    deep_algos = ['bilstm', 'cnn']
    
    for arch in deep_algos:
        model = DeepModel(architecture=arch)
        model.load_tokenizer() # Load the shared tokenizer
        
        # Train
        model.train(X_train_seq, y_train_np, X_val_seq, y_val_np, epochs=5)
        model.save()
        
        # Evaluate
        print(f"  [Evaluate] Predicting with {arch}...")
        preds_probs = model.model.predict(X_test_seq)
        preds = preds_probs.argmax(axis=1)
        save_metrics(y_test_np, preds, arch, results)

    # Final Report
    with open(os.path.join(REPORT_DIR, "training_report_all.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✨ All models trained successfully! Reports in {REPORT_DIR}/")

if __name__ == "__main__":
    run_training()