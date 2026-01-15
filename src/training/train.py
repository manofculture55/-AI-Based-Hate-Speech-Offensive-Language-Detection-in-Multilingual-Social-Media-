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

    # --- STAGE 1: BASELINE (Classical ML) ---
    print("\n🚀 STAGE 1: BASELINE MODELS")
    for algo in ['lr', 'nb', 'svm']:
        print(f"   Training {algo.upper()}...")
        try:
            model = BaselineModel(algorithm=algo)
            model.train(X_train_txt, y_train)
            
            # Save Model
            save_path = os.path.join(BASELINE_DIR, f"{algo}_model.pkl")
            model.save(save_path)
            
            # Evaluate
            preds = model.predict(X_test_txt)
            save_metrics(y_test, preds, algo, results)
        except Exception as e:
            print(f"   Failed {algo}: {e}")

    # --- STAGE 2: DEEP LEARNING (BiLSTM & CNN) ---
    print("\n🚀 STAGE 2: DEEP LEARNING")
    
    # We must fit the tokenizer once so both models use the same vocabulary
    helper = DeepModel(architecture='bilstm')
    helper.prepare_tokenizer(X_train_txt) 

    for arch in ['bilstm', 'cnn']:
        print(f"   Training {arch.upper()}...")
        try:
            model = DeepModel(architecture=arch)
            model.load_tokenizer()
            
            # Convert text to sequences (Required for Neural Networks)
            X_train_seq = model.preprocess(X_train_txt)
            X_val_seq = model.preprocess(X_val_txt)
            X_test_seq = model.preprocess(X_test_txt)

            model.train(X_train_seq, y_train, X_val_seq, y_val, epochs=5)
            model.save()
            
            # Evaluate
            raw_probs = model.model.predict(X_test_seq, verbose=0)
            preds = raw_probs.argmax(axis=1)
            save_metrics(y_test, preds, arch, results)
        except Exception as e:
            print(f"   Failed {arch}: {e}")

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
    with open(os.path.join(REPORT_DIR, "training_report_all.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✨ All Stages Complete! Check {REPORT_DIR}/")

if __name__ == "__main__":
    run_training()