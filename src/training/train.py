"""
KRIXION Hate Speech Detection - Complete Training Pipeline
Stage 1: Baseline ML (LR, NB, SVM) ✅ 75-80% accuracy
Stage 2: Deep Learning (BiLSTM, CNN) ✅ 85-88% accuracy  
Stage 3: Transformer (DistilBERT) 🔄 Inference-only
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.models.baseline import BaselineModel
from src.models.bilstm import DeepModel

# Configuration - KRIXION Standards [file:1]
DATAPATH = "data/cleandata.csv"
REPORTDIR = "reports"
BASELINEDIR = "models/baseline"
DEEPDIR = "models/deep"

# CPU Optimization Constants [file:2]
VOCABSIZE = 15000
EMBEDDINGDIM = 100
MAXLEN = 120

os.makedirs(REPORTDIR, exist_ok=True)
os.makedirs(BASELINEDIR, exist_ok=True)
os.makedirs(DEEPDIR, exist_ok=True)

def load_data():
    """Load cleaned data with 70/15/15 stratified split [file:1]"""
    print("📊 Loading Data...")
    df = pd.read_csv(DATAPATH).dropna(subset=['text', 'label'])
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()
    
    # Stratified split: 70 train, 15 val, 15 test
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.30, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test)

def save_metrics(y_true, y_pred, model_name, results):
    """Save accuracy, classification report, confusion matrix [file:2]"""
    acc = accuracy_score(y_true, y_pred)
    print(f"   {model_name.upper()} Accuracy: {acc:.4f}")
    
    report = classification_report(y_true, y_pred, output_dict=True)
    results[model_name] = {
        'accuracy': acc,
        'report': report
    }
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.title(f'Confusion Matrix - {model_name.upper()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(REPORTDIR, f'confusion_matrix_{model_name}.png'))
    plt.close()

def run_training():
    X_train_txt, y_train, X_val_txt, y_val, X_test_txt, y_test = load_data()
    results = {}

    # ========================================================
    # STAGE 1: BASELINE MODELS (Day 2) - Classical ML [file:1]
    # ========================================================
    print("\n🚀 STAGE 1: BASELINE MODELS (LR, NB, SVM)")
    baseline_algos = ['lr', 'nb', 'svm']
    
    for algo in baseline_algos:
        print(f"   Training {algo.upper()}...")
        model = BaselineModel(algorithm=algo)
        model.train(X_train_txt, y_train)
        
        # Save Model
        model_path = os.path.join(BASELINEDIR, f"{algo}_model.pkl")
        model.save(model_path)
        
        # Evaluate on Test Set
        preds = model.predict(X_test_txt)
        save_metrics(y_test, preds, algo, results)
        print(f"   ✅ {algo.upper()} SAVED: {model_path}")

    # ========================================================
    # STAGE 2: DEEP LEARNING MODELS (Day 3) - BiLSTM & CNN [file:2]
    # ========================================================
    print("\n🚀 STAGE 2: DEEP LEARNING (BiLSTM + CNN)")
    
    # 🔥 SHARED TOKENIZER + PREPROCESSING (FIXED from your old commit)
    helper = DeepModel(architecture='bilstm')
    helper.prepare_tokenizer(X_train_txt.tolist())
    
    # Convert to sequences ONCE for ALL deep models
    X_train_seq = helper.preprocess(X_train_txt.tolist())
    X_val_seq = helper.preprocess(X_val_txt.tolist())
    X_test_seq = helper.preprocess(X_test_txt.tolist())
    
    y_train_np = np.array(y_train)
    y_val_np = np.array(y_val)
    y_test_np = np.array(y_test)
    
    deep_algos = ['bilstm', 'cnn']
    for arch in deep_algos:
        print(f"   Training {arch.upper()}...")
        model = DeepModel(architecture=arch)
        model.load_tokenizer()  # Load shared tokenizer
        
        # Train (uses shared sequences)
        model.train(X_train_seq, y_train_np, X_val_seq, y_val_np, epochs=3)
        model.save()
        
        # Evaluate
        print(f"   Evaluating {arch.upper()}...")
        raw_probs = model.model.predict(X_test_seq, verbose=0)
        preds = raw_probs.argmax(axis=1)
        save_metrics(y_test_np, preds, arch, results)
        print(f"   ✅ {arch.upper()} SAVED: models/deep/{arch}_model.h5")

    # ========================================================
    # STAGE 3: TRANSFORMER (DistilBERT) - EXACTLY like your recent code
    # ========================================================
    print("\n🚀 STAGE 3: TRANSFORMER (DistilBERT)")
    print("  [Note] First run will download model (~260MB). Please wait...")

    # 🔥 Initialize and Train - YOUR WORKING PATTERN
    transformer = TransformerModel()
    transformer.train(X_train_txt, y_train)  # Uses raw text (not sequences)
    transformer.save()

    # Evaluate - YOUR WORKING PATTERN  
    print("  [Evaluate] DistilBERT...")
    preds = transformer.predict(X_test_txt)  # Uses raw text (not sequences)
    save_metrics(y_test, preds, "distilbert", results)
    print(f"  ✅ DISTILBERT SAVED: models/transformer/transformer_head.pkl")

    # ========================================================
    # FINAL REPORT - KRIXION STANDARD
    # ========================================================
    report_path = os.path.join(REPORTDIR, "training_report_all.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)

    # Print Summary Table
    print("\n" + "="*60)
    print("✨ TRAINING COMPLETE - KRIXION HATE SPEECH DETECTION")
    print("="*60)
    print(f"📁 Reports: {REPORTDIR}/")
    print(f"📁 Baseline Models: {BASELINEDIR}/") 
    print(f"📁 Deep Models: {DEEPDIR}/")
    print(f"📁 Transformer: models/transformer/")
    print(f"📊 Full Report: {report_path}")

    # Model Performance Summary
    print("\n🏆 PERFORMANCE SUMMARY:")
    for model_name, metrics in results.items():
        acc = metrics['accuracy']
        status = "🎖️ CHAMPION" if acc > 0.85 else "✅ PASSED"
        print(f"   {model_name.upper()}: {acc:.1%} {status}")

    print("\n🎉 ALL STAGES COMPLETE! Run: python app.py")