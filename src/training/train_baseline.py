import pandas as pd
import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from src.models.baseline import BaselineModel

# Constants
DATA_PATH = "data/clean_data.csv"
MODEL_DIR = "models/baseline"
REPORT_DIR = "reports"

def load_and_split_data():
    """
    Loads data and performs the 70/15/15 split required by Source [2].
    """
    print("📂 Loading Data...")
    df = pd.read_csv(DATA_PATH)
    
    # Drop rows where text is missing (safety check)
    df = df.dropna(subset=['text', 'label'])
    
    X = df['text']
    y = df['label']

    # 1. First Split: Separate Train (70%) from the rest (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # 2. Second Split: Split the rest (30%) into Validation (15%) and Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"📊 Data Split Complete:")
    print(f"   - Training:   {len(X_train)} rows (70%)")
    print(f"   - Validation: {len(X_val)} rows (15%)")
    print(f"   - Testing:    {len(X_test)} rows (15%)")
    
    return X_train, y_train, X_test, y_test

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Generates and saves a confusion matrix image (Source 38)."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Offensive', 'Hate'],
                yticklabels=['Normal', 'Offensive', 'Hate'])
    plt.title(f'Confusion Matrix - {model_name.upper()}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Save to reports folder
    save_path = os.path.join(REPORT_DIR, f"confusion_matrix_{model_name}.png")
    plt.savefig(save_path)
    plt.close()

def train_and_evaluate():
    # Ensure directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Load Data
    X_train, y_train, X_test, y_test = load_and_split_data()

    # Algorithms to train
    algorithms = ['lr', 'nb', 'svm']
    results = {}

    for algo in algorithms:
        print(f"\n🚀 --- Training {algo.upper()} Model ---")
        
        # 1. Initialize and Train
        model = BaselineModel(algorithm=algo)
        model.train(X_train, y_train)
        
        # 2. Evaluate on Test Data
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"🎯 {algo.upper()} Accuracy: {acc:.4f}")

        # 3. Generate Reports
        report_dict = classification_report(y_test, preds, output_dict=True)
        plot_confusion_matrix(y_test, preds, algo)
        
        # 4. Save Model
        save_path = os.path.join(MODEL_DIR, f"{algo}_model.pkl")
        model.save(save_path)

        # Store results for final JSON
        results[algo] = {
            "accuracy": acc,
            "classification_report": report_dict
        }

    # Save full JSON report
    json_path = os.path.join(REPORT_DIR, "classification_report.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✅ All models trained. Reports saved to '{REPORT_DIR}/'.")

if __name__ == "__main__":
    train_and_evaluate()