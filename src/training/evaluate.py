import argparse
import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.models.bilstm import DeepModel

# Setup Arguments
parser = argparse.ArgumentParser(description="Evaluate a trained model.")
parser.add_argument('--model', type=str, required=True, help="Model name (e.g., 'lr', 'bilstm', 'cnn')")
parser.add_argument('--export', type=str, default='reports/', help="Directory to export reports")
args = parser.parse_args()

DATA_PATH = "data/clean_data.csv"
MODEL_DIR_BASE = "models/baseline"
MODEL_DIR_DEEP = "models/deep"

def load_test_data():
    """Loads data and filters for the test set (last 15%)."""
    df = pd.read_csv(DATA_PATH).dropna()
    # Logic to get the test split (simulated here for consistency with train.py)
    # In a real scenario, you might save X_test separately, but splitting with same seed works.
    from sklearn.model_selection import train_test_split
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()
    _, X_temp, _, y_temp = train_test_split(texts, labels, test_size=0.30, random_state=42, stratify=labels)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    return np.array(X_test), np.array(y_test)

def evaluate():
    os.makedirs(args.export, exist_ok=True)
    X_test, y_test = load_test_data()
    
    y_pred = None
    
    # Check if Baseline
    base_path = os.path.join(MODEL_DIR_BASE, f"{args.model}_model.pkl")
    if os.path.exists(base_path):
        print(f"Loading Baseline Model: {args.model}")
        model = joblib.load(base_path)
        y_pred = model.predict(X_test)

    # Check if Deep
    else:
        # For Deep Learning, we rely on the class structure
        helper = DeepModel(architecture=args.model)
        try:
            helper.load_tokenizer()
            # We need to reload the .h5 model into the helper
            # (The class loads it automatically in some setups, but let's be explicit)
            from tensorflow.keras.models import load_model
            h5_path = os.path.join(MODEL_DIR_DEEP, f"{args.model}_model.h5")
            if not os.path.exists(h5_path):
                raise FileNotFoundError(f"Model {args.model} not found in baseline or deep folders.")
            
            print(f"Loading Deep Model: {args.model}")
            helper.model = load_model(h5_path)
            X_seq = helper.preprocess(X_test)
            preds_probs = helper.model.predict(X_seq)
            y_pred = preds_probs.argmax(axis=1)
        except Exception as e:
            print(f"Error loading deep model: {e}")
            return

    # Generate Reports [Source 29]
    report = classification_report(y_test, y_pred, output_dict=True)
    report_path = os.path.join(args.export, f"classification_report_{args.model}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Report saved to {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title(f"Confusion Matrix - {args.model}")
    plt.savefig(os.path.join(args.export, f"confusion_matrix_{args.model}.png"))
    print(f"Matrix saved to {args.export}")

if __name__ == "__main__":
    evaluate()