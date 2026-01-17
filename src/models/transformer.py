import os

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertTokenizer
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# Configuration
MODEL_NAME = 'distilbert-base-multilingual-cased'
CACHE_DIR = "models/transformer"
os.makedirs(CACHE_DIR, exist_ok=True)

class TransformerModel:
    def __init__(self):
        """
        Initializes DistilBERT for Feature Extraction (Inference Only).
        We do NOT fine-tune BERT [Source 25].
        """
        print(f"  [Transformer] Loading {MODEL_NAME}...")
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
            # FIX: Disable safetensors to force download of standard compatible weights
            self.encoder = TFDistilBertModel.from_pretrained(MODEL_NAME, use_safetensors=False)
        except Exception as e:
            print(f"  [Error] Failed to load Transformer. Check internet for first run? {e}")
            raise e
            
        # We use a simple classifier on top of BERT embeddings
        self.classifier = LogisticRegression(class_weight='balanced', max_iter=1000)
        self.is_fitted = False

    def get_embeddings(self, texts, batch_size=32):
        """
        Converts text to vector embeddings using DistilBERT.
        Process in batches to prevent CPU RAM overflow.
        """
        print(f"  [Transformer] Generating embeddings for {len(texts)} texts...")
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 1. Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='tf'
            )
            
            # 2. Inference (Get hidden states)
            output = self.encoder(encoded)
            
            # 3. Extract CLS token (First token represents the whole sentence)
            cls_embeddings = output.last_hidden_state[:, 0, :].numpy()
            all_embeddings.extend(cls_embeddings)
            
        return np.array(all_embeddings)

    def train(self, texts, labels):
        """
        Stage 3 Training Strategy:
        1. Extract Embeddings (Deep Learning)
        2. Train Classifier (Classical ML) on those embeddings
        """
        X_embeddings = self.get_embeddings(texts)
        
        print("  [Transformer] Training head classifier...")
        self.classifier.fit(X_embeddings, labels)
        self.is_fitted = True

    def predict(self, texts):
        if not self.is_fitted:
            raise Exception("Model not trained yet!")
            
        X_embeddings = self.get_embeddings(texts)
        return self.classifier.predict(X_embeddings)

    def save(self):
        """
        We only save the lightweight classifier head.
        We do NOT save the 500MB+ BERT model to keep the project ZIP small [Source 53].
        """
        path = os.path.join(CACHE_DIR, "transformer_head.pkl")
        joblib.dump(self.classifier, path)
        print(f"  [Transformer] Head saved to {path}")

    def load(self):
        path = os.path.join(CACHE_DIR, "transformer_head.pkl")
        if os.path.exists(path):
            self.classifier = joblib.load(path)
            self.is_fitted = True
        else:
            print("  [Transformer] No saved head found.")

