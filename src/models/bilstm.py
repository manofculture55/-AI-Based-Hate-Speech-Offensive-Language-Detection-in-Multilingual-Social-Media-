import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, Conv1D, GlobalMaxPooling1D, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import numpy as np

# CPU Optimization Constants [Source 24]
VOCAB_SIZE = 15000 
EMBEDDING_DIM = 100 
MAX_LEN = 120       
MODEL_DIR = "models/deep"  # Source 49 mentions 'models/deep/' folder

class DeepModel:
    def __init__(self, architecture='bilstm'):
        """
        architecture: 'bilstm' or 'cnn'
        """
        self.architecture = architecture
        self.model = None
        self.tokenizer = None
        os.makedirs(MODEL_DIR, exist_ok=True)

    def prepare_tokenizer(self, texts):
        """Fits tokenizer on training text."""
        print(f"  [DeepModel] Fitting tokenizer on {len(texts)} texts...")
        self.tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)
        
        # SAVE TOKENIZER: Required for offline inference
        tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.pickle')
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_tokenizer(self):
        """Loads the saved tokenizer."""
        tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.pickle')
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
        else:
            raise FileNotFoundError("Tokenizer not found. Train the model first.")

    def preprocess(self, texts):
        """Converts text to padded sequences."""
        if not self.tokenizer:
            raise Exception("Tokenizer not initialized!")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
        return np.array(padded)

    def build_model(self):
        """Builds the Neural Network based on Source 23/24."""
        self.model = Sequential()
        self.model.add(Input(shape=(MAX_LEN,)))
        self.model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM))
        
        if self.architecture == 'bilstm':
            # BiLSTM Architecture [Source 23]
            self.model.add(Bidirectional(LSTM(64, return_sequences=False)))
            self.model.add(Dropout(0.3))
            
        elif self.architecture == 'cnn':
            # CNN-TextCNN Architecture [Source 23]
            self.model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
            self.model.add(GlobalMaxPooling1D())
            self.model.add(Dropout(0.3))

        # Output Layer (0=Normal, 1=Offensive, 2=Hate)
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(3, activation='softmax'))

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=32):
        if self.model is None:
            self.build_model()
        
        print(f"\n  [DeepModel] Training {self.architecture.upper()}...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        return history

    def save(self):
        """Saves to .h5 file."""
        path = os.path.join(MODEL_DIR, f"{self.architecture}_model.h5")
        self.model.save(path)
        print(f"  [DeepModel] Saved to {path}")