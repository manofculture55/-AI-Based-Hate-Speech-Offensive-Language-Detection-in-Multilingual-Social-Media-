import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

class BaselineModel:
    def __init__(self, algorithm='lr'):
        """
        Initialize Baseline Model Architecture.
        algorithms: 
        - 'lr': Logistic Regression (Source 22)
        - 'nb': Naive Bayes (Source 22)
        - 'svm': Support Vector Machine (Source 22)
        """
        self.algorithm = algorithm
        self.model = self._build_pipeline()

    def _build_pipeline(self):
        # We use a Pipeline to combine TF-IDF Vectorization with the Classifier.
        # This ensures raw text is automatically converted to numbers during prediction.
        
        # 1. Choose Classifier
        if self.algorithm == 'lr':
            # Logistic Regression: Good baseline, balanced class weights handles uneven data
            clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')
        elif self.algorithm == 'nb':
            # Naive Bayes: Very fast, standard for text classification
            clf = MultinomialNB()
        elif self.algorithm == 'svm':
            # SVM: Uses Linear kernel for speed and high accuracy on text data
            # probability=True is required to get confidence scores for the UI
            clf = SVC(kernel='linear', probability=True, class_weight='balanced')
        else:
            raise ValueError("Unknown algorithm. Choose 'lr', 'nb', or 'svm'.")

        # 2. Create Pipeline
        # TfidfVectorizer: Converts text to number vectors (Source 11)
        # max_features=5000: Keeps model size small (< 1.2 GB requirement)
        return Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', clf)
        ])

    def train(self, X_train, y_train):
        """Train the model on text data."""
        print(f"⚡ Training {self.algorithm.upper()} model...")
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Return predicted class labels (0, 1, 2)."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Return probability scores (confidence) for the UI."""
        return self.model.predict_proba(X)

    def save(self, filepath):
        """Save model using joblib as required by Source 28."""
        joblib.dump(self.model, filepath)
        print(f"✅ Model saved to {filepath}")