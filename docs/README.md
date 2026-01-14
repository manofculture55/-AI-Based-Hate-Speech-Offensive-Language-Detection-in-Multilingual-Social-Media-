# 🛡️ AI-Based Hate Speech Detection (Multilingual)

**KRIXION Internship Project**  
**Mode:** Offline Execution (No API Calls)  
**Languages:** Hindi, English, Hinglish  

## 📌 Project Overview
A fully offline web application to detect hate speech and offensive language in social media text. The system uses Machine Learning (Logic Regression, SVM) and Deep Learning (BiLSTM) to classify text into three categories: **Normal**, **Offensive**, or **Hate**.The system uses a multi-stage architecture:
1.  **Stage 1:** Classical ML (Logistic Regression, Naïve Bayes, SVM).
2.  **Stage 2:** Deep Learning (BiLSTM, CNN) for context-aware detection.
3.  **Stage 3:** Transformers (DistilBERT) for high-precision inference (Day 4).

## 🚀 How to Run (Offline)

### Prerequisites
- Python 3.10 or higher
- Windows/Mac/Linux

### One-Click Installation
1. Double-click **`install.bat`**.
2. This will:
   - Create a virtual environment (`venv`).
   - Install all dependencies (TensorFlow, Scikit-learn, NiceGUI)..
   - Automatically train ALL models** (Baseline + Deep Learning).
   - (On Day 5) Launch the App.

### Manual Execution
If you prefer running commands manually:

```bash
# 1. Activate Virtual Environment
.\venv\Scripts\activate   # (Windows)
# source venv/bin/activate  # (Mac/Linux)

# 2. Train Models (Retrain all stages)
python -m src.training.train

# 3. Run App (Day 5 Task)
# python app.py