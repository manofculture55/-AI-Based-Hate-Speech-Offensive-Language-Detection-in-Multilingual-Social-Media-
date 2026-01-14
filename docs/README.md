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


# KRIXION: AI-Based Hate Speech Detection (Offline)

**Internship Project | KRIXION Technologies Pvt. Ltd.**
**Developer:** [Your Name]
**Mode:** 100% Offline (No APIs) | **Stack:** Python, NiceGUI, TensorFlow, SQLite

## 📌 Project Overview
This application is a multilingual AI system designed to detect **Hate Speech** and **Offensive Language** in social media text. It specifically targets **Hindi-English Code-Mixed (Hinglish)** content, which is common in India but hard for traditional models to catch.

The system uses a **BiLSTM (Deep Learning)** architecture to analyze context and saves all predictions to a local SQLite database for auditing.

## 🚀 Key Features
*   **Multilingual Support:** Handles Hindi (`hi`), English (`en`), and Hinglish (`hi-en`).
*   **Offline Execution:** Runs entirely on CPU without internet.
*   **Real-Time Dashboard:** NiceGUI interface with History and Analytics tabs.
*   **Admin Panel:** Hidden route (`/admin`) for retraining and data management.
*   **Latency:** < 2 seconds per prediction.

## 🛠️ Installation & Setup
1.  **Clone/Unzip** the project folder.
2.  **Run the Installer** (Windows):
    ```bash
    install.bat
    ```
    *This creates a virtual environment, installs dependencies, and trains the models.*

3.  **Manual Setup** (Linux/Mac/Alternative):
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python -m src.training.train
    ```

## 💻 How to Run
Once installed, launch the application:
```bash
python app.py