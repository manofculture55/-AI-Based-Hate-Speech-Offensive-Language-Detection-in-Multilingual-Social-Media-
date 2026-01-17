# AI-Based Hate Speech Detection (Multilingual)
**KRIXION Internship Project**  
**Mode:** Offline Execution (No API Calls)  
**Languages:** Hindi, English, Hinglish  
**Developer:** [Your Name]

## 1. Project Overview
A fully offline web application to detect hate speech and offensive language in social media text. The system uses Machine Learning (Logistic Regression, SVM) and Deep Learning (BiLSTM) to classify text into three categories: **Normal**, **Offensive**, or **Hate** [Source 83].

## 2. Key Features
*   **Multilingual Support:** Handles Hindi (`hi`), English (`en`), and Hinglish (`hi-en`).
*   **Offline Execution:** Runs entirely on CPU without internet.
*   **Real-Time Dashboard:** NiceGUI interface with History and Analytics tabs.
*   **Latency:** < 2 seconds per prediction.
*   **Hidden Admin Panel:** A secret route for model management and retraining [Source 87].
*   **Dynamic UI:** Responsive design with built-in Dark/Light Mode toggle for comfortable viewing in any lighting condition.

## 3. How to Run (Offline)

### Prerequisites
- Python 3.10 or higher
- Windows/Mac/Linux

### One-Click Installation (Windows)
1. Double-click **`install.bat`** in the project folder.
2. This will automatically:
   - Create a virtual environment (`venv`).
   - Install all dependencies.
   - Train ALL models (Baseline + Deep Learning).
   - Launch the App.

### Manual Execution
If you prefer running commands manually:
```bash
# 1. Activate Virtual Environment
.\venv\Scripts\activate

# 2. Train Models (Required for first run)
python -m src.training.train

# 3. Launch Application
python app.py
4. 🔐 Secret Admin Panel Access
This application includes a hidden administrative console that is not linked from the main navigation bar.
How to Access:
1. Open the app in your browser (usually http://localhost:8080).
2. Manually append /admin to the URL:
Admin Features:
• Retrain Models: Click the "RE-TRAIN ALL MODELS" button to trigger the training pipeline in the background [Source 33].
• System Logs: View real-time logs of the training process and errors.
• Note: Do not close the application window while retraining is in progress.

---