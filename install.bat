@echo off
echo [INFO] KRIXION Internship Project - Setup Script
echo [INFO] -----------------------------------------

REM 1. Check/Create Virtual Environment
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
) else (
    echo [INFO] Virtual environment 'venv' found.
)

REM 2. Activate Environment  
echo [INFO] Activating virtual environment...
call venv\Scripts\activate

REM 3. Install Dependencies (Source 51)
echo [INFO] Installing requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM 3.5. Initialize Database (Day 1 - KRIXION Section 7)
echo [INFO] Initializing SQLite database...
python -m src.utils.db

REM 4. Normalize Datasets (Day 1)
echo [INFO] Processing multilingual datasets (HASOC, MDPI, IndoHate)...
python -m src.data.normalize

REM 5. Train All Models (Day 2-4)
echo [INFO] Running Unified Training Script...
echo [INFO] This will train LR, NB, SVM, BiLSTM, CNN + Transformer models.
python -m src.training.train

REM 6. Launch App (Day 5 - Ready!)
echo [INFO] ========================================
echo [INFO] ✅ KRIXION Setup COMPLETE!
echo [INFO] ========================================
echo [INFO] Launch app: python app.py
echo [INFO] Open: http://localhost:8080
echo [INFO] ========================================
pause
