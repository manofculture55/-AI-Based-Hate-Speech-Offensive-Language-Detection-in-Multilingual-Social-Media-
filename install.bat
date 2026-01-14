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

REM 4. Train All Models (Baseline + Deep)
echo [INFO] Running Unified Training Script...
echo [INFO] This will train LR, NB, SVM, BiLSTM, and CNN models.
python -m src.training.train

REM 5. Launch App (To be enabled on Day 5)
echo [INFO] Note: Main Application (app.py) is scheduled for Day 5.
:: echo [INFO] Starting Application...
:: python app.py

echo [INFO] Setup and Training Complete!
pause