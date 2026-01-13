@echo off
echo [INFO] KRIXION Internship Project - Setup Script
echo [INFO] -----------------------------------------

REM 1. Create Virtual Environment
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

REM 2. Activate Environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate

REM 3. Install Dependencies
echo [INFO] Installing requirements...
pip install -r requirements.txt

REM 4. Launch App
echo [INFO] Starting Application...
python app.py

pause