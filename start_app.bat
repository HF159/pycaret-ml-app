@echo off
REM Batch file to run the PyCaret ML App with Ngrok

echo ============================================================
echo PyCaret ML App - Easy Launcher
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo [2/3] Starting application...
echo.
echo Choose how to run the app:
echo 1. Run locally (only accessible on this computer)
echo 2. Run with Ngrok (publicly accessible via internet)
echo.
set /p choice="Enter your choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo Starting locally at http://localhost:8501
    echo Press Ctrl+C to stop
    echo.
    streamlit run main.py
) else if "%choice%"=="2" (
    echo.
    echo Starting with Ngrok...
    python run_with_ngrok.py
) else (
    echo Invalid choice. Please run the script again.
    pause
)
