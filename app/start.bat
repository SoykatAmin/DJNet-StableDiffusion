@echo off
echo 🚀 Starting DJ Transition Generator Web App
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python first.
    pause
    exit /b 1
)

REM Check if virtual environment exists, create if not
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo 📥 Installing requirements...
pip install -r requirements.txt

REM Check if model checkpoint exists
echo 🔍 Checking for model checkpoint...
if exist "..\checkpoints\5k\best_model_kaggle.pt" (
    echo ✅ Model checkpoint found
) else (
    echo ⚠️ Model checkpoint not found. The app will run but won't be able to generate transitions.
    echo    Please ensure you have a trained model in the checkpoints directory.
)

REM Start the application
echo 🌐 Starting web application...
echo.
echo Open your browser and go to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python app.py

pause
