@echo off
REM QURE Business Demo Launcher
REM Launches the enhanced Streamlit UI for business demonstrations

echo ========================================
echo QURE - Business Demo Launcher
echo ========================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then: venv\Scripts\activate
    echo Then: pip install -r ui/requirements.txt
    pause
    exit /b 1
)

REM Activate venv and launch
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Checking dependencies...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [WARNING] Streamlit not found. Installing dependencies...
    pip install -r ui\requirements.txt
)

echo [INFO] Launching QURE Business Demo UI...
echo [INFO] Opening browser to http://localhost:8502
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

streamlit run ui\streamlit_app.py --server.port 8502
