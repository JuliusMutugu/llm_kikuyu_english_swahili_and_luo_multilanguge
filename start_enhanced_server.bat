@echo off
echo Enhanced Trilingual LLM - Starting Server
echo ========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Install dependencies if needed
echo Checking dependencies...
python -c "import fastapi, uvicorn, torch" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install fastapi uvicorn torch numpy jinja2 python-multipart
)

REM Start the enhanced server
echo Starting enhanced server...
python enhanced_api_server.py

pause
