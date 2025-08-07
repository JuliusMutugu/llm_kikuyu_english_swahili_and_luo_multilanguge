@echo off
title Trilingual AI - Modern Streamlit Interface

echo ========================================
echo    Trilingual AI Assistant - Streamlit
echo ========================================
echo.

echo Checking Python environment...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo.
echo Checking dependencies...
python -c "import streamlit, requests, torch" 2>nul
if errorlevel 1 (
    echo Installing missing dependencies...
    pip install streamlit requests torch
)

echo.
echo Starting API server in background...
start "API Server" cmd /c "python api_server.py"

echo Waiting for API server to start...
timeout /t 3 /nobreak >nul

echo.
echo ========================================
echo    Starting Streamlit Interface
echo ========================================
echo.
echo The modern web interface will open at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo ========================================
echo.

streamlit run streamlit_app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

echo.
echo Application stopped.
pause
