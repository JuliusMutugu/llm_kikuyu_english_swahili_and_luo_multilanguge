@echo off
title Trilingual AI - Enhanced Learning Platform

echo ========================================
echo   Trilingual AI Assistant - Enhanced
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
    echo Installing core dependencies...
    pip install streamlit requests torch
)

echo.
echo Checking enhanced learning dependencies...
python -c "import aiohttp, beautifulsoup4" 2>nul
if errorlevel 1 (
    echo Installing learning dependencies...
    pip install aiohttp beautifulsoup4 numpy pandas matplotlib seaborn
)

echo.
echo ========================================
echo    Starting Dictionary Learning
echo ========================================
echo.
echo Learning vocabulary from online sources...
echo This will improve Luo, Kiswahili, and Kikuyu support
echo.

REM Run dictionary learning in background
start "Dictionary Learning" cmd /c "python online_dictionary_learner.py && echo Dictionary learning complete && pause"

echo.
echo Starting API server in background...
start "API Server" cmd /c "python multi_model_api.py"

echo Waiting for services to start...
timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo    Starting Enhanced Streamlit UI
echo ========================================
echo.
echo Features available:
echo  - Real-time trilingual chat
echo  - Federated learning from online sources
echo  - Advanced analytics and feedback
echo  - Privacy-preserving learning
echo  - Cultural context optimization
echo.
echo The interface will open at: http://localhost:8501
echo.
echo Key Features:
echo  ^> Learning Tab - Monitor federated learning
echo  ^> Dictionary Integration - Luo vocabulary from Glosbe
echo  ^> Analytics Dashboard - Performance tracking
echo  ^> Feedback System - Continuous improvement
echo.
echo Press Ctrl+C to stop the application
echo ========================================
echo.

streamlit run streamlit_app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

echo.
echo Application stopped.
echo.
echo To restart dictionary learning manually:
echo   python online_dictionary_learner.py
echo.
pause
