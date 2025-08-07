@echo off
echo 🚀 Trilingual LLM API Server
echo =============================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python first.
    pause
    exit /b 1
)

echo ✅ Python found

REM Navigate to project directory
cd /d "%~dp0"

echo 📂 Current directory: %CD%

REM Check if virtual environment exists and activate it
if exist "venv\" (
    echo 🔧 Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ⚠️ No virtual environment found. Using system Python.
)

REM Install dependencies if needed
echo 🔍 Checking dependencies...

python -c "import fastapi" 2>nul || (
    echo 📦 Installing FastAPI...
    pip install fastapi uvicorn pydantic
)

python -c "import torch" 2>nul || (
    echo 📦 Installing PyTorch...
    pip install torch --index-url https://download.pytorch.org/whl/cpu
)

python -c "import requests" 2>nul || (
    echo 📦 Installing requests...
    pip install requests
)

echo.
echo 🎯 Choose an option:
echo [1] Start API Server
echo [2] Test API with sample conversations
echo [3] Start server and open browser
echo [4] Train model first, then start server
echo [5] Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo 🚀 Starting Trilingual LLM API Server...
    echo 📡 Server will be available at: http://localhost:8000
    echo 📖 API Documentation: http://localhost:8000/docs
    echo 🏠 Homepage: http://localhost:8000
    echo.
    echo Press Ctrl+C to stop the server
    python api_server.py
) else if "%choice%"=="2" (
    echo 🧪 Testing API with multilingual conversations...
    timeout /t 2 /nobreak >nul
    python test_api.py
) else if "%choice%"=="3" (
    echo 🚀 Starting server and opening browser...
    start http://localhost:8000
    python api_server.py
) else if "%choice%"=="4" (
    echo 🏋️ Training model first...
    python quick_train.py
    if not errorlevel 1 (
        echo ✅ Training completed! Starting API server...
        python api_server.py
    ) else (
        echo ❌ Training failed. Starting server anyway...
        python api_server.py
    )
) else if "%choice%"=="5" (
    echo 👋 Goodbye!
    exit /b 0
) else (
    echo ❌ Invalid choice. Starting API server...
    python api_server.py
)

echo.
echo 🎉 Session completed!
pause
