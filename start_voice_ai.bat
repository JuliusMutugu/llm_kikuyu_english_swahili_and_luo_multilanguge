@echo off
echo 🎙️ Starting Trilingual Voice AI Assistant...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Install dependencies if needed
echo 📦 Checking dependencies...
pip show fastapi >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing FastAPI...
    pip install fastapi uvicorn pydantic
)

pip show torch >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing PyTorch...
    pip install torch --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo 🚀 Starting Voice AI Server...
echo.
echo 📍 Access URLs:
echo    • Simple Voice Chat: http://localhost:8000/simple
echo    • Advanced Chat:     http://localhost:8000/chat-ui  
echo    • API Docs:          http://localhost:8000/docs
echo    • Home Page:         http://localhost:8000
echo.
echo 💡 TIP: The Simple Voice Chat (/simple) is recommended for voice interaction!
echo.

REM Start the server
python api_server.py

pause
