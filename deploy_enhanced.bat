@echo off
REM Docker Deployment Script with Enhanced Learning (Windows)

title Trilingual AI - Enhanced Docker Deploy

echo ==========================================
echo   Trilingual AI - Enhanced Docker Deploy
echo ==========================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo ✅ Docker is running
echo.

REM Build and start services
echo 🔨 Building Docker images...
docker-compose build

if errorlevel 1 (
    echo ❌ ERROR: Failed to build Docker images!
    pause
    exit /b 1
)

echo ✅ Docker images built successfully
echo.

REM Start core services
echo 🚀 Starting core services...
docker-compose up -d trilingual-api streamlit-ui

if errorlevel 1 (
    echo ❌ ERROR: Failed to start services!
    pause
    exit /b 1
)

echo ✅ Core services started
echo.

REM Wait for services to be ready
echo ⏳ Waiting for services to initialize...
timeout /t 10 /nobreak >nul

REM Check service health
echo 🔍 Checking service health...

REM Check API health
curl -s -o nul -w "%%{http_code}" http://localhost:8001/health >temp_api_health.txt 2>nul
set /p API_HEALTH=<temp_api_health.txt
del temp_api_health.txt 2>nul

if "%API_HEALTH%"=="200" (
    echo ✅ API Server: Healthy
) else (
    echo ⚠️  API Server: Starting... ^(may take a few more seconds^)
)

REM Check Streamlit health
curl -s -o nul -w "%%{http_code}" http://localhost:8501/_stcore/health >temp_streamlit_health.txt 2>nul
set /p STREAMLIT_HEALTH=<temp_streamlit_health.txt
del temp_streamlit_health.txt 2>nul

if "%STREAMLIT_HEALTH%"=="200" (
    echo ✅ Streamlit UI: Healthy
) else (
    echo ⚠️  Streamlit UI: Starting... ^(may take a few more seconds^)
)

echo.

REM Initialize dictionary learning (optional)
echo 📚 Initialize Luo dictionary learning? ^(y/n^)
set /p INIT_LEARNING=

if /i "%INIT_LEARNING%"=="y" (
    echo 🔄 Running initial dictionary learning...
    docker-compose --profile learning run --rm dictionary-learner
    
    if errorlevel 0 (
        echo ✅ Dictionary learning completed
    ) else (
        echo ⚠️  Dictionary learning had issues ^(check logs^)
    )
)

echo.
echo ==========================================
echo 🎉 Deployment Complete!
echo ==========================================
echo.
echo 📱 Access your Enhanced Trilingual AI:
echo    🌐 Web Interface: http://localhost:8501
echo    🔌 API Server:    http://localhost:8001
echo.
echo 🔧 Enhanced Features Available:
echo    ✅ Federated Learning from online sources
echo    ✅ Luo dictionary learning from Glosbe.com
echo    ✅ Advanced analytics and feedback
echo    ✅ Privacy-preserving vocabulary expansion
echo.
echo 📊 Useful Commands:
echo    docker-compose logs -f streamlit-ui  ^# View UI logs
echo    docker-compose logs -f trilingual-api  ^# View API logs
echo    docker-compose down  ^# Stop all services
echo    docker-compose --profile learning run dictionary-learner  ^# Run learning manually
echo.
echo 🚀 Your enhanced AI is ready!
echo    Open http://localhost:8501 and check the 'Learning' tab!
echo.
pause
