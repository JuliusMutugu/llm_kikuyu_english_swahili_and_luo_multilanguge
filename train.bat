@echo off
echo 🚀 Laptop LLM Training Launcher
echo ================================

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

REM Check if virtual environment exists
if exist "venv\" (
    echo 🔧 Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ⚠️ No virtual environment found. Using system Python.
)

REM Check dependencies
echo 🔍 Checking dependencies...
python -c "import torch; print('✅ PyTorch:', torch.__version__)" 2>nul || (
    echo ❌ PyTorch not found. Installing...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

python -c "import transformers; print('✅ Transformers:', transformers.__version__)" 2>nul || (
    echo ❌ Transformers not found. Installing...
    pip install transformers
)

python -c "import tensorboard; print('✅ TensorBoard found')" 2>nul || (
    echo ❌ TensorBoard not found. Installing...
    pip install tensorboard
)

echo.
echo 🎯 Choose training mode:
echo [1] Quick test (1 epoch, small batch)
echo [2] Standard training (3 epochs)
echo [3] Custom training
echo [4] Resume from checkpoint
echo [5] Test model only
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo 🏃 Starting quick test training...
    python train_laptop.py --epochs 1 --batch_size 1 --sequence_length 128
) else if "%choice%"=="2" (
    echo 🏋️ Starting standard training...
    python train_laptop.py --epochs 3 --sequence_length 256
) else if "%choice%"=="3" (
    echo 🔧 Custom training mode...
    set /p epochs="Number of epochs (default 3): "
    set /p batch_size="Batch size (0 for auto): "
    set /p seq_len="Sequence length (default 256): "
    
    if "%epochs%"=="" set epochs=3
    if "%batch_size%"=="" set batch_size=0
    if "%seq_len%"=="" set seq_len=256
    
    python train_laptop.py --epochs %epochs% --batch_size %batch_size% --sequence_length %seq_len%
) else if "%choice%"=="4" (
    echo 🔄 Resume training...
    set /p checkpoint="Checkpoint path (default: ./checkpoints/latest_model.pt): "
    if "%checkpoint%"=="" set checkpoint=./checkpoints/latest_model.pt
    python train_laptop.py --resume "%checkpoint%"
) else if "%choice%"=="5" (
    echo 🧪 Testing model...
    python src\models\modern_llm.py
) else (
    echo ❌ Invalid choice. Exiting.
    pause
    exit /b 1
)

echo.
echo 🎉 Operation completed!
echo.
echo 📊 Check your results:
echo    • Model checkpoints: ./checkpoints/
echo    • Training logs: ./logs/
echo    • TensorBoard: tensorboard --logdir=./logs
echo.
pause
