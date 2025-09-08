@echo off
echo 🚦 Starting Traffic Sign Recognition Web App...
echo.

REM Check if FastAPI is installed
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo 📥 Installing FastAPI dependencies...
    pip install fastapi uvicorn python-multipart
    echo.
)

REM Check if model file exists
if not exist "custom_model.keras" (
    echo ❌ Model file 'custom_model.keras' not found!
    echo Please make sure the trained model is in the current directory.
    pause
    exit /b 1
)

echo 🚀 Starting FastAPI server...
echo.
echo 🌐 Your app will be available at: http://localhost:8000
echo 📊 API documentation at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

pause
