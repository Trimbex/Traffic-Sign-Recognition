Write-Host "🚦 Starting Traffic Sign Recognition Web App..." -ForegroundColor Cyan
Write-Host ""

# Check if FastAPI is installed
try {
    python -c "import fastapi" 2>$null
    if ($LASTEXITCODE -ne 0) { throw }
} catch {
    Write-Host "📥 Installing FastAPI dependencies..." -ForegroundColor Yellow
    pip install fastapi uvicorn python-multipart
    Write-Host ""
}

# Check if model file exists
if (-Not (Test-Path "custom_model.keras")) {
    Write-Host "❌ Model file 'custom_model.keras' not found!" -ForegroundColor Red
    Write-Host "Please make sure the trained model is in the current directory." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "🚀 Starting FastAPI server..." -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Your app will be available at: " -NoNewline -ForegroundColor White
Write-Host "http://localhost:8000" -ForegroundColor Cyan
Write-Host "📊 API documentation at: " -NoNewline -ForegroundColor White
Write-Host "http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Read-Host "Press Enter to exit"
