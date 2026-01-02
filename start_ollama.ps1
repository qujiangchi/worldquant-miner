# PowerShell script to start Ollama on a specific port
# Usage: .\start_ollama.ps1 [port]
# Example: .\start_ollama.ps1 11434

param(
    [int]$Port = 11434
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Ollama Server" -ForegroundColor Cyan
Write-Host "Port: $Port" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Ollama is already running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:$Port/api/tags" -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "⚠️  Ollama is already running on port $Port" -ForegroundColor Yellow
        Write-Host "Press Ctrl+C to stop, or close this window to keep it running" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Testing connection..." -ForegroundColor Cyan
        $testResponse = Invoke-WebRequest -Uri "http://localhost:$Port/api/tags" -TimeoutSec 5
        Write-Host "✅ Ollama is responding!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Available models:" -ForegroundColor Cyan
        $models = ($testResponse.Content | ConvertFrom-Json).models
        foreach ($model in $models) {
            Write-Host "  - $($model.name)" -ForegroundColor White
        }
        exit 0
    }
} catch {
    # Ollama is not running, continue
}

# Try to find Ollama executable
$ollamaPaths = @(
    "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
    "C:\Program Files\Ollama\ollama.exe",
    "C:\Program Files (x86)\Ollama\ollama.exe"
)

$ollamaExe = $null
foreach ($path in $ollamaPaths) {
    if (Test-Path $path) {
        $ollamaExe = $path
        Write-Host "✅ Found Ollama at: $path" -ForegroundColor Green
        break
    }
}

# Check if ollama is in PATH
if (-not $ollamaExe) {
    $ollamaInPath = Get-Command ollama -ErrorAction SilentlyContinue
    if ($ollamaInPath) {
        $ollamaExe = "ollama"
        Write-Host "✅ Found Ollama in PATH" -ForegroundColor Green
    }
}

if (-not $ollamaExe) {
    Write-Host "❌ Ollama not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Ollama from: https://ollama.ai" -ForegroundColor Yellow
    Write-Host "Or ensure Ollama is in your PATH" -ForegroundColor Yellow
    exit 1
}

# Set environment variable for port
$env:OLLAMA_HOST = "0.0.0.0:$Port"

Write-Host ""
Write-Host "Starting Ollama server on port $Port..." -ForegroundColor Cyan
Write-Host "Environment: OLLAMA_HOST=$env:OLLAMA_HOST" -ForegroundColor Gray
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Ollama Server Output (Press Ctrl+C to stop):" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Start Ollama and show output
try {
    if ($ollamaExe -eq "ollama") {
        # Ollama is in PATH
        & ollama serve
    } else {
        # Use full path
        & $ollamaExe serve
    }
} catch {
    Write-Host ""
    Write-Host "❌ Failed to start Ollama: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Make sure Ollama is installed" -ForegroundColor White
    Write-Host "  2. Check if port $Port is already in use" -ForegroundColor White
    Write-Host "  3. Try running as Administrator" -ForegroundColor White
    exit 1
}
