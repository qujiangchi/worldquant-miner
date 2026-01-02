# Test build script - test locally before pushing to GitHub
# This will test the build process on your local machine

param(
    [switch]$Windows,
    [switch]$All,
    [switch]$Clean
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Local Build Test Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Detect platform
$platform = $PSVersionTable.Platform
if ($IsWindows -or $env:OS -like "*Windows*") {
    $currentPlatform = "Windows"
} elseif ($IsMacOS -or $env:OSTYPE -like "*darwin*") {
    $currentPlatform = "macOS"
} elseif ($IsLinux -or $env:OSTYPE -like "*linux*") {
    $currentPlatform = "Linux"
} else {
    $currentPlatform = "Unknown"
}

Write-Host "Detected platform: $currentPlatform" -ForegroundColor Yellow
Write-Host ""

# Clean previous builds if requested
if ($Clean) {
    Write-Host "Cleaning previous builds..." -ForegroundColor Yellow
    $distDirs = @(
        "generation_two\dist",
        "generation_two\build",
        "dist",
        "build"
    )
    foreach ($dir in $distDirs) {
        if (Test-Path $dir) {
            Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
            Write-Host "  Removed: $dir" -ForegroundColor Gray
        }
    }
    Write-Host ""
}

# Check if we're in the right directory
if (-not (Test-Path "generation_two\build.py")) {
    Write-Host "ERROR: generation_two\build.py not found!" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory." -ForegroundColor Red
    exit 1
}

# Check if required files exist
Write-Host "Checking required files..." -ForegroundColor Cyan
$requiredFiles = @(
    "generation_two\constants\operatorRAW.json",
    "generation_two\gui\run_gui.py",
    "generation_two\build.py",
    "generation_two\setup.py"
)

$allExist = $true
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $file (MISSING!)" -ForegroundColor Red
        $allExist = $false
    }
}

if (-not $allExist) {
    Write-Host ""
    Write-Host "ERROR: Some required files are missing!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "All required files found!" -ForegroundColor Green
Write-Host ""

# Test build based on platform or parameter
if ($Windows -or ($All -and $currentPlatform -eq "Windows")) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Testing Windows EXE Build" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    Push-Location generation_two
    try {
        python build.py --exe
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "✓ Windows build test PASSED!" -ForegroundColor Green
            if (Test-Path "dist\generation-two.exe") {
                $size = (Get-Item "dist\generation-two.exe").Length / 1MB
                Write-Host "  EXE size: $([math]::Round($size, 2)) MB" -ForegroundColor Gray
            }
        } else {
            Write-Host ""
            Write-Host "✗ Windows build test FAILED!" -ForegroundColor Red
            exit 1
        }
    } finally {
        Pop-Location
    }
    Write-Host ""
}

if ($All -or $currentPlatform -eq "Linux") {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Testing Linux DEB Build" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Note: Linux build requires Linux environment" -ForegroundColor Yellow
    Write-Host "Skipping on $currentPlatform..." -ForegroundColor Yellow
    Write-Host ""
}

if ($All -or $currentPlatform -eq "macOS") {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Testing macOS DMG Build" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Note: macOS build requires macOS environment" -ForegroundColor Yellow
    Write-Host "Skipping on $currentPlatform..." -ForegroundColor Yellow
    Write-Host ""
}

# If no specific platform, test the current one
if (-not $Windows -and -not $All) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Testing Build for $currentPlatform" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    Push-Location generation_two
    try {
        python build.py
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "✓ Build test PASSED!" -ForegroundColor Green
        } else {
            Write-Host ""
            Write-Host "✗ Build test FAILED!" -ForegroundColor Red
            exit 1
        }
    } finally {
        Pop-Location
    }
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Build Test Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "If all tests passed, you can:" -ForegroundColor Yellow
Write-Host "  1. Commit the changes" -ForegroundColor White
Write-Host "  2. Push to GitHub" -ForegroundColor White
Write-Host "  3. Create a release tag" -ForegroundColor White
Write-Host ""
