# Local Build Testing Guide

Before pushing to GitHub, test your builds locally to catch issues early.

## Quick Test

### Windows
```powershell
.\test_build_local.ps1
```

### Linux/Mac
```bash
chmod +x test_build_local.sh
./test_build_local.sh
```

## Manual Test

### 1. Test Windows Build
```powershell
cd generation_two
python build.py --exe
```

### 2. Test Linux Build (on Linux)
```bash
cd generation_two
python build.py --deb
```

### 3. Test macOS Build (on macOS)
```bash
cd generation_two
python build.py --dmg
```

## Common Issues

### Constants File Missing
If you see errors about `operatorRAW.json`:
1. The file should be in `constants/operatorRAW.json` (root)
2. The build script will copy it to `generation_two/constants/` automatically
3. If it's missing, copy it manually:
   ```powershell
   Copy-Item constants\operatorRAW.json generation_two\constants\operatorRAW.json
   ```

### Setup.py Errors
If you see "package directory 'generation_two' does not exist":
- Make sure you're running from the project root
- The build script should handle this automatically

### Path Issues
If you see path duplication or "file not found" errors:
- The build script now uses resolved absolute paths
- Make sure all files exist before building

## What Gets Built

- **Windows**: `generation_two/dist/generation-two.exe`
- **Linux**: `generation_two/dist/generation-two_*.deb`
- **macOS**: `generation_two/dist/generation-two.dmg`

## After Local Testing

Once local tests pass:
1. Commit your changes
2. Push to GitHub
3. Create a release tag
4. GitHub Actions will build for all platforms
