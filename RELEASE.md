# How to Create Releases

This guide explains how to create GitHub releases with built executables (.exe, .deb, .dmg).

## Quick Start

### Use the Release Script (Recommended)

**Windows:**
```powershell
.\release.ps1 --patch    # 1.0.0 -> 1.0.1
```

**Linux/Mac:**
```bash
./release.sh --patch
```

The script will:
1. Read current version from `pyproject.toml`
2. Update version files automatically
3. Create and push a Git tag
4. Trigger GitHub Actions to build and release

### Manual Method

```bash
# Create and push a version tag
git tag v1.0.0
git push origin v1.0.0
```

GitHub Actions will automatically:
- Build Windows EXE
- Build Linux DEB package
- Build macOS DMG
- Create a GitHub Release with all artifacts

## What Gets Built

- **Windows**: `generation-two-1.0.0-windows.exe`
- **Linux**: `generation-two_1.0.0-1_all.deb`
- **macOS**: `generation-two.dmg`

All files are automatically attached to the GitHub Release.

## Troubleshooting

### Release Not Created

If you created a tag but no release appeared:

```powershell
# Create release manually
.\create_release.ps1 -Version 1.0.0
```

Or go to GitHub → **Releases** → **Draft a new release** and select your tag.

### Build Fails

1. Check **Actions** tab for error logs
2. Verify `generation_two/constants/operatorRAW.json` exists
3. Test locally first: `cd generation_two && python build.py --exe`

### Constants File Missing

The build script automatically copies `constants/operatorRAW.json` to `generation_two/constants/` if needed. If it's still missing, copy it manually:

```powershell
Copy-Item constants\operatorRAW.json generation_two\constants\operatorRAW.json
```

## Testing Locally

Before releasing, test the build:

```powershell
# Windows
.\test_build_local.ps1

# Or manually
cd generation_two
python build.py --exe
```

---

**That's it!** Run `.\release.ps1 --patch` to create a new release.
