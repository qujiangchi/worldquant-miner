# GitHub Release Guide

This guide explains how to create GitHub releases with built executables (.exe, .deb, .dmg) for Generation Two.

## 🚀 Quick Start

### Option 1: Automatic Release (Recommended)

1. **Create and push a version tag:**
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. **GitHub Actions will automatically:**
   - Build Windows EXE
   - Build Linux DEB package
   - Build macOS DMG
   - Create a GitHub Release with all artifacts

### Option 2: Manual Release

1. Go to your GitHub repository
2. Click **Actions** → **Build and Release** → **Run workflow**
3. Enter the version number (e.g., `1.0.0`)
4. Click **Run workflow**
5. Wait for all builds to complete
6. The release will be created automatically

## 📋 Prerequisites

- GitHub repository with Actions enabled
- Proper permissions to create releases
- The workflow file is already set up at `.github/workflows/release.yml`

## 🔧 How It Works

The GitHub Actions workflow:

1. **Triggers on:**
   - Version tags (e.g., `v1.0.0`, `v1.2.3`)
   - Manual workflow dispatch

2. **Builds for all platforms:**
   - **Windows**: Creates `generation-two-{version}-windows.exe`
   - **Linux**: Creates `generation-two_{version}_*.deb`
   - **macOS**: Creates `generation-two.dmg`

3. **Creates a GitHub Release:**
   - Tag name: `v{version}`
   - Release name: `Release {version}`
   - Includes all built artifacts
   - Includes installation instructions

## 📦 Release Artifacts

After the workflow completes, you'll find:

- **Windows**: `generation-two-1.0.0-windows.exe`
- **Linux**: `generation-two_1.0.0-1_all.deb` (or similar)
- **macOS**: `generation-two.dmg`

All files are attached to the GitHub Release and can be downloaded by users.

## 🎯 Step-by-Step Release Process

### 1. Prepare Your Code

```bash
# Make sure all changes are committed
git add .
git commit -m "Prepare for release v1.0.0"

# Push to GitHub
git push origin main
```

### 2. Create a Version Tag

```bash
# Create an annotated tag (recommended)
git tag -a v1.0.0 -m "Release version 1.0.0"

# Or create a lightweight tag
git tag v1.0.0

# Push the tag to GitHub
git push origin v1.0.0
```

### 3. Monitor the Build

1. Go to **Actions** tab in GitHub
2. You'll see the workflow running
3. Wait for all three jobs to complete:
   - ✅ Build Windows EXE
   - ✅ Build Linux DEB
   - ✅ Build macOS DMG
   - ✅ Create GitHub Release

### 4. Verify the Release

1. Go to **Releases** in your GitHub repository
2. You should see the new release with all artifacts
3. Download and test the artifacts if needed

## 🔍 Troubleshooting

### Build Fails

**Check the Actions logs:**
- Go to **Actions** → Click on the failed workflow
- Check which job failed
- Review the error messages

**Common issues:**
- Missing dependencies in `requirements.txt`
- Path issues in `build.py`
- Platform-specific build tools missing

### Release Not Created

- Ensure all three build jobs completed successfully
- Check that you have write permissions for releases
- Verify the `GITHUB_TOKEN` is available (it's automatic in Actions)

### Artifacts Missing

- Check that files were created in `generation_two/dist/`
- Verify the artifact upload steps succeeded
- Check file paths in the workflow

## 📝 Customizing the Release

### Change Version Format

Edit `.github/workflows/release.yml` and modify the version extraction:

```yaml
- name: Determine version
  run: |
    VERSION=${GITHUB_REF#refs/tags/v}
    # Your custom version logic here
```

### Add Release Notes

The workflow includes basic release notes. To customize, edit the `body` section in the `create-release` step.

### Add Additional Artifacts

To include additional files in the release:

```yaml
- name: Prepare release assets
  run: |
    # Add your custom files
    cp README.md release-assets/
    cp LICENSE release-assets/
```

## 🏷️ Version Tagging Best Practices

1. **Use Semantic Versioning**: `v1.0.0`, `v1.0.1`, `v1.1.0`, `v2.0.0`
2. **Create Annotated Tags**: `git tag -a v1.0.0 -m "Release message"`
3. **Push Tags Explicitly**: `git push origin v1.0.0`
4. **Tag After Testing**: Only tag when code is ready for release

## 🔐 Security Notes

- **Credentials are NOT included** in the built executables
- Users must provide their own `credential.txt` file
- The build process doesn't access any secrets
- All dependencies are bundled with the executable

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Git Tagging Guide](https://git-scm.com/book/en/v2/Git-Basics-Tagging)
- [Semantic Versioning](https://semver.org/)

## 🆘 Getting Help

If you encounter issues:

1. Check the workflow logs in GitHub Actions
2. Review the build script: `generation_two/build.py`
3. Test builds locally first: `python generation_two/build.py --exe`
4. Check GitHub Actions status page for service issues

---

**Happy Releasing! 🎉**
