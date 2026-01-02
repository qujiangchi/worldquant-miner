"""
Build script for Generation Two
Creates executables for Windows (exe), Linux (deb), and macOS (dmg)
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, check=True):
    """Run a shell command"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check)
    return result.returncode == 0

def build_windows_exe():
    """Build Windows executable using PyInstaller"""
    print("\n" + "="*60)
    print("Building Windows EXE...")
    print("="*60)
    
    # Install PyInstaller if not available
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        run_command([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Create spec file
    spec_content = """# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['gui/run_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('constants/operatorRAW.json', 'constants'),
    ],
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
        'generation_two',
        'generation_two.gui',
        'generation_two.core',
        'generation_two.ollama',
        'generation_two.data_fetcher',
        'generation_two.storage',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='generation-two',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path if you have one
)
"""
    
    spec_file = Path("generation_two.spec")
    spec_file.write_text(spec_content)
    
    # Build
    os.chdir("generation_two")
    run_command([sys.executable, "-m", "PyInstaller", "--clean", "../generation_two.spec"])
    os.chdir("..")
    
    # Move exe to dist
    dist_dir = Path("dist")
    dist_dir.mkdir(exist_ok=True)
    
    exe_path = Path("generation_two/dist/generation-two.exe")
    if exe_path.exists():
        shutil.move(str(exe_path), "dist/generation-two.exe")
        print("✅ Windows EXE built: dist/generation-two.exe")
    else:
        print("❌ EXE not found in expected location")

def build_linux_deb():
    """Build Debian package"""
    print("\n" + "="*60)
    print("Building Linux DEB package...")
    print("="*60)
    
    # Install build dependencies
    print("Installing build dependencies...")
    run_command([sys.executable, "-m", "pip", "install", "stdeb"], check=False)
    
    # Build source distribution
    print("Building source distribution...")
    run_command([sys.executable, "setup.py", "sdist"])
    
    # Convert to deb
    print("Converting to DEB...")
    os.chdir("dist")
    run_command([sys.executable, "-m", "stdeb", "generation-two-1.0.0.tar.gz"])
    os.chdir("..")
    
    # Find and move deb file
    deb_files = list(Path(".").rglob("*.deb"))
    if deb_files:
        deb_file = deb_files[0]
        dist_dir = Path("dist")
        dist_dir.mkdir(exist_ok=True)
        shutil.move(str(deb_file), f"dist/{deb_file.name}")
        print(f"✅ Linux DEB built: dist/{deb_file.name}")
    else:
        print("❌ DEB file not found")

def build_macos_dmg():
    """Build macOS DMG package"""
    print("\n" + "="*60)
    print("Building macOS DMG...")
    print("="*60)
    
    if sys.platform != "darwin":
        print("⚠️  DMG can only be built on macOS")
        return
    
    # Install PyInstaller if not available
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        run_command([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Build app bundle
    os.chdir("generation_two")
    run_command([
        sys.executable, "-m", "PyInstaller",
        "--name=GenerationTwo",
        "--windowed",
        "--onedir",
        "--add-data=constants/operatorRAW.json:constants",
        "gui/run_gui.py"
    ])
    os.chdir("..")
    
    # Create DMG using create-dmg (requires: brew install create-dmg)
    print("Creating DMG...")
    app_path = Path("generation_two/dist/GenerationTwo.app")
    if app_path.exists():
        run_command([
            "create-dmg",
            "--volname", "Generation Two",
            "--window-pos", "200", "120",
            "--window-size", "800", "400",
            "--icon-size", "100",
            "--app-drop-link", "600", "185",
            "dist/generation-two.dmg",
            str(app_path)
        ], check=False)
        print("✅ macOS DMG built: dist/generation-two.dmg")
    else:
        print("❌ App bundle not found")

def main():
    """Main build function"""
    print("Generation Two Build Script")
    print("="*60)
    
    # Clean previous builds
    if Path("dist").exists():
        shutil.rmtree("dist")
    if Path("build").exists():
        shutil.rmtree("build")
    
    # Create dist directory
    Path("dist").mkdir(exist_ok=True)
    
    # Detect platform and build accordingly
    platform = sys.platform.lower()
    
    if platform.startswith("win"):
        build_windows_exe()
    elif platform.startswith("linux"):
        build_linux_deb()
    elif platform == "darwin":
        build_macos_dmg()
    else:
        print(f"⚠️  Unknown platform: {platform}")
        print("Available build options:")
        print("  - Windows: python build.py --exe")
        print("  - Linux: python build.py --deb")
        print("  - macOS: python build.py --dmg")
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        if "--exe" in sys.argv:
            build_windows_exe()
        if "--deb" in sys.argv:
            build_linux_deb()
        if "--dmg" in sys.argv:
            build_macos_dmg()
        if "--all" in sys.argv:
            build_windows_exe()
            build_linux_deb()
            if sys.platform == "darwin":
                build_macos_dmg()
    
    print("\n" + "="*60)
    print("Build complete!")
    print("="*60)

if __name__ == "__main__":
    main()
