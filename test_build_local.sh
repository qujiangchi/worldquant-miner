#!/bin/bash
# Test build script - test locally before pushing to GitHub
# This will test the build process on your local machine

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse arguments
WINDOWS=false
ALL=false
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --windows)
            WINDOWS=true
            shift
            ;;
        --all)
            ALL=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Local Build Test Script${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Detect platform
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    CURRENT_PLATFORM="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    CURRENT_PLATFORM="macOS"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    CURRENT_PLATFORM="Windows"
else
    CURRENT_PLATFORM="Unknown"
fi

echo -e "${YELLOW}Detected platform: $CURRENT_PLATFORM${NC}"
echo ""

# Clean previous builds if requested
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}Cleaning previous builds...${NC}"
    rm -rf generation_two/dist generation_two/build dist build
    echo -e "${GREEN}  Cleaned build directories${NC}"
    echo ""
fi

# Check if we're in the right directory
if [ ! -f "generation_two/build.py" ]; then
    echo -e "${RED}ERROR: generation_two/build.py not found!${NC}"
    echo -e "${RED}Please run this script from the project root directory.${NC}"
    exit 1
fi

# Check if required files exist
echo -e "${CYAN}Checking required files...${NC}"
REQUIRED_FILES=(
    "generation_two/constants/operatorRAW.json"
    "generation_two/gui/run_gui.py"
    "generation_two/build.py"
    "generation_two/setup.py"
)

ALL_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}  ✓ $file${NC}"
    else
        echo -e "${RED}  ✗ $file (MISSING!)${NC}"
        ALL_EXIST=false
    fi
done

if [ "$ALL_EXIST" = false ]; then
    echo ""
    echo -e "${RED}ERROR: Some required files are missing!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}All required files found!${NC}"
echo ""

# Test build based on platform
if [ "$CURRENT_PLATFORM" = "Linux" ] || [ "$ALL" = true ]; then
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  Testing Linux DEB Build${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    cd generation_two
    python build.py --deb
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Linux build test PASSED!${NC}"
    else
        echo ""
        echo -e "${RED}✗ Linux build test FAILED!${NC}"
        exit 1
    fi
    cd ..
    echo ""
fi

if [ "$CURRENT_PLATFORM" = "macOS" ] || [ "$ALL" = true ]; then
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  Testing macOS DMG Build${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    cd generation_two
    python build.py --dmg
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ macOS build test PASSED!${NC}"
    else
        echo ""
        echo -e "${RED}✗ macOS build test FAILED!${NC}"
        exit 1
    fi
    cd ..
    echo ""
fi

if [ "$CURRENT_PLATFORM" = "Windows" ] || [ "$WINDOWS" = true ]; then
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  Testing Windows EXE Build${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo -e "${YELLOW}Note: Windows build requires Windows environment${NC}"
    echo -e "${YELLOW}Skipping on $CURRENT_PLATFORM...${NC}"
    echo ""
fi

# If no specific platform, test the current one
if [ "$WINDOWS" = false ] && [ "$ALL" = false ]; then
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  Testing Build for $CURRENT_PLATFORM${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    cd generation_two
    python build.py
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Build test PASSED!${NC}"
    else
        echo ""
        echo -e "${RED}✗ Build test FAILED!${NC}"
        exit 1
    fi
    cd ..
    echo ""
fi

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Build Test Complete!${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo -e "${YELLOW}If all tests passed, you can:${NC}"
echo "  1. Commit the changes"
echo "  2. Push to GitHub"
echo "  3. Create a release tag"
echo ""
