@echo off
REM ============================================================================
REM BeatBox DAW - Audio Engine Build Script (Windows)
REM ============================================================================
REM This script compiles the Python audio engine into a standalone executable
REM using PyInstaller, then copies it to the Tauri binaries directory.
REM
REM Prerequisites:
REM   - Python 3.10+ installed and in PATH
REM   - PyInstaller installed: pip install pyinstaller
REM   - All engine dependencies installed
REM
REM Usage:
REM   scripts\build-engine.bat
REM
REM Output:
REM   src-tauri\binaries\audio-engine-x86_64-pc-windows-msvc.exe
REM ============================================================================

setlocal enabledelayedexpansion

REM Store the script's directory (project root)
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
cd /d "%PROJECT_ROOT%"

echo.
echo ============================================================
echo   BeatBox DAW - Audio Engine Build Script
echo ============================================================
echo.

REM ============================================================================
REM Step 1: Verify we're in the correct directory
REM ============================================================================
if not exist "engine\main.py" (
    echo ERROR: Cannot find engine\main.py
    echo Please run this script from the project root directory.
    exit /b 1
)

if not exist "engine\audio-engine.spec" (
    echo ERROR: Cannot find engine\audio-engine.spec
    echo Please create the PyInstaller spec file first.
    exit /b 1
)

REM ============================================================================
REM Step 2: Check for Python/PyInstaller
REM ============================================================================
echo [1/5] Checking Python environment...

REM Try to activate venv if it exists
if exist "engine\.venv\Scripts\activate.bat" (
    echo   Found virtual environment in engine\.venv
    call "engine\.venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    echo   Found virtual environment in .venv
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    echo   Found virtual environment in venv
    call "venv\Scripts\activate.bat"
) else (
    echo   No virtual environment found, using system Python
)

REM Verify Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    exit /b 1
)

REM Verify PyInstaller is available
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo ERROR: PyInstaller not installed
    echo Please install it with: pip install pyinstaller
    exit /b 1
)

echo   Python and PyInstaller verified

REM ============================================================================
REM Step 3: Clean previous build artifacts
REM ============================================================================
echo [2/5] Cleaning previous build artifacts...

if exist "engine\dist" (
    rmdir /s /q "engine\dist"
    echo   Removed engine\dist
)
if exist "engine\build" (
    rmdir /s /q "engine\build"
    echo   Removed engine\build
)

REM ============================================================================
REM Step 4: Run PyInstaller
REM ============================================================================
echo [3/5] Running PyInstaller...
echo   This may take several minutes...
echo.

cd engine

REM Run PyInstaller with the spec file
pyinstaller audio-engine.spec --noconfirm

if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller build failed
    cd ..
    exit /b 1
)

cd ..

REM Verify the executable was created
if not exist "engine\dist\audio-engine.exe" (
    echo ERROR: Build succeeded but audio-engine.exe not found
    exit /b 1
)

echo.
echo   Build successful: engine\dist\audio-engine.exe
echo.

REM ============================================================================
REM Step 5: Copy to Tauri binaries directory
REM ============================================================================
echo [4/5] Copying to Tauri binaries directory...

REM Create binaries directory if it doesn't exist
if not exist "src-tauri\binaries" (
    mkdir "src-tauri\binaries"
    echo   Created src-tauri\binaries directory
)

REM Copy with correct Tauri sidecar naming convention
REM Format: <binary-name>-<target-triple>.exe
REM For Windows x64: audio-engine-x86_64-pc-windows-msvc.exe
copy /y "engine\dist\audio-engine.exe" "src-tauri\binaries\audio-engine-x86_64-pc-windows-msvc.exe" >nul

if errorlevel 1 (
    echo ERROR: Failed to copy executable to Tauri binaries
    exit /b 1
)

echo   Copied to: src-tauri\binaries\audio-engine-x86_64-pc-windows-msvc.exe

REM ============================================================================
REM Step 6: Report results
REM ============================================================================
echo [5/5] Build complete!
echo.
echo ============================================================
echo   BUILD SUMMARY
echo ============================================================
echo.
echo   Source:      engine\main.py
echo   Spec file:   engine\audio-engine.spec
echo   Output:      engine\dist\audio-engine.exe
echo   Tauri copy:  src-tauri\binaries\audio-engine-x86_64-pc-windows-msvc.exe
echo.

REM Get file size
for %%A in ("engine\dist\audio-engine.exe") do (
    set "SIZE=%%~zA"
    set /a "SIZE_MB=!SIZE!/1048576"
    echo   Executable size: !SIZE_MB! MB
)

echo.
echo   Next steps:
echo   1. Test standalone: engine\dist\audio-engine.exe
echo      (Should start WebSocket server on port 8765)
echo   2. Update src-tauri\tauri.conf.json to include externalBin
echo   3. Run: npm run tauri build
echo.
echo ============================================================

endlocal
exit /b 0
