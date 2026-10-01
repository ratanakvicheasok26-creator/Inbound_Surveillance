@echo off
setlocal enabledelayedexpansion
title Champei Spa Intelligence - Automatic Setup
color 0A

echo ====================================================================
echo      CHAMPEI SPA INTELLIGENCE SYSTEM - 1-CLICK AUTOMATIC SETUP
echo ====================================================================
echo.

cd /d "%~dp0"

:: 1. Check for Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    py --version >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        color 0C
        echo [ERROR] Python is not installed on this computer!
        echo.
        echo Please download and install Python 3.10, 3.11, or 3.12:
        echo 1. Go to: https://www.python.org/downloads/
        echo 2. IMPORTANT: Check the box "Add Python to PATH" during installation.
        echo 3. Run this INSTALL_WINDOWS.bat again after installing Python.
        echo.
        pause
        exit /b 1
    ) else (
        set "PY_CMD=py"
    )
) else (
    set "PY_CMD=python"
)

echo [*] Python detected:
%PY_CMD% --version
echo.

:: 2. Create Virtual Environment
echo [*] Setting up isolated application environment...
if not exist "edge\.venv" (
    %PY_CMD% -m venv edge\.venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)
echo [OK] Virtual environment ready.
echo.

:: 3. Install Requirements
echo [*] Installing AI and vision components (this takes 1-2 minutes)...
call edge\.venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
python -m pip install -r edge\requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install requirements. Please check your internet connection.
    pause
    exit /b 1
)
echo [OK] All dependencies successfully installed.
echo.

:: 4. Verify Installation
echo [*] Running diagnostic check...
python edge\verify_customer_install.py
echo.

:: 5. Create Desktop Shortcut
echo [*] Creating Desktop Shortcut...
set "TARGET_BAT=%~dp0start_champei.bat"
set "SHORTCUT_PATH=%USERPROFILE%\Desktop\Champei Spa Intelligence.lnk"
set "ICON_PATH=%~dp0favicon.svg"

powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT_PATH%'); $s.TargetPath = '%TARGET_BAT%'; $s.WorkingDirectory = '%~dp0'; $s.Save()"

if exist "%SHORTCUT_PATH%" (
    echo [OK] Shortcut created on Desktop: "Champei Spa Intelligence"
) else (
    echo [INFO] You can launch the system using start_champei.bat in this folder.
)

echo.
echo ====================================================================
echo   SUCCESS! Installation Complete.
echo   You can now start the system by double-clicking:
echo   - "Champei Spa Intelligence" on your Desktop, OR
echo   - "start_champei.bat" in this folder.
echo ====================================================================
echo.
pause
