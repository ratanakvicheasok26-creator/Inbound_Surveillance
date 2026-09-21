@echo off
title Inbound Surveillance - Diagnostic Console
echo ===================================================
echo   Inbound Surveillance - Live Diagnostic Console
echo ===================================================
echo.
echo Starting Inbound Surveillance with live console logging...
echo If the application window fails to appear, all diagnostics
echo and error tracebacks will remain visible in this window.
echo.

set "EXE_PATH=Inbound Surveillance.exe"
if not exist "%EXE_PATH%" (
    for %%f in (*Surveillance*.exe) do (
        set "EXE_PATH=%%f"
        goto :found
    )
)

:found
if not exist "%EXE_PATH%" (
    echo [ERROR] Could not find Inbound Surveillance.exe in this folder.
    echo Please place run-debug.bat in the same folder as Inbound Surveillance.exe.
    echo.
    pause
    exit /b 1
)

echo Found executable: %EXE_PATH%
echo Running "%EXE_PATH%" --console ...
echo.
"%EXE_PATH%" --console
set EXIT_CODE=%errorlevel%

echo.
echo ===================================================
echo Inbound Surveillance process finished (Exit Code: %EXIT_CODE%)
echo ===================================================
echo If an error occurred, check:
echo   1. The log file: inbound-surveillance.log in this folder
echo   2. The Desktop report: INBOUND_CRASH_REPORT.txt on your Desktop
echo.
pause
