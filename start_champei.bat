@echo off
title Champei Spa Intelligence
echo Starting Champei Spa Intelligence System...

cd /d "%~dp0"

:: Activate virtual environment if present
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

:: Run Champei daemon
cd edge
python run_champei.py

pause
