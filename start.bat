@echo off
rem OmniBrain one-click launcher: first run builds .venv, every run starts the app and opens the UI.
setlocal
cd /d "%~dp0"
title OmniBrain

if not exist ".venv\Scripts\python.exe" (
  echo First run: creating the virtual environment...
  py -3 -m venv .venv 2>nul
  if errorlevel 1 python -m venv .venv
  if not exist ".venv\Scripts\python.exe" (
    echo Python 3.11 or newer is required. Install it from python.org, then run this again.
    pause
    exit /b 1
  )
  echo Installing dependencies...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Dependency install failed. See the messages above.
    pause
    exit /b 1
  )
)

".venv\Scripts\python.exe" run.py serve --open %*
if errorlevel 1 (
  echo.
  echo OmniBrain stopped with an error. See above, or data\omnibrain.log.
  pause
)
endlocal