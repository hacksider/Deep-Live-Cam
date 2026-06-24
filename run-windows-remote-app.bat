@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" ".\run_windows_remote_app.py"
) else (
  python ".\run_windows_remote_app.py"
)
