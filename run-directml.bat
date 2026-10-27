@echo off
set NO_ALBUMENTATIONS_UPDATE=1
if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" "%~dp0run.py" --execution-provider dml %*
) else (
    python "%~dp0run.py" --execution-provider dml %*
)
