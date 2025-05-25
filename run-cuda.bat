@echo off
set VENV_DIR=.venv

:: Check if virtual environment exists
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Virtual environment '%VENV_DIR%' not found.
    echo Please run setup_windows.bat first.
    pause
    exit /b 1
)

echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo Starting the application with CUDA execution provider...
python run.py --execution-provider cuda %*
