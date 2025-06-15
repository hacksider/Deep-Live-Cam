@echo off
set VENV_DIR=.venv

:: Check if virtual environment exists
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Virtual environment '%VENV_DIR%' not found.
    echo Please run setup_windows.bat first to create the environment and install dependencies.
    pause
    exit /b 1
)

echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo Starting the application with CPU execution provider...
:: Passes all arguments passed to this script to run.py
python run.py --execution-provider cpu %*

:: Optional: Deactivate after script finishes
:: call deactivate
