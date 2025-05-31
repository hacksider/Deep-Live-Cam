@echo off
echo Starting Windows setup...

:: 1. Check for Python
echo Checking for Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python could not be found in your PATH.
    echo Please install Python 3 (3.10 or higher recommended) and ensure it's added to your PATH.
    echo You can download Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Optional: Check Python version (e.g., >= 3.9 or >=3.10).
:: This is a bit more complex in pure batch. For now, rely on user having a modern Python 3.
:: The README will recommend 3.10.
:: If we reach here, Python is found.
echo Python was found. Attempting to display version:
for /f "delims=" %%i in ('python --version 2^>^&1') do echo %%i

:: 2. Check for ffmpeg (informational)
echo Checking for ffmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo WARNING: ffmpeg could not be found in your PATH. This program requires ffmpeg for video processing.
    echo Please download ffmpeg from https://ffmpeg.org/download.html and add it to your system's PATH.
    echo (The README.md contains a link for a potentially easier ffmpeg install method using a PowerShell command)
    echo Continuing with setup, but video processing might fail later.
    pause
) else (
    echo ffmpeg found.
)

:: 3. Define virtual environment directory
set VENV_DIR=.venv

:: 4. Create virtual environment
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Virtual environment '%VENV_DIR%' already exists. Skipping creation.
) else (
    echo Creating virtual environment in '%VENV_DIR%'...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create virtual environment. Please check your Python installation.
        pause
        exit /b 1
    )
)

:: 5. Activate virtual environment (for this script's session)
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

:: 6. Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: 7. Install requirements
echo Installing requirements from requirements.txt...
if exist "requirements.txt" (
    python -m pip install -r requirements.txt
) else (
    echo ERROR: requirements.txt not found. Cannot install dependencies.
    pause
    exit /b 1
)

echo.
echo Setup complete!
echo.
echo To activate the virtual environment in your command prompt, run:
echo   %VENV_DIR%\Scripts\activate.bat
echo.
echo After activating, you can run the application using:
echo   python run.py [arguments]
echo Or use one of the run-*.bat scripts (e.g., run-cuda.bat, run_windows.bat).
echo.
pause
exit /b 0
