@echo off
setlocal EnableDelayedExpansion

:: 1. Setup your platform
echo Setting up your platform...
call :check_installation python "Python 3.10 or later"
call :check_installation pip "Pip"
call :install_if_missing git "Git" "winget install --id Git.Git -e --source winget"
call :install_if_missing ffmpeg "FFMPEG" "winget install --id Gyan.FFmpeg -e --source winget"

:: Visual Studio 2022 Runtimes
echo Installing Visual Studio 2022 Runtimes...
winget install --id Microsoft.VC++2015-2022Redist-x64 -e --source winget

:: 2. Clone Repository
call :clone_repository "https://github.com/iVideoGameBoss/iRoopDeepFaceCam.git" "iRoopDeepFaceCam"

:: 3. Download Models
echo Downloading models...
if not exist models mkdir models
curl -L -o models\GFPGANv1.4.pth https://huggingface.co/ivideogameboss/iroopdeepfacecam/resolve/main/GFPGANv1.4.pth
curl -L -o models\inswapper_128_fp16.onnx https://huggingface.co/ivideogameboss/iroopdeepfacecam/resolve/main/inswapper_128_fp16.onnx

:: 4. Install dependencies
echo Creating a virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo Installing required Python packages...
pip install --upgrade pip
pip install -r requirements.txt
echo Setup complete. You can now run the application.

:menu
:: GPU Acceleration Options
echo.
echo Choose the GPU Acceleration Option if applicable:
echo 1. CUDA (Nvidia)
echo 2. CoreML (Apple Silicon)
echo 3. CoreML (Apple Legacy)
echo 4. DirectML (Windows)
echo 5. OpenVINO (Intel)
echo 6. None
set /p choice="Enter your choice (1-6): "

set "exec_provider="
call :set_execution_provider %choice%

:end_choice
echo.
echo GPU Acceleration setup complete.
echo Selected provider: !exec_provider!
echo.

:: Run the application
if defined exec_provider (
    echo Running the application with !exec_provider! execution provider...
    python run.py --execution-provider !exec_provider!
) else (
    echo Running the application...
    python run.py
)

:: Deactivate the virtual environment
call venv\Scripts\deactivate.bat

echo.
echo Script execution completed.
pause
exit /b

:check_installation
where %1 >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo %2 is not installed. Please install %2.
    pause
    exit /b
)

:install_if_missing
where %1 >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo %2 is not installed. Installing %2...
    %3
)

:clone_repository
if exist %2 (
    echo %2 directory already exists.
    set /p overwrite="Do you want to overwrite? (Y/N): "
    if /i "%overwrite%"=="Y" (
        rmdir /s /q %2
        git clone %1
    ) else (
        echo Skipping clone, using existing directory.
    )
) else (
    git clone %1
)

:set_execution_provider
if "%1"=="1" (
    call :install_onnxruntime "onnxruntime-gpu" "1.16.3" "cuda"
) else if "%1"=="2" (
    call :install_onnxruntime "onnxruntime-silicon" "1.13.1" "coreml"
) else if "%1"=="3" (
    call :install_onnxruntime "onnxruntime-coreml" "1.13.1" "coreml"
) else if "%1"=="4" (
    call :install_onnxruntime "onnxruntime-directml" "1.15.1" "directml"
) else if "%1"=="5" (
    call :install_onnxruntime "onnxruntime-openvino" "1.15.0" "openvino"
) else if "%1"=="6" (
    echo Skipping GPU acceleration setup.
    set "exec_provider=none"
) else (
    echo Invalid choice. Please try again.
    goto menu
)

:install_onnxruntime
echo Installing %1 dependencies...
pip uninstall -y onnxruntime %1
pip install %1==%2
set "exec_provider=%3"
goto end_choice
