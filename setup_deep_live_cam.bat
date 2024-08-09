@echo off
setlocal EnableDelayedExpansion

:: 1. Setup your platform
echo Setting up your platform...

:: Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed. Please install Python 3.10 or later.
    pause
    exit /b
)

:: Pip
where pip >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Pip is not installed. Please install Pip.
    pause
    exit /b
)

:: Git
where git >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Git is not installed. Installing Git...
    winget install --id Git.Git -e --source winget
)

:: FFMPEG
where ffmpeg >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo FFMPEG is not installed. Installing FFMPEG...
    winget install --id Gyan.FFmpeg -e --source winget
)

:: Visual Studio 2022 Runtimes
echo Installing Visual Studio 2022 Runtimes...
winget install --id Microsoft.VC++2015-2022Redist-x64 -e --source winget

:: 2. Clone Repository
echo Cloning the repository...
git clone https://github.com/hacksider/Deep-Live-Cam.git
cd Deep-Live-Cam

:: 3. Download Models
echo Downloading models...
mkdir models
curl -L -o models/GFPGANv1.4.pth https://path.to.model/GFPGANv1.4.pth
curl -L -o models/inswapper_128_fp16.onnx https://path.to.model/inswapper_128_fp16.onnx

:: 4. Install dependencies
echo Creating a virtual environment...
python -m venv venv
call venv\Scripts\activate

echo Installing required Python packages...
pip install --upgrade pip
pip install -r requirements.txt

echo Setup complete. You can now run the application.

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

if "%choice%"=="1" (
    echo Installing CUDA dependencies...
    pip uninstall -y onnxruntime onnxruntime-gpu
    pip install onnxruntime-gpu==1.16.3
    set exec_provider="cuda"
) else if "%choice%"=="2" (
    echo Installing CoreML (Apple Silicon) dependencies...
    pip uninstall -y onnxruntime onnxruntime-silicon
    pip install onnxruntime-silicon==1.13.1
    set exec_provider="coreml"
) else if "%choice%"=="3" (
    echo Installing CoreML (Apple Legacy) dependencies...
    pip uninstall -y onnxruntime onnxruntime-coreml
    pip install onnxruntime-coreml==1.13.1
    set exec_provider="coreml"
) else if "%choice%"=="4" (
    echo Installing DirectML dependencies...
    pip uninstall -y onnxruntime onnxruntime-directml
    pip install onnxruntime-directml==1.15.1
    set exec_provider="directml"
) else if "%choice%"=="5" (
    echo Installing OpenVINO dependencies...
    pip uninstall -y onnxruntime onnxruntime-openvino
    pip install onnxruntime-openvino==1.15.0
    set exec_provider="openvino"
) else (
    echo Skipping GPU acceleration setup.
)

:: Run the application
if defined exec_provider (
    echo Running the application with %exec_provider% execution provider...
    python run.py --execution-provider %exec_provider%
) else (
    echo Running the application...
    python run.py
)

pause
