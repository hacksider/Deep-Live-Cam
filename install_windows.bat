@echo off
REM Deep-Live-Cam Windows Automated Setup

REM 1. Create virtual environment
python -m venv venv
if errorlevel 1 (
    echo Failed to create virtual environment. Ensure Python 3.10+ is installed and in PATH.
    exit /b 1
)

REM 2. Activate virtual environment
call venv\Scripts\activate
if errorlevel 1 (
    echo Failed to activate virtual environment.
    exit /b 1
)

REM 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies.
    exit /b 1
)

REM 4. Download models (manual step if not present)
echo Downloading models (if not already in models/)...
if not exist models\GFPGANv1.4.pth (
    powershell -Command "Invoke-WebRequest -Uri https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth -OutFile models\GFPGANv1.4.pth"
)
if not exist models\inswapper_128_fp16.onnx (
    powershell -Command "Invoke-WebRequest -Uri https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx -OutFile models\inswapper_128_fp16.onnx"
)

REM 5. Run the app
python run.py
