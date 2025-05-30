#!/bin/bash
# Deep-Live-Cam macOS Automated Setup
set -e

# 1. Ensure Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Please install Homebrew first: https://brew.sh/"
    exit 1
fi

# 2. Install Python 3.10 and tkinter
brew install python@3.10 python-tk@3.10

# 3. Create and activate virtual environment
PYTHON_BIN=$(brew --prefix python@3.10)/bin/python3.10
$PYTHON_BIN -m venv venv
source venv/bin/activate

# 4. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Download models if not present
mkdir -p models
if [ ! -f models/GFPGANv1.4.pth ]; then
    curl -L -o models/GFPGANv1.4.pth "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth"
fi
if [ ! -f models/inswapper_128_fp16.onnx ]; then
    curl -L -o models/inswapper_128_fp16.onnx "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx"
fi

# 6. Run instructions for user

echo "\nSetup complete!"
echo "To activate your environment and run Deep-Live-Cam, use one of the following commands:" 
echo ""
echo "# For CUDA (Nvidia GPU, if supported):"
echo "source venv/bin/activate && python run.py --execution-provider cuda"
echo ""
echo "# For Apple Silicon (M1/M2/M3) CoreML:"
echo "source venv/bin/activate && python3.10 run.py --execution-provider coreml"
echo ""
echo "# For CPU only:"
echo "source venv/bin/activate && python run.py"
