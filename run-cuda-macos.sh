#!/bin/zsh
# run-cuda-macos.sh - Run Deep-Live-Cam with CUDA (Nvidia GPU) on macOS
source venv/bin/activate
python run.py --execution-provider cuda
