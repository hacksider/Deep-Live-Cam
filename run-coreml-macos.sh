#!/bin/zsh
# run-coreml-macos.sh - Run Deep-Live-Cam with CoreML (Apple Silicon) on macOS
source venv/bin/activate
python3.10 run.py --execution-provider coreml
