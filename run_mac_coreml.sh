#!/usr/bin/env bash

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' not found."
    echo "Please run ./setup_mac.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"
echo "Starting the application with CoreML execution provider..."
python3 run.py --execution-provider coreml "$@"
