#!/usr/bin/env bash

VENV_DIR=".venv"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' not found."
    echo "Please run ./setup_mac.sh first to create the environment and install dependencies."
    exit 1
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Starting the application with CPU execution provider..."
# Passes all arguments passed to this script (e.g., --source, --target) to run.py
python3 run.py --execution-provider cpu "$@"

# Deactivate after script finishes (optional, as shell context closes)
# deactivate
