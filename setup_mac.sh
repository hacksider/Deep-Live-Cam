#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting macOS setup..."

# 1. Check for Python 3
echo "Checking for Python 3..."
if ! command -v python3 &> /dev/null
then
    echo "Python 3 could not be found. Please install Python 3."
    echo "You can often install it using Homebrew: brew install python"
    exit 1
fi

# 2. Check Python version (>= 3.9)
echo "Checking Python 3 version..."
python3 -c 'import sys; exit(0) if sys.version_info >= (3,9) else exit(1)'
if [ $? -ne 0 ]; then
    echo "Python 3.9 or higher is required."
    echo "Your version is: $(python3 --version)"
    echo "Please upgrade your Python version. Consider using pyenv or Homebrew to manage Python versions."
    exit 1
fi
echo "Python 3.9+ found: $(python3 --version)"

# 3. Check for ffmpeg
echo "Checking for ffmpeg..."
if ! command -v ffmpeg &> /dev/null
then
    echo "WARNING: ffmpeg could not be found. This program requires ffmpeg for video processing."
    echo "You can install it using Homebrew: brew install ffmpeg"
    echo "Continuing with setup, but video processing might fail later."
else
    echo "ffmpeg found: $(ffmpeg -version | head -n 1)"
fi

# 4. Define virtual environment directory
VENV_DIR=".venv"

# 5. Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists. Skipping creation."
else
    echo "Creating virtual environment in '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
fi

# 6. Activate virtual environment (for this script's session)
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# 7. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 8. Install requirements
echo "Installing requirements from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "ERROR: requirements.txt not found. Cannot install dependencies."
    # Deactivate on error if desired, or leave active for user to debug
    # deactivate
    exit 1
fi

echo ""
echo "Setup complete!"
echo ""
echo "To activate the virtual environment in your terminal, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "After activating, you can run the application using:"
echo "  python3 run.py [arguments]"
echo "Or use one of the run_mac_*.sh scripts (e.g., ./run_mac_cpu.sh)."
echo ""

# Deactivate at the end of the script's execution (optional, as script session ends)
# deactivate
