#!/bin/bash

# Deep-Live-Cam macOS Launcher (Apple Silicon Optimized)

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./install-mac.sh first."
    exit 1
fi

# Activate Virtual Environment
source venv/bin/activate

# Set environment variables for optimization
export OMP_NUM_THREADS=1
# export PYTORCH_ENABLE_MPS_FALLBACK=1 # Optional, if needed for some ops

echo "🚀 Starting Deep-Live-Cam on macOS..."
echo "ℹ️  Mode: CoreML Execution Provider (Optimized)"

# Run with python3 (which is the venv python)
# We use --execution-provider coreml explicitly
# We can also add --max-memory 4 as suggested by the code for Darwin, though the code does it automatically.
python run.py --execution-provider coreml "$@"
