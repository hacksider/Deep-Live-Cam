#!/bin/bash

# Deep-Live-Cam macOS Installation Script (Apple Silicon Optimized)

set -e  # Exit on error

echo "🍎 Setting up Deep-Live-Cam for macOS (Apple Silicon)..."

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install Homebrew first: https://brew.sh/"
    exit 1
fi

# Check for Python 3.10
if ! command -v python3.10 &> /dev/null; then
    echo "⚠️ Python 3.10 not found. Installing via Homebrew..."
    brew install python@3.10
    brew install python-tk@3.10
else
    echo "✅ Python 3.10 found."
fi

# Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3.10 -m venv venv
else
    echo "✅ Virtual environment already exists."
fi

# Activate Virtual Environment
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install Dependencies
echo "📥 Installing dependencies from requirements.txt..."
# Note: requirements.txt in the repo already has platform markers, but we want to ensure clean install
pip install -r requirements.txt

# Post-Install Fixes for macOS
echo "🔧 Applying macOS specific fixes..."

# Fix 1: Ensure onnxruntime-silicon is used (requirements.txt has it, but let's double check)
if pip show onnxruntime &> /dev/null; then
    echo "⚠️ Removing standard onnxruntime to avoid conflicts..."
    pip uninstall -y onnxruntime
fi

# Fix 2: BasicSR and GFPGAN are git deps, ensured by requirements.txt

echo "✅ Installation complete!"
echo "🚀 To run the application, use: ./run-mac.sh"
