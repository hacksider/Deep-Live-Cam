# Deep-Live-Cam justfile — run `just` or `just help` to see recipes

set positional-arguments
set windows-shell := ["sh", "-cu"]

models_dir := "models"
default_provider := if os() == "macos" { "coreml" } else if os() == "windows" { "cuda" } else { "cuda" }

# Tcl/Tk library path for standalone Python builds (needed for tkinter)
export TCL_LIBRARY := `(python3 -c "import os, sys, glob; hits=glob.glob(os.path.join(sys.base_prefix,'lib','tcl*','init.tcl'))+glob.glob(os.path.join(sys.base_prefix,'tcl','tcl*','init.tcl')); print(os.path.dirname(hits[0]) if hits else '')" 2>/dev/null || python -c "import os, sys, glob; hits=glob.glob(os.path.join(sys.base_prefix,'lib','tcl*','init.tcl'))+glob.glob(os.path.join(sys.base_prefix,'tcl','tcl*','init.tcl')); print(os.path.dirname(hits[0]) if hits else '')" 2>/dev/null || echo "")`
export TK_LIBRARY := `(python3 -c "import os, sys, glob; hits=glob.glob(os.path.join(sys.base_prefix,'lib','tk*'))+glob.glob(os.path.join(sys.base_prefix,'tcl','tk*')); print(hits[0] if hits else '')" 2>/dev/null || python -c "import os, sys, glob; hits=glob.glob(os.path.join(sys.base_prefix,'lib','tk*'))+glob.glob(os.path.join(sys.base_prefix,'tcl','tk*')); print(hits[0] if hits else '')" 2>/dev/null || echo "")`

# Show available recipes
default:
    @just --list

##########
# Setup
##########

# Full setup: install dependencies and download models
[group: "setup"]
setup: install models

# Install Python dependencies using uv
[group: "setup"]
install:
    uv sync

# Download required models
[group: "setup"]
models:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p {{ models_dir }}
    if [ ! -f "{{ models_dir }}/inswapper_128_fp16.onnx" ]; then
        echo "Downloading inswapper_128_fp16.onnx..."
        curl -L -o "{{ models_dir }}/inswapper_128_fp16.onnx" \
            "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx"
    else
        echo "inswapper_128_fp16.onnx already exists, skipping."
    fi
    if [ ! -f "{{ models_dir }}/gfpgan-1024.onnx" ]; then
        echo "Downloading gfpgan-1024.onnx..."
        curl -L -o "{{ models_dir }}/gfpgan-1024.onnx" \
            "https://huggingface.co/hacksider/deep-live-cam/resolve/main/gfpgan-1024.onnx"
    else
        echo "gfpgan-1024.onnx already exists, skipping."
    fi

##########
# Run
##########

# Run with platform-default GPU acceleration (coreml on macOS, cuda on Linux/Windows)
[group: "run"]
start:
    uv run run.py --execution-provider {{ default_provider }}

# Run with CPU only
[group: "run"]
start-cpu:
    uv run run.py

# Run with specific execution provider
[group: "run"]
start-with provider:
    uv run run.py --execution-provider {{ provider }}

##########
# Test
##########

# Run tests
[group: "test"]
test *args:
    uv run pytest tests/ {{ args }}

##########
# Maintenance
##########

# Clean up virtual environment and lock file
[group: "maintenance"]
[confirm("Remove virtual environment?")]
clean:
    rm -rf .venv uv.lock
