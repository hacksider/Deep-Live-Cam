# Deep-Live-Cam justfile

set windows-shell := ["bash", "-cu"]

python_version := "3.10"
venv := ".venv"
models_dir := "models"

# Show available recipes
default:
    @just --list

# Full setup: install dependencies and download models
setup: install models

# Install Python dependencies using uv
install:
    uv venv --python {{ python_version }} {{ venv }}
    uv pip install --python {{ venv }}/Scripts/python.exe -r requirements.txt

# Download required models
models:
    #!/usr/bin/env bash
    mkdir -p {{ models_dir }}
    if [ ! -f "{{ models_dir }}/inswapper_128_fp16.onnx" ]; then
        echo "Downloading inswapper_128_fp16.onnx..."
        curl -L -o "{{ models_dir }}/inswapper_128_fp16.onnx" \
            "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx"
    else
        echo "inswapper_128_fp16.onnx already exists, skipping."
    fi
    if [ ! -f "{{ models_dir }}/GFPGANv1.4.pth" ]; then
        echo "Downloading GFPGANv1.4.pth..."
        curl -L -o "{{ models_dir }}/GFPGANv1.4.pth" \
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
    else
        echo "GFPGANv1.4.pth already exists, skipping."
    fi

# Run with CUDA GPU acceleration (default)
start:
    {{ venv }}/Scripts/python.exe run.py --execution-provider cuda

# Run with CPU only
start-cpu:
    {{ venv }}/Scripts/python.exe run.py

# Clean up virtual environment
clean:
    rm -rf {{ venv }}
