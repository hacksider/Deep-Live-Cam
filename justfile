# Deep-Live-Cam justfile — run `just` or `just help` to see recipes

set positional-arguments
set windows-shell := ["sh", "-cu"]

models_dir := "models"
default_provider := if os() == "macos" { "coreml" } else if os() == "windows" { "cuda" } else { "cuda" }

# Tcl/Tk library path for standalone Python builds (needed for tkinter)
# Use python3 on Unix, python on Windows; also search 'tcl' subdir (Windows installer layout)
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


# Convert ONNX swap model to CoreML .mlpackage for native ANE dispatch (macOS only)
[group: "setup"]
convert-coreml:
    uv run scripts/convert_to_coreml.py

# Benchmark MLX inference vs ONNX Runtime CoreML EP (macOS ARM only)
[group: "setup"]
benchmark-mlx *args:
    uv run scripts/benchmark_mlx.py {{ args }}

# Check NumPy BLAS configuration (useful on macOS ARM to verify Accelerate is in use)
[group: "setup"]
check-blas:
    uv run python -c "from modules.blas_check import check_apple_silicon_blas, log_blas_config; import logging; logging.basicConfig(level=logging.INFO); log_blas_config(); check_apple_silicon_blas()"

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

# Run with virtual camera output enabled
[group: "run"]
start-virtualcam:
    uv run run.py --execution-provider {{ default_provider }} --virtual-cam

##########
# Setup (Virtual Camera)
##########

# Install pyvirtualcam optional dependency and show platform setup instructions
[group: "setup"]
setup-virtualcam:
    #!/usr/bin/env bash
    set -euo pipefail
    uv sync --extra virtualcam
    echo ""
    echo "pyvirtualcam installed successfully."
    echo ""
    case "$(uname -s)" in
        Darwin)
            echo "macOS setup:"
            echo "  1. Install OBS Studio 30+"
            echo "  2. Open OBS → Tools → Start Virtual Camera → Stop → Close OBS"
            echo "  3. The 'OBS Virtual Camera' device is now available system-wide"
            ;;
        Linux)
            echo "Linux setup:"
            echo "  sudo apt install v4l2loopback-dkms"
            echo "  sudo modprobe v4l2loopback devices=1"
            ;;
        *)
            echo "Windows setup:"
            echo "  Install OBS Studio 26+ (provides the virtual camera backend)"
            ;;
    esac

##########
# Debug
##########

# Run with Python fault handler — prints Python stack on SIGSEGV/SIGFPE
[group: "debug"]
start-faulthandler:
    PYTHONFAULTHANDLER=1 uv run run.py --execution-provider {{ default_provider }}

# Run under lldb — attach debugger, type `run` then `bt` after crash
[group: "debug"]
start-lldb:
    lldb -- uv run run.py --execution-provider {{ default_provider }}

# Show most recent macOS crash report for this process (parsed summary)
[group: "debug"]
crash-report:
    #!/usr/bin/env python3
    import glob, json, os, sys
    reports = sorted(glob.glob(os.path.expanduser("~/Library/Logs/DiagnosticReports/python3.10-*.ips")), reverse=True)
    if not reports:
        print("No crash reports found"); sys.exit(1)
    report = reports[0]
    print(f"=== {report} ===")
    with open(report) as f:
        content = f.read()
    _, body = content.split("\n", 1)
    data = json.loads(body)
    exc = data.get("exception", {})
    print(f"Exception : {exc.get('type', '?')} {exc.get('subtype', '')} (signal {exc.get('signal', '?')})")
    print(f"Codes     : {exc.get('codes', '?')}")
    for t in data.get("threads", []):
        if t.get("triggered"):
            print(f"Crashed thread {t['id']} {t.get('name', '')}:")
            for frame in t.get("frames", [])[:20]:
                print(f"  {frame.get('symbol', '???')}  +{frame.get('symbolLocation', '?')}")
            break

# List recent crash reports for this process
[group: "debug"]
crash-list:
    ls -lt ~/Library/Logs/DiagnosticReports/ | grep python3 | head -20

# Trace which dylib is loaded last before a crash (macOS only)
[group: "debug"]
start-dyld-trace:
    DYLD_PRINT_LIBRARIES=1 uv run run.py --execution-provider {{ default_provider }} 2>&1 | tail -30

##########
# Test
##########

# Run fast unit tests — after every change
[group: "test"]
test-quick *args:
    uv run pytest tests/ -x -q {{ args }}

# Run full test suite with verbose output — before commit/PR
[group: "test"]
test *args:
    uv run pytest tests/ -v {{ args }}

# Run tests with coverage report
[group: "test"]
test-cov *args:
    uv run pytest tests/ --cov=modules --cov-report=term-missing {{ args }}

# Verify NumPy BLAS configuration (Apple Accelerate on macOS ARM)
[group: "test"]
test-blas:
    uv run pytest tests/test_numpy_blas.py -v

##########
# Maintenance
##########

# Clean up virtual environment and lock file
[group: "maintenance"]
[confirm("Remove virtual environment?")]
clean:
    rm -rf .venv uv.lock
