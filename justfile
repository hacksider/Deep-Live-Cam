# Deep-Live-Cam justfile — run `just` or `just help` to see recipes

set positional-arguments
set windows-shell := ["bash", "-cu"]

models_dir := "models"
default_provider := if os() == "macos" { "coreml" } else if os() == "windows" { "cuda" } else { "cuda" }

# Tcl/Tk library path for standalone Python builds (needed for tkinter)
export TCL_LIBRARY := `python3 -c "import os, sys, glob; c=glob.glob(os.path.join(sys.base_prefix,'lib','tcl*','init.tcl')); print(os.path.dirname(c[0]) if c else '')" 2>/dev/null || echo ""`
export TK_LIBRARY := `python3 -c "import os, sys, glob; c=glob.glob(os.path.join(sys.base_prefix,'lib','tk*')); print(c[0] if c else '')" 2>/dev/null || echo ""`

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
    if [ ! -f "{{ models_dir }}/GFPGANv1.4.pth" ]; then
        echo "Downloading GFPGANv1.4.pth..."
        curl -L -o "{{ models_dir }}/GFPGANv1.4.pth" \
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
    else
        echo "GFPGANv1.4.pth already exists, skipping."
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

# Run tests
[group: "debug"]
test *args:
    uv run pytest {{ args }}

##########
# Maintenance
##########

# Clean up virtual environment and lock file
[group: "maintenance"]
[confirm("Remove virtual environment?")]
clean:
    rm -rf .venv uv.lock
