# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep-Live-Cam is a real-time face swap and video deepfake application (v2.0.3c). It performs face swapping on live webcam feeds, images, and videos using InsightFace for detection and ONNX models for inference.

## Setup

- Python 3.10 required (pinned via `.python-version` and `pyproject.toml requires-python = "==3.10.*"`)
- `uv` for package management and running (`uv sync`, `uv run`)
- `mise` for Python version management (provides Python with working Tcl/Tk)
- `just` as task runner — run `just` to see all recipes
- Requires `ffmpeg` on PATH
- Models in `models/`: `GFPGANv1.4.pth` and `inswapper_128_fp16.onnx`
- Platform-specific ONNX runtimes: `onnxruntime-silicon` (macOS ARM), `onnxruntime-gpu` (CUDA)

```bash
just setup    # uv sync + model download
# or manually:
uv sync       # Install dependencies from pyproject.toml
```

### Tkinter / Tcl/Tk

Standalone Python builds (python-build-standalone, used by mise and uv) ship Tcl/Tk but hardcode build-time paths for `init.tcl`. The justfile auto-detects and exports `TCL_LIBRARY`/`TK_LIBRARY`. When running outside of just, ensure mise is activated or set these env vars manually.

## Running the Application

```bash
# Using just (recommended — handles Tcl/Tk env vars)
just start              # Platform-default GPU (coreml on macOS, cuda on Linux/Windows)
just start-cpu          # CPU only
just start-with rocm    # Specific provider

# Using uv directly (requires TCL/TK env vars)
uv run run.py --execution-provider coreml

# Headless mode (no GUI, no Tcl/Tk needed)
uv run run.py -s source.jpg -t target.mp4 -o output.mp4

# With options
uv run run.py --many-faces --mouth-mask --frame-processor face_swapper face_enhancer
```

## Architecture

### Entry Point

`run.py` → `modules/core.py:run()` — parses CLI args, runs pre-checks (Python version, ffmpeg), then either starts headless processing or launches the CustomTkinter GUI.

### Global State

`modules/globals.py` — mutable module-level variables hold all runtime configuration (paths, processing flags, execution providers). Set by CLI arg parsing in `core.py` and toggled by the UI.

### Frame Processor Plugin System

`modules/processors/frame/core.py` dynamically loads processor modules from `modules/processors/frame/`:

- **`face_swapper.py`** — core face swap using InsightFace + inswapper ONNX model
- **`face_enhancer.py`** — GFPGAN-based face restoration/enhancement
- **`face_masking.py`** — mouth mask region handling

Each processor implements: `pre_check()`, `pre_start()`, `process_frame()`, `process_image()`, `process_video()`.

Processing uses `ThreadPoolExecutor` for parallel frame processing with configurable thread count.

### UI

`modules/ui.py` — CustomTkinter GUI with live webcam preview. Uses `modules/video_capture.py` for camera access and `modules/face_analyser.py` for face detection/analysis (InsightFace). Supports i18n via `modules/gettext.py` with translations in `locales/`.

### Key Processing Flow (Video)

1. Extract frames from target video (ffmpeg via `modules/utilities.py`)
2. Run frame processors in pipeline on each frame (parallel via ThreadPoolExecutor)
3. Reassemble frames into video, restore audio

## Key Config Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Dependencies, build config, uv index/sources |
| `.python-version` | Pins Python 3.10 for uv/mise |
| `mise.toml` | Python version + auto-venv creation |
| `justfile` | Task runner recipes (setup, run, clean) |
| `uv.lock` | Locked dependency versions |

## Contributing

- Push to `premain` branch first, not `main` — changes merge to `main` after testing
- `experimental` branch for large/disruptive changes
- Test realtime faceswap (with/without enhancer), map faces, camera listing
- Verify no FPS drops, GPU overloading (15min minimum), or boot time regressions
