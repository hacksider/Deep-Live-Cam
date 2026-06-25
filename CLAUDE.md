# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Deep-Live-Cam Remote** is a downstream fork of [hacksider/Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) that adds Google Colab batch processing and a desktop remote controller app. The codebase combines a headless face-swapping engine (inherited from Deep-Live-Cam), Colab integration, a FastAPI WebSocket server, and a Windows PySide6 desktop application for remote job management via Tailscale.

**Key Workflows:**
- Local desktop processing (video/photo swapping)
- Colab batch processing (resumed via manifest)
- Desktop remote control of Colab jobs over Tailscale

## Line Endings & Code Style

**This repository enforces LF (Unix-style) line endings for all files:**
- `.editorconfig`: `end_of_line = lf` globally
- `.gitattributes`: `* text=auto eol=lf` (auto-normalizes on commit)

When editing files or running scripts, use `--eol lf` (not auto). Most editors respect `.editorconfig` automatically, but be aware if using Windows tooling.

## Development Commands

### Setup & Installation

```powershell
# Windows development setup
git clone https://github.com/djebaz/Deep-Live-Cam-Remote.git
Set-Location .\Deep-Live-Cam-Remote
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Colab notebook round-trip setup (same venv)
python scripts/py_to_ipynb.py .\google-colab\Deep_Live_Cam_Remote_Batch.py .\google-colab\Deep_Live_Cam_Remote_Batch.ipynb --eol lf
```

### Validation & Testing

```powershell
# Syntax check (run after editing colab_batch.py, colab_api.py, or windows_app/app.py)
python -m py_compile .\colab_batch.py .\colab_api.py

# Run focused tests
python -m pytest .\tests -q

# Ruff linting (enforced rules: E701, E711, E712, F401, F541)
python -m ruff check --select E701,E711,E712,F401,F541 .
```

### Running the Application

```powershell
# Desktop remote app (Windows)
.\run_windows_remote_app.ps1

# Colab batch from command line (example)
python colab_batch.py process --source-face source.png --input-dir ./videos --output-dir ./output

# Colab API server (inside Colab)
python -c "from colab_api import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=7860)"
```

### Building Standalone Desktop Executable

```powershell
# Full build (includes live webcam dependencies: cv2, numpy, pyvirtualcam)
py -3.11 -m venv .venv_build
.\.venv_build\Scripts\python.exe -m pip install -r requirements-build.txt
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_remote_app.ps1

# Lite build (smaller, no live webcam)
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_remote_app.ps1 -Lite

# Single-file executable
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_remote_app.ps1 -OneFile

# With explicit version
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_remote_app.ps1 -Version 0.1.0
```

Output is versioned under `dist/<version>/` with names like `Deep-Live-Cam-Remote-0.1.0-py3.11.exe`.

### Notebook Round-Trip Workflow

The Colab notebook is generated from a markerized Python source.

```powershell
# After editing google-colab/Deep_Live_Cam_Remote_Batch.py, rebuild the notebook
python scripts/py_to_ipynb.py `
  .\google-colab\Deep_Live_Cam_Remote_Batch.py `
  .\google-colab\Deep_Live_Cam_Remote_Batch.ipynb `
  --eol lf

# If you edited the notebook directly in Colab, export it back
python scripts/ipynb_to_py.py `
  .\google-colab\Deep_Live_Cam_Remote_Batch.ipynb `
  .\google-colab\Deep_Live_Cam_Remote_Batch.py `
  --eol lf

# Preserve cell IDs, marker lines, meta_b64, NOTEBOOK_META_B64, and markdown/raw sentinels
```

## Architecture

The codebase is organized into **four main entry points**, each serving a different workflow:

### 1. **Local Desktop Mode** (`run.py` + `modules/`)

- **Entry:** `run.py`
- **Flow:** Parse args → Initialize GPU/CUDA → Load PySide6 UI (or headless mode)
- **Processing:** Local video/photo swapping via frame processor pipeline
- **Components:**
  - `modules/core.py`: Main orchestration (process_video_in_memory, extract_frames)
  - `modules/face_analyser.py`: InsightFace integration (RetinaFace detection, embedding)
  - `modules/processors/frame/`: Pluggable processor modules (face_swapper, face_enhancer_gpen256/512, face_masking)
  - `modules/utilities.py`: FFmpeg wrapper (decode, frame extraction, encode, audio muxing)
  - `modules/ui.py`: PySide6 UI (57KB, local desktop interface)

### 2. **Colab Batch Processing** (`colab_batch.py`)

- **Entry:** `colab_batch.py process|photos|scan` CLI
- **Flow:** Config → File discovery (recursive) → Manifest check (skip if processed) → FFmpeg + frame processing → ZIP output
- **Key Features:**
  - Resumable via `.deep_live_cam_processed.json` manifest (prevents reprocessing)
  - Bounded queues decouple decode/encode to manage memory
  - Face mapping configuration support (scan + apply)
  - Batch report JSON with per-file results
- **Components:**
  - `ProcessConfig` dataclass (all processing options)
  - FFmpeg pipeline (decode, scale to max_width, cap FPS, extract frames)
  - Face analysis + processor pipeline loop
  - Manifest-based idempotency

### 3. **Colab API Server** (`colab_api.py`)

- **Entry:** `colab_api.py` (FastAPI on port 7860)
- **Flow:** REST/WebSocket endpoints → Job queue → `colab_batch.py` subprocess
- **Endpoints:**
  - `POST /process`, `/photos`, `/scan` (submit jobs)
  - `POST /cancel` (stop running job)
  - `GET /outputs`, `/preview`, `/health` (query results)
  - `WebSocket /ws/{job_id}` (live log streaming)
- **Components:**
  - `JobRequest`, `JobState` dataclasses
  - Thread-based job execution with cancel event
  - Log buffering and queue management
  - ZIP download with background cleanup

### 4. **Desktop Remote Controller** (`windows_app/app.py` + patches)

- **Entry:** `windows_app/app.py` (PySide6 QMainWindow)
- **Flow:** Connect to Colab API (via Tailscale IP) → Submit jobs → Monitor via WebSocket → Browse outputs
- **Components:**
  - `app.py` (39KB): Main window, tabs (connection, processing options, job submission, output browser, live preview)
  - `AppSettings` dataclass (persistence to `~/.deep_live_cam_remote_windows_app.json`)
  - `ApiClient` wrapper (HTTP/WebSocket via urllib3)
  - `async_outputs.py` (16KB): Output task worker (QThread for async file listing, preview, download)
  - `ui_patches.py`, `live_webcam_patches.py`, `processing_options_patches.py`: UI customizations
  - `dark_theme.qss`: Styling

## Core Processing Pipeline

The face-swapping engine is a **pluggable frame processor pipeline**:

```
Input Video/Photo
  ↓
FFmpeg Decode (scale to max_width, cap FPS)
  ↓
Frame Queue (bounded, decouple decode from processing)
  ↓
Per-Frame Loop:
  ├─ Face Analysis (InsightFace: detect + extract embeddings)
  ├─ Processor Chain (face_swapper, face_enhancer, face_masking)
  └─ Collect output frames
  ↓
FFmpeg Encode (H.264/H.265, quality preset)
  ↓
Audio Mux (restore original audio if present)
  ↓
Output File
```

**Processor Modules** (`modules/processors/frame/`):
- **`face_swapper.py`** (76KB): Core face-swap implementation (warping, blending, mouth masking, Poisson blending, color correction)
- **`face_enhancer.py`**, `face_enhancer_gpen256.py`, `face_enhancer_gpen512.py`: GFPGAN-based quality enhancement
- **`face_masking.py`** (24KB): Mouth region masking, feathering, temporal smoothing
- Each module has a common interface: `pre_check()`, `pre_start()`, `process_frame()`, `process_image()`, `process_video()`

**Dynamic Loading:**
- `modules/processors/frame/core.py` loads processor modules at runtime
- Validates interface compliance
- Manages module state (in_memory vs. on_disk)

## Key Patterns & Conventions

### Global State Management
- `modules/globals.py`: Centralized configuration (face_processors, execution_providers, processing_options)
- Thread-safe singletons: `FACE_ANALYSER` (expensive resource, cached per thread)
- Reentrant locks for shared resources

### Configuration & Dataclasses
- Heavy use of frozen/unfrozen dataclasses for configuration (ProcessConfig, JobRequest, AppSettings, JobState)
- Settings persistence to JSON files (`~/.deep_live_cam_remote_windows_app.json`)

### Batch Idempotency
- Manifest file (`.deep_live_cam_processed.json`) tracks processed inputs + options
- Batch skip logic prevents reprocessing unchanged files
- Enables resumable workflows in Colab

### Queue-Based Parallelism
- Decode/encode are decoupled via bounded queues
- Prevents memory explosion on large files
- Thread pool for concurrent frame processing

### Error Handling & Logging
- Batch report captures per-file results and errors
- WebSocket streaming for live remote monitoring
- Manifest-based recovery (skip already-processed files)

## Dependency Organization

| Layer | Key Libraries |
|-------|---------------|
| **Core ML** | insightface, onnxruntime-gpu/silicon, tensorflow |
| **CV/Media** | opencv-python, Pillow, scikit-learn |
| **GPU** | onnx, onnxruntime-gpu, nvidia-cuda (wheels) |
| **Desktop UI** | PySide6 |
| **Remote API** | FastAPI, uvicorn, websockets |
| **Video I/O** | FFmpeg, ffprobe (bundled in project root) |
| **Virtual Camera** | pyvirtualcam (Windows only, for live preview) |
| **Utilities** | tqdm, numpy, psutil, urllib3 |

**Build vs. Runtime:**
- `requirements.txt`: Full runtime (includes Colab/ML dependencies)
- `requirements-build.txt`: Desktop app only (excludes InsightFace, TensorFlow, ONNX GPU, OpenNSFW)

## Branch Model & Workflows

- **`main`**: Product/release branch for this fork
- **`upstream-main`**: Clean sync from `hacksider/Deep-Live-Cam`
- **Feature branches**: `feat/*`, `fix/*`, `docs/*` (short-lived)

**Upstream Contributions:** Branch from `upstream-main`, keep patches small, merge back to `main` if accepted.

## Important Files to Know

| File | Lines | Purpose |
|------|-------|---------|
| `colab_batch.py` | 770 | Batch processor CLI (resumable, manifest-driven) |
| `colab_api.py` | 842 | FastAPI server for remote job management |
| `windows_app/app.py` | 1100+ | Desktop controller UI (PySide6, Tailscale) |
| `modules/core.py` | 500+ | Main orchestration (video processing, frame loops) |
| `modules/face_analyser.py` | 400+ | InsightFace integration, face detection |
| `modules/processors/frame/core.py` | 200+ | Processor module loader and orchestrator |
| `modules/processors/frame/face_swapper.py` | 1900+ | Core face-swap implementation |
| `modules/utilities.py` | 300+ | FFmpeg wrapper, file utilities |
| `google-colab/Deep_Live_Cam_Remote_Batch.py` | 600+ | Markerized notebook source (edit this, not .ipynb) |
| `scripts/py_to_ipynb.py` | 200+ | Markerized → Jupyter notebook conversion |

## Testing & Validation

**Before committing:**
```powershell
# Syntax check
python -m py_compile .\colab_batch.py .\colab_api.py .\windows_app\app.py

# Run tests (if intentionally needed)
python -m pytest .\tests -q

# Manual validation notes required in PR for:
# - UI changes (screenshots)
# - Colab changes (GPU, Drive, Tailscale behavior varies)
# - Batch processing (manifest, resume logic)
```

**Do not commit:**
- `.venv/`, `__pycache__/`, `.pytest_cache/`
- Downloaded model files
- Local app state files (`~/.deep_live_cam_remote_windows_app.json` with personal settings)
- Generated temporary notebooks or scratch exports
- Personal Google Drive paths, tokens, or Tailscale secrets

## Responsible Use

This fork maintains the upstream Deep-Live-Cam licensing and model constraints. InsightFace models may carry non-commercial research restrictions. Review upstream model and dependency licenses before distributing outputs or packaged builds.

Use only with appropriate rights and consent. If using a real person's face, obtain permission and clearly label generated or altered media. You are responsible for complying with applicable laws, platform rules, and ethical requirements.

## Related Documentation

- **`README.md`**: User-facing overview, quick start, batch CLI examples
- **`CONTRIBUTING.md`**: Branch model, PR checklist, validation expectations, notebook editing
- **`devdocs/`**: Planning docs, performance notes, release planning
- **`AGENTS.md`**: Agent task definitions and automation workflows
