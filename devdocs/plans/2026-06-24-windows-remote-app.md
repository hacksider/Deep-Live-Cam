# Plan: Windows app for Deep-Live-Cam-Remote

## Scope
- In:
  - Add a Windows desktop app for `projects/Deep-Live-Cam-Remote` inspired by `tmp/Deep-Live-Cam-Google-Colab` GUI flows.
  - Support three user modes: many photos at once, videos, and live webcam.
  - Reuse existing Deep-Live-Cam-Remote processing code where possible instead of duplicating the engine.
  - Keep Colab notebook-backed workflow synchronized through `$ipynb-roundtrip` if notebook cells need changes.
- Out:
  - No full rewrite of `colab_batch.py` or the model pipeline.
  - No cloud account automation beyond launching/opening user-provided Colab/Tailscale setup guidance.
  - No agent-run tests or smoke tests under `$fast-implement`; final validation remains with the user.

## Action items
- [x] Merge PR #2, refresh `main`, create `feature/windows-remote-app`, then keep this plan updated as the working source of truth.
- [x] Audit the reference GUI in `tmp/Deep-Live-Cam-Google-Colab/modules/ui.py`, `modules/core.py`, `modules/globals.py`, and `modules/processors/frame/remote_processor.py` for reusable UX and remote-connection behavior.
- [x] Audit `projects/Deep-Live-Cam-Remote` current capabilities, especially `colab_batch.py`, `modules/ui.py`, live webcam options in `modules/core.py`, and the `google-colab/Deep_Live_Cam_Remote_Batch.py` notebook source.
- [x] Choose a minimal architecture: add a small Windows app package under `projects/Deep-Live-Cam-Remote/windows_app/` that wraps existing local commands and imports shared helper modules only where safe.
- [x] Add shared job/config helpers for app state: source face path, input photo folder/files, input video files/folder, output directory, Colab/Tailscale host, quality/performance flags, enhancer choice, overwrite/skip behavior, and recent paths.
- [x] Add the Photos tab/workflow: allow many images at once or a folder, queue all supported images, process with one selected source face or optional map config, preserve per-file output names, and show skipped/done/failed counts.
- [x] Extend or adapt batch processing for photo inputs if `colab_batch.py` is currently video-only: add image discovery, image output naming, manifest/report integration, and shared source-face/model loading with the existing engine.
- [x] Add the Videos tab/workflow: wrap existing `colab_batch.py scan`/`process` behavior, expose recursive folder selection, max width/FPS, start/duration, encoder/quality, many faces, enhancer, overwrite, skip-processed, ZIP/report options, and live log streaming.
- [x] Add the Live Webcam tab/workflow: reuse existing `run.py --live --frame-processor remote_processor --remote-host <tailscale-ip>` style flow where available, with source image, camera index, virtual camera name, width/height/FPS, mirror/resizable toggles, and start/stop controls.
- [x] Add a Connection panel: accept a single Tailscale/Colab host and derive `tcp://host:5555`, `tcp://host:5556`, and `tcp://host:5557`; also allow advanced manual endpoint override like the reference app.
- [x] Add background process management: run long jobs off the UI thread, stream stdout/stderr to a log box, disable conflicting controls during active jobs, support cancel/stop, and avoid orphaned subprocesses.
- [x] Add durable local settings, likely JSON under a user-local app folder or `projects/Deep-Live-Cam-Remote/.windows_app_state.json` if repo-local state is acceptable and gitignored.
- [x] Add launcher scripts for Windows: `run_windows_app.ps1` and/or `run-windows-app.bat`, using PowerShell-safe paths and the project virtual environment when present.
- [x] Update docs: add Windows app usage to `projects/Deep-Live-Cam-Remote/README.md`, add repo guidance to `AGENTS.md`, and link this plan from the relevant devdocs feature note.
- [x] If notebook changes are required, edit `projects/Deep-Live-Cam-Remote/google-colab/Deep_Live_Cam_Remote_Batch.py`, then rebuild `Deep_Live_Cam_Remote_Batch.ipynb` with `$ipynb-roundtrip` and remove temporary round-trip files.
- [x] Minimal validation only under `$fast-implement`: ran `python -m py_compile` for changed Python files; did not run unit tests or smoke tests; final checks are deferred to the user.
- [x] Prepare handoff: summarize touched files, commands added, remaining manual checks for photo batch/video/live webcam, and any known limitations around Colab/Tailscale availability.

## Decisions and Design Changes
- 2026-06-24 Use a dedicated Windows app wrapper around `Deep-Live-Cam-Remote` instead of porting the older reference GUI wholesale; this avoids copying stale upstream code and keeps the modern batch pipeline authoritative.
- 2026-06-24 User selected a new HTTP/WebSocket API layer over private Tailscale, fixed Google Drive folders, PySide6 UI, remote-only processing, no NSFW filter, and Python 3.11 `.venv` for local Win10.
- 2026-06-24 Treat photos, videos, and live webcam as separate UI workflows sharing one connection/settings layer.
- 2026-06-24 Prefer a single host input with derived ZMQ endpoints for normal users, while retaining manual endpoint override for compatibility with the reference app flow.
- 2026-06-24 Under `$fast-implement`, tests and smoke tests are intentionally not run by the agent; final validation remains with the user.

## Open questions
- None blocking. Assumption: the first implementation can be a Python/CustomTkinter Windows app to match the existing codebase and the reference app.

## Validation
- [x] `python -m py_compile` completed for changed Python files.
- [ ] Unit tests and smoke tests deferred to user.

## Handoff Summary

### Key Files
- `windows_app/app.py` - Main app with settings, API client, workers
- `windows_app/ui_patches.py` - UI enhancements (tabs, outputs, sync)
- `windows_app/async_outputs.py` - Async output fetching/preview
- `windows_app/icon.ico` - App icon
- `colab_api.py` - FastAPI server with job endpoints
- `colab_batch.py` - Batch processor (photos/videos)
- `google-colab/Deep_Live_Cam_Remote_Batch.py` - Markerized notebook source
- `google-colab/Deep_Live_Cam_Remote_Batch.ipynb` - Colab notebook

### Launch Commands
```powershell
# Windows app
.\run_windows_remote_app.ps1

# Or directly
.\.venv\Scripts\python.exe run_windows_remote_app.py
```

### Latest Features (PR #3)
- Dark title bar + custom icon
- Full processing options on both Photos/Videos tabs
- Video percentage range (start/end %)
- Start/Stop toggle with graceful cancel
- Resumable Colab cells with auto git pull
- Settings sync between tabs