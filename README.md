# Deep-Live-Cam Remote

Deep-Live-Cam Remote is a fork of [hacksider/Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) focused on Google Colab batch processing and a desktop remote controller app.

This variant keeps the upstream Deep-Live-Cam local pipeline, then adds:

- A Colab notebook workflow for resumable setup and batch jobs.
- A headless photo/video batch CLI for files already in Colab or Google Drive.
- A FastAPI/WebSocket controller API for remote jobs.
- A desktop controller app for managing Colab jobs over a private Tailscale connection.
- Output browsing, previews, cancel support, and shared processing options for photo/video jobs.

> Status: early downstream fork. The current desktop controller implementation is Windows-oriented, but the repo direction is cross-platform remote control.

## Project layout

| Path | Purpose |
| --- | --- |
| `google-colab/Deep_Live_Cam_Remote_Batch.ipynb` | Colab notebook users run directly. |
| `google-colab/Deep_Live_Cam_Remote_Batch.py` | Markerized source for the notebook. Edit this, then rebuild the `.ipynb`. |
| `colab_batch.py` | Standalone batch processor for photos/videos. |
| `colab_api.py` | FastAPI server used by the desktop remote app. |
| `windows_app/` | Current desktop controller implementation with the canonical app entrypoint plus focused helper modules for UI, outputs, options, and Live webcam. |
| `run_windows_remote_app.py` / `.ps1` | Local launcher for the current desktop app. |
| `scripts/` | Notebook round-trip helpers. |
| `devdocs/` | Planning and release notes for this fork. |

## Quick start: Google Colab batch processing

1. Open `google-colab/Deep_Live_Cam_Remote_Batch.ipynb` in Google Colab.
2. Run the setup cells.
3. Mount Google Drive when prompted.
4. Put your files under the default Drive layout:

```text
MyDrive/DeepLiveCamRemote/source/source.png
MyDrive/DeepLiveCamRemote/photos/
MyDrive/DeepLiveCamRemote/videos/
MyDrive/DeepLiveCamRemote/outputs/
```

5. Run the photo or video batch cells.

The notebook clones this repository from:

```text
https://github.com/djebaz/Deep-Live-Cam-Remote.git
```

## Batch CLI

From a Colab/runtime checkout:

```bash
python colab_batch.py process \
  --source-face /content/drive/MyDrive/DeepLiveCamRemote/source/source.png \
  --input-dir /content/drive/MyDrive/DeepLiveCamRemote/videos \
  --output-dir /content/drive/MyDrive/DeepLiveCamRemote/outputs/videos
```

Photo batch example:

```bash
python colab_batch.py photos \
  --source-face /content/drive/MyDrive/DeepLiveCamRemote/source/source.png \
  --input-dir /content/drive/MyDrive/DeepLiveCamRemote/photos \
  --output-dir /content/drive/MyDrive/DeepLiveCamRemote/outputs/photos
```

Face mapping workflow:

```bash
python colab_batch.py scan --input-dir /content/in --mapping-dir /content/mapping
python colab_batch.py process \
  --input-dir /content/in \
  --output-dir /content/out \
  --map-config /content/mapping/face_mapping.json
```

Useful processing options include many-face mode, mouth mask, opacity, sharpening, interpolation, Poisson blending, color correction, GFPGAN/GPEN enhancement, start/duration, output FPS, and max width. Run:

```bash
python colab_batch.py --help
python colab_batch.py process --help
```

## Desktop remote controller

The current controller app connects to `colab_api.py` running in Colab over Tailscale.

### In Colab

1. Open `google-colab/Deep_Live_Cam_Remote_Batch.ipynb`.
2. Run setup.
3. Install/connect Tailscale from the notebook cells.
4. Start the API server cell.
5. Copy the displayed Tailscale IP and use port `7860` in the desktop app.

### On Windows today

```powershell
git clone https://github.com/djebaz/Deep-Live-Cam-Remote.git
Set-Location .\Deep-Live-Cam-Remote
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run_windows_remote_app.ps1
```

The app currently provides:

- Photos and Videos tabs with shared processing options.
- Recursive input scanning, overwrite/skip behavior, and graceful cancel.
- Video percentage range selection.
- Live webcam InsightFace pack selection (`buffalo_l`, `buffalo_m`, `buffalo_s`) with `buffalo_l` as the safest default, plus fp32/fp16 swapper precision selection for `swap_ms` comparison.
- Live webcam uses configurable buffered, fixed-cadence preview rendering so bursty backend frames are delayed briefly, evenly paced, and coalesced instead of repainting stale frame backlogs.
- Live preview size can be switched between fit, real pixels, 1.5x, and 2x; scaled modes automatically fall back to fit when larger than the panel.
- Live webcam can send and receive either JPEG or WebP frames, with frame quality and codec values reported in backend diagnostics/perf logs.
- Output browser with preview/player support.
- Local source/input upload to the Colab API.
- Settings sync between photo and video workflows.

## Standalone desktop app build

Use an isolated build virtual environment so PyInstaller only bundles dependencies intended for the desktop remote controller. `requirements-build.txt` intentionally does not install Colab/server/model packages such as InsightFace, TensorFlow, ONNX Runtime GPU, or OpenNSFW.

```powershell
py -3.11 -m venv .venv_build
.\.venv_build\Scripts\python.exe -m pip install -r requirements-build.txt
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_remote_app.ps1
```

The default build is `--onedir` and writes versioned output under `dist/<version>/`. Artifact names include the resolved version and Python version, for example `Deep-Live-Cam-Remote-0.1.0-py3.11.exe`.

Optional switches:

```powershell
# Build with an explicit release version
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_remote_app.ps1 -Version 0.1.0

# Reinstall requirements and clean PyInstaller cache/build output
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_remote_app.ps1 -Clean

# Try a single-file executable
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_remote_app.ps1 -OneFile

# Build a smaller controller without Live webcam dependencies
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_remote_app.ps1 -OneFile -Lite

# Recreate a failed or stale build environment
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_remote_app.ps1 -RecreateVenv

# Reuse an already prepared .venv_build environment
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_remote_app.ps1 -SkipInstall
```

Build outputs and `.venv_build/` are intentionally ignored by git. If `-Version` is omitted, the script tries `pyproject.toml` and then `git describe --tags --always`. Lite builds exclude `cv2`, `numpy`, and `pyvirtualcam`, so the Live webcam workflow is not bundled.

### GitHub Actions packaging

Two manual workflows are available under the repository **Actions** tab:

- **Build Desktop App EXE** (`.github/workflows/build-desktop-app.yml`) builds Full, Lite, or both flavors and uploads the versioned `dist/<version>/` output as a short-retention Actions artifact.
- **Build and Release Desktop App** (`.github/workflows/build-and-release.yml`) builds from an existing release tag and uploads the generated `.exe` assets to that GitHub Release. For `onedir` builds, the workflow zips each bundle before uploading.

Both workflows use Python 3.11, call `scripts/build_remote_app.ps1`, keep UPX disabled, and expose the same Full/Lite plus `onefile`/`onedir` choices as the local build script.

## Notebook round-trip workflow

The Colab notebook is generated from the markerized Python source.

Edit:

```text
google-colab/Deep_Live_Cam_Remote_Batch.py
```

Then rebuild:

```powershell
python scripts/py_to_ipynb.py `
  .\google-colab\Deep_Live_Cam_Remote_Batch.py `
  .\google-colab\Deep_Live_Cam_Remote_Batch.ipynb `
  --eol auto
```

If editing the notebook in Colab, export back to markerized source:

```powershell
python scripts/ipynb_to_py.py `
  .\google-colab\Deep_Live_Cam_Remote_Batch.ipynb `
  .\google-colab\Deep_Live_Cam_Remote_Batch.py `
  --eol auto
```

Preserve cell IDs, marker lines, `meta_b64`, `NOTEBOOK_META_B64`, and markdown/raw sentinels.

## Relationship to upstream

This repository is an official GitHub fork of `hacksider/Deep-Live-Cam`, but it is maintained as a downstream remote/Colab variant.

Suggested branch roles:

- `main` - this fork's product/release branch.
- `upstream-main` - clean sync branch from `hacksider/main`.
- feature branches - focused changes or upstream PR candidates.

For upstream contributions, keep patches small and branch from `upstream-main`.

## Models and licenses

This project follows the upstream Deep-Live-Cam licensing and model constraints.

- Code license: see `LICENSE`.
- Upstream project: <https://github.com/hacksider/Deep-Live-Cam>
- InsightFace models may carry non-commercial research restrictions. Review upstream model and dependency licenses before distributing outputs or packaged builds.

Model files are not committed. The Colab/batch setup downloads and validates the required swapper model when needed.

## Responsible use

Use this software only with appropriate rights and consent. If using a real person's face, obtain permission and clearly label generated or altered media where appropriate. You are responsible for complying with applicable laws, platform rules, and ethical requirements.

The remote/Colab path in this fork intentionally focuses on workflow and does not add extra consent modals or content-gate UI beyond upstream behavior.

## Development notes

Common local commands:

```powershell
# Inspect status
git status --short

# Python syntax check for edited files
python -m py_compile .\colab_batch.py .\colab_api.py

# Focused tests, when intentionally requested
python -m pytest .\tests -q
```

Do not commit virtual environments, caches, downloaded models, local state, or generated temporary notebooks.
