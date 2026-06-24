# AGENTS.md

## Start Here

- If you are working on Windows, use PowerShell 7 (`pwsh`) and prefer explicit PowerShell cmdlets such as `Get-ChildItem`, `Set-Location`, `Get-Content`, and `Set-Content`; avoid Unix-only shell assumptions.
- Before editing, inspect the target tree and read repo-local guidance: `AGENTS.md`, `README.md`, `CONTRIBUTING.md`, and `HOWTO.md` when present.
- Keep commands runnable from the repository root unless a section explicitly says otherwise.
- This repository is a downstream fork of `hacksider/Deep-Live-Cam` focused on Google Colab batch processing and a desktop remote controller app.
- `main` is the fork product/release branch. `upstream-main` is the clean sync branch from `hacksider/main`.

## A0. Environment Gate for Windows PowerShell Workflows

When working in a Windows PowerShell terminal, run this first:

```powershell
$PSVersionTable
```

Windows PowerShell workflows must satisfy:

- `$PSVersionTable.PSEdition -eq 'Core'`
- `$PSVersionTable.PSVersion.Major -eq 7`

If either condition is not satisfied: **ABORT(A0)** for Windows-specific commands and switch to `pwsh` 7 before continuing.

This gate applies to Windows/PowerShell workflows. It does not require Linux, macOS, or Colab-only contributors to use PowerShell.

## Repository Map

- `run.py` - upstream Deep-Live-Cam local GUI/CLI entry point.
- `colab_batch.py` - remote/Colab batch processor for photos and videos.
- `colab_api.py` - FastAPI/WebSocket server used by the desktop remote controller.
- `google-colab/Deep_Live_Cam_Remote_Batch.py` - readable markerized notebook source.
- `google-colab/Deep_Live_Cam_Remote_Batch.ipynb` - generated Colab notebook artifact kept synchronized with the markerized source.
- `windows_app/` - current desktop remote controller implementation. The near-term app is Windows-oriented, but keep naming and docs open to future cross-platform clients where practical.
- `scripts/` - notebook round-trip helpers.
- `devdocs/` - planning and release notes.
- `tests/` - focused tests for batch and helper behavior.

## Python / Deep-Live-Cam Workflow

- Use Python 3.11 when possible:
  ```powershell
  py -3.11 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install -r requirements.txt
  ```
- Local GUI entry point: `python run.py`.
- Batch/remote entry points: `python colab_batch.py ...`, `python colab_api.py --host 0.0.0.0 --port 7860`, and `python run_windows_remote_app.py`.
- For Python syntax checks, prefer `python -m py_compile <file>` on the files you changed.
- Do not run tests unless the user explicitly asks for validation.

## Desktop Remote App / Colab API

- The current desktop app lives in `windows_app/` and launches via `run_windows_remote_app.py`, `run_windows_remote_app.ps1`, or `run-windows-remote-app.bat`.
- Use the repo `.venv` for Python package installs and app runs. Example: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- The Colab API lives in `colab_api.py` and exposes HTTP job endpoints plus WebSocket live/progress endpoints on port `7860` by default.
- The preset Drive layout is `/content/drive/MyDrive/DeepLiveCamRemote/{source,photos,videos,outputs}`.
- The remote app/backend is intentionally remote-only and does not add extra NSFW filtering, consent modals, or safety-gate UI beyond upstream behavior.
- Photo batches use `python colab_batch.py photos ...`; video batches use `python colab_batch.py process ...`. Keep output paths mirrored relative to the selected input root.

### Current Desktop App Features

- Dark title bar on Windows 10/11 via DWM API; custom app icon in title bar/taskbar.
- Photos and Videos tabs both expose full processing options: recursive, overwrite, skip processed, many faces, enhancer, opacity, sharpness, mouth mask, interpolation, poisson blend, color correction.
- Video percentage range: start/end % spinboxes to process only a portion of videos.
- Start/Stop toggle: batch start buttons switch to red Stop when running; cancel is graceful.
- Outputs tab: resizable split view with list panel and preview/player; autoplay with prefetch.
- Local file upload: source faces and input folders can be local desktop paths; the app uploads to Colab before starting jobs.
- Settings sync: changes in one tab sync to the other when saving or starting jobs.

### Colab Notebook Features

- Resumable cells: Clone/install, Tailscale install, and Tailscale auth cells skip already-completed steps.
- Auto-update on re-run: setup cell runs `git pull` when repo already exists, so code updates apply without deleting the cloned directory.
- The notebook clones `https://github.com/djebaz/Deep-Live-Cam-Remote.git` branch `main`.

## Notebook Round-Trip

The `Deep_Live_Cam_Remote_Batch.ipynb` Colab notebook uses git clone to fetch the latest code from this repository. Changes to the notebook structure should be made in the markerized `.py` source and rebuilt to `.ipynb`.

Rules:

- Edit the markerized `.py` source (`google-colab/Deep_Live_Cam_Remote_Batch.py`) for deterministic diffs; rebuild the `.ipynb` before committing.
- After edits made in Colab or directly in an `.ipynb`, export back to markerized `.py` and review the diff.
- Preserve cell ids, marker lines, `meta_b64`, `NOTEBOOK_META_B64`, `MARKDOWN` / `ENDMARKDOWN`, and `RAW` / `ENDRAW` sentinels.
- Remove throwaway round-trip files such as `_roundtrip.py`, `_roundtrip.ipynb`, or temp notebooks after validation unless the user asks to keep them.
- Run conversions from the repo root, then check `git diff`.

Commands:

```powershell
# Markerized py -> notebook
python scripts/py_to_ipynb.py `
  .\google-colab\Deep_Live_Cam_Remote_Batch.py `
  .\google-colab\Deep_Live_Cam_Remote_Batch.ipynb `
  --eol auto

# Notebook -> markerized py
python scripts/ipynb_to_py.py `
  .\google-colab\Deep_Live_Cam_Remote_Batch.ipynb `
  .\google-colab\Deep_Live_Cam_Remote_Batch.py `
  --eol auto
```

Important: since the notebook clones from GitHub, push changes to `main` before running the notebook in Colab if the notebook/runtime needs those changes.

## Context7 Documentation Rule

Use Context7 MCP to fetch current documentation whenever the user asks about a library, framework, SDK, API, CLI tool, or cloud service. Start with `resolve-library-id` unless the user provides an exact `/org/project` library ID, then call `query-docs` with the selected ID and the user's full question. Prefer Context7 over web search for library docs.

Do not use Context7 for general refactoring, writing scripts from scratch, business-logic debugging, code review, or general programming concepts.

## Git / Safety

- Check `git status --short` before and after edits.
- Keep generated artifacts, caches, downloaded models, local app state, and temp files out of commits unless explicitly requested.
- Avoid broad upstream rewrites; this repository is a fork and should keep upstream-compatible changes isolated when possible.
- When changing notebook-backed workflows, keep `.py` and `.ipynb` synchronized and mention the conversion command used in the handoff.
- For upstream PRs to `hacksider/Deep-Live-Cam`, branch from `upstream-main` and keep patches small.
