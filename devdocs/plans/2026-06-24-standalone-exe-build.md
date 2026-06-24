# Plan: Standalone desktop app EXE build

## Scope
- In:
  - Add an isolated build environment workflow using `.venv_build/`.
  - Add `requirements-build.txt` for PyInstaller/build-only dependencies.
  - Add a repeatable PowerShell build script for the current desktop remote app.
  - Document the build workflow in README and release notes.
- Out:
  - No build execution by Codex.
  - No tests or validation by Codex.
  - No installer/MSIX/signing pipeline yet.
  - No cross-platform packaging yet.

## GUI Components Affected
- [ ] Main window
- [ ] Panels/widgets
- [ ] Dialogs
- [x] Assets (icons, qss, etc.)
- [x] Config/build files
- [x] PyInstaller build script

## Action items
- [x] Create feature branch `feature/standalone-exe-build`.
- [x] Add `requirements-build.txt` with desktop-controller-only runtime requirements plus PyInstaller.
- [x] Add `.venv_build/` ignore rule.
- [x] Add `scripts/build_remote_app.ps1` for repeatable PyInstaller builds, including `-RecreateVenv` for failed/stale build environments.
- [x] Add `-Lite` build mode to exclude live webcam dependencies (`cv2`, `numpy`, `pyvirtualcam`) and emit `Deep-Live-Cam-Remote-Lite`.
- [x] Add versioned artifact names and versioned `dist/<version>/` output folders.
- [x] Add `-Version` parameter with fallback to `pyproject.toml` then `git describe --tags --always`.
- [x] Add Python-version suffixes, final artifact existence checks, and build summary output.
- [x] Add a manual GitHub Actions build workflow for versioned desktop app artifacts.
- [x] Add a manual GitHub Actions release-upload workflow for existing GitHub Releases.
- [x] Include desktop app assets (`windows_app/icon.ico`, `windows_app/dark_theme.qss`) in the build command.
- [x] Document build usage in README.
- [x] Update unreleased notes.

## Decisions
- Use `.venv_build/` instead of `.venv` so PyInstaller bundles from a clean build environment; keep it desktop-client-only instead of installing Colab/server/model packages.
- Start with `--onedir` as the default because it is easier to inspect and debug than `--onefile`.
- Provide `-OneFile` as an opt-in build script switch for later release experiments.
- Provide `-Lite` as a smaller controller build that omits Live webcam dependencies; photo/video remote jobs and API control remain the focus.
- Keep versioned release output under `dist/<version>/` and PyInstaller scratch/spec files under `build/`; both are ignored.
- Keep GitHub Actions packaging manual-only so release artifacts are produced deliberately.
- Use Python 3.11 for Actions packaging because the desktop app/build dependencies are already known around the current local build baseline.
- Do not add UPX support for now; the expected size reduction is not worth the added complexity and potential AV false-positive risk.

## Open questions
- Whether the eventual default release artifact should be onedir zip, onefile exe, lite onefile exe, or installer. Versioned names/folders and Actions workflows now support the current options.
- Whether to rename `windows_app/` and launchers to neutral `remote_app/` names before the first public release.

## Build outputs observed
- Full onefile: `dist/Deep-Live-Cam-Remote.exe` - 107,627,357 bytes.
- Lite onefile: `dist/Deep-Live-Cam-Remote-Lite.exe` - 56,643,485 bytes.
- Lite saved 50,983,872 bytes (47.4%) by excluding `cv2`, `numpy`, and `pyvirtualcam`.

## Validation
- [x] Build on Windows: USER confirmed onedir build worked; Codex ran full and Lite onefile build commands successfully
- [ ] Launch packaged app: USER
- [ ] Connect to Colab API from packaged app: USER
- [ ] Run GitHub Actions build workflow: USER
- [ ] Run GitHub Actions release workflow against a real release tag: USER
- [ ] Verify icon/QSS/media playback assets: USER
