# Cross-Platform Development Patterns

Derived from platform-specific fix commits: Apple Silicon regressions, Windows-only
pygrabber dependency, Linux camera enumeration, macOS Tcl/Tk path issues (2024-2026).

## Platform Guards

- Use `sys.platform` and `platform.machine()` for platform-specific branches — not `os.name`
- Windows-only imports (e.g., `pygrabber`) must be wrapped in a platform guard:
  ```python
  if sys.platform == 'win32':
      import pygrabber
  ```
- Apple Silicon detection: `sys.platform == 'darwin' and platform.machine() == 'arm64'`

## Camera Enumeration

- **Do NOT use `cv2_enumerate_cameras(cv2.CAP_AVFOUNDATION)` on macOS** — it probes indices
  0–99 through AVFoundation's native backend, which intermittently segfaults (exit code 139)
  when invalid device indices are probed. The crash is non-deterministic and has no Python-level
  catch.
- Use `cv2.VideoCapture(i)` in a bounded loop (e.g., `range(10)`) on macOS and Linux to safely
  enumerate available cameras
- Use `pygrabber` on Windows (inside `if sys.platform == 'win32'` guard)
- Do not assume camera index 0 is the built-in webcam on macOS — FaceTime Camera may have
  a different index (evidence: "FaceTime Camera Index to 0" commit)

## Tcl/Tk Environment

- Set `TCL_LIBRARY` and `TK_LIBRARY` before importing `tkinter` when running outside justfile
- The justfile auto-detects the correct Tcl/Tk paths from the mise-managed Python build
- Use `sys.base_prefix` (not `sys.prefix`) when searching for `init.tcl` — inside a venv,
  `sys.prefix` points to `.venv/` which has no Tcl/Tk files; `sys.base_prefix` points to the
  actual Python install (mise-managed) which does
- Do not hardcode Tcl/Tk version paths in source code; derive them at runtime
- Test `ImageTk.PhotoImage` creation in CI to catch `PyImagingPhoto` linkage errors early

## ffmpeg Invocation

- Pass hardware acceleration flags conditionally: `-hwaccel auto` only if provider is not `cpu`
- Use `-hwaccel_output_format cuda` only on NVIDIA; omit on other platforms
- Always verify ffmpeg is on PATH in `pre_check()` before starting any video processing

## ONNX Runtime Variants

- A single environment must have exactly one ONNX Runtime variant installed
- Mixing `onnxruntime` and `onnxruntime-gpu` in the same environment causes import errors;
  use `pyproject.toml` markers to enforce mutual exclusivity
- After changing ONNX Runtime version, run a smoke test on all supported providers before
  merging (evidence: "Downgrade onnxruntime version to 1.16.0 to fix requirements installation")

## Dependency Declarations

- Platform-conditional dependencies go in `pyproject.toml` using PEP 508 environment markers
- Never use `requirements.txt` for new dependencies — it is legacy and ignored by `uv`
- Windows extras (e.g., `pygrabber`) declared with `sys_platform == 'win32'` marker

## Testing Across Platforms

- macOS ARM and NVIDIA CUDA are the highest-priority test targets
- AMD (ROCM) and CPU-only are secondary — verify at least once per release
- GitHub Actions matrix should include `macos-14` (Apple Silicon) and `ubuntu-latest` (CUDA)
  runners for integration tests
