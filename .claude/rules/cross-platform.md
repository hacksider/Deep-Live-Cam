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

- **Do NOT probe cameras with `cv2.VideoCapture(i)` on macOS** — any probe of an invalid index
  (one beyond the number of attached cameras) triggers the OBSENSOR (OrbbecSDK) backend, which
  corrupts global OpenCV state and causes SIGSEGV (exit 139). This happens with all backends
  including `CAP_ANY` and `CAP_AVFOUNDATION`. Even one probe is enough to crash.
- **Do NOT run camera enumeration in a subprocess on macOS** — `subprocess.run()` calls `fork()`
  internally; forking after cv2/AVFoundation is initialised in a multithreaded process is unsafe
  on macOS (Objective-C runtime) and crashes the **parent** process.
- **macOS solution**: skip `cv2.VideoCapture` probing entirely. Default to `[0, 1]` (Camera 0,
  Camera 1) which covers FaceTime and a common USB webcam. The user can select the correct index
  from the UI dropdown if they have more cameras.
- Use `cv2.VideoCapture(i)` in a bounded loop on **Linux only** — break after 3 consecutive
  failures to avoid probing large index ranges and virtual cameras gaps.
- Use `pygrabber` on Windows (inside `if sys.platform == 'win32'` guard)
- Do not assume camera index 0 is the built-in webcam on macOS — FaceTime Camera may have
  a different index (evidence: "FaceTime Camera Index to 0" commit)

## Tkinter Thread Safety

- **Never call tkinter widget methods from background threads** — Tcl/Tk is single-threaded.
  Calling `widget.configure()`, `ROOT.update()`, or any Tcl function from a non-main thread
  causes SIGSEGV (exit 139) via Tcl heap corruption or SIGTRAP (exit 133) via `Tcl_Panic`
  after `TclpFree` detects an invalid block (`alloc: invalid block`).
- Use `ROOT.after(0, lambda: widget.configure(...))` to schedule UI updates from background
  threads — this posts the call to the main thread's event queue.
- Use a default argument capture (`lambda t=text: ...`) to avoid late-binding closure bugs
  when scheduling inside a loop.
- **Do NOT call `ROOT.update()` from within event handlers or `after()` callbacks** — it
  re-enters the event loop and can cause recursive processing. Let the event loop handle
  redraws on its own schedule.
- Common offenders: status label updates from model-loading threads (`get_face_swapper`,
  `get_face_enhancer`), progress messages from frame processing threads.

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
