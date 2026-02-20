# TODO — Code Review Findings

Generated from code review of v2.0.3c (2026-02-19). Items ordered by severity.

---

## Critical

- [x] **`imwrite_unicode` is broken** — dead code on line 14, double-dot extension bug, only writes files *without* extensions
  `modules/__init__.py:11-17`
  Fixed in `fix(globals,core)`: removed dead line, fixed double-dot, added else-branch for files with existing extension.

- [x] **Memory limit is non-functional on macOS** — `1024 ** 6` (exbibytes) should be `1024 ** 3`
  `modules/core.py:159`
  Fixed in `fix(globals,core)`.

- [x] **Face swapper model download URL is wrong** — points to HuggingFace blob HTML page; use `/resolve/` instead of `/blob/`; also saves to wrong directory (frame processor dir instead of `models/`)
  `modules/processors/frame/face_swapper.py:50`
  Fixed in `fix(face-swapper)`: changed `blob` → `resolve` and `download_directory_path = abs_dir` → `models_dir`.

- [x] **`face_masking.py` references undefined globals** — `mouth_mask_size`, `eyes_mask_size`, `eyebrows_mask_size` are not defined in `globals.py`; crash when mouth masking is used
  `modules/processors/frame/face_masking.py:94-95, 159, 295`
  Fixed in `fix(globals,core)`: added all three float globals with default `1.0`.

- [x] **Unbound local crash in `create_eyes_mask`/`create_eyebrows_mask`** — variables `min_x`, `min_y`, etc. only defined inside `if landmarks is not None` block but referenced on return path
  `modules/processors/frame/face_masking.py:219, 414`
  Fixed in `fix(face-masking,face-analyser,ui-analysis)`: added early return `(mask, None, (0,0,0,0), None)` when `landmarks is None`.

- [x] **Model downloads have no TLS certificate verification** — `urllib.request.urlopen` used without explicit SSL context; checksums are optional
  `modules/utilities.py:294`
  Fixed in `fix(utilities)`: wrapped `urlopen` with `ssl.create_default_context()`.

---

## High

- [x] **`process_frame_v2` called but commented out in face enhancer** — raises `AttributeError` during live mode with face enhancement and `map_faces=True`
  `modules/ui_webcam.py:116`, `modules/processors/frame/face_enhancer.py:199-204`
  Fixed in `fix(face-enhancer,ui-webcam)`: uncommented `process_frame_v2`; fixed live-mode no-target branch to call `process_frame(None, frame)` for non-enhancer processors.

- [x] **Webcam preview loop blocks main thread** — `while not stop_event.is_set()` with manual `ROOT.update()` instead of `root.after()`; GUI unresponsive during live preview
  `modules/ui_webcam.py:194-217`
  Fixed in `refactor(ui-webcam)`: extracted loop body into `_display_next_frame()`, rescheduled via `ROOT.after(1, ...)`.

- [x] **`live_resizable` flag has no effect** — both branches of the conditional execute identical `fit_image_to_size` calls
  `modules/ui_webcam.py:201-208`
  Fixed in `refactor(ui-webcam)`: `live_resizable=False` branch now skips `fit_image_to_size` and displays at native camera resolution.

- [x] **Silent `sys.exit()` on missing processor interface method** — no error message; users get a silent crash
  `modules/processors/frame/core.py:26`
  Fixed in `fix(ui-mapper,processor-core)`: replaced `sys.exit()` with `raise ImportError(f"... missing method: {method_name}")`.

- [ ] **Duplicated masking logic** — `create_face_mask`, `create_lower_mouth_mask`, `apply_color_transfer`, `apply_mouth_area` all exist in both `face_swapper.py` and `face_masking.py` with diverging implementations; consolidate into `face_masking.py`
  `modules/processors/frame/face_swapper.py`, `modules/processors/frame/face_masking.py`
  *Deferred to Wave 13 — requires stable codebase first.*

- [ ] **Pervasive unprotected mutable global state** — UI thread and processing threads read/write `globals.py` variables (`opacity`, `source_path`, `many_faces`, etc.) without synchronisation; `MAP_LOCK` only protects the face map
  `modules/globals.py`
  *Deferred to Wave 13.*

---

## Medium

- [x] **`default_target_face` crashes when no faces detected** — `best_face` stays `None`, then `best_face['bbox']` raises `TypeError`
  `modules/face_analyser.py:225`
  Fixed in `fix(face-masking,face-analyser,ui-analysis)`: added `if best_face is None: continue` guard.

- [x] **`check_and_ignore_nsfw` unbound variable** — `check_nsfw` is never assigned if `target` is neither `str` nor `ndarray`, causing `UnboundLocalError`; also uses `type()` instead of `isinstance()`
  `modules/ui_analysis.py:49-53`
  Fixed in `fix(face-masking,face-analyser,ui-analysis)`: initialised `check_nsfw = None`; replaced `type(x) is T` with `isinstance(x, T)`.

- [x] **`POPUP_SCROLL_WIDTH` and `POPUP_LIVE_SCROLL_WIDTH` are tuples, not integers** — trailing commas; passed to `CTkScrollableFrame(width=...)`
  `modules/ui_mapper.py:26, 31`
  Fixed in `fix(ui-mapper,processor-core)`: removed trailing commas.

- [x] **`get_face_enhancer` `RuntimeError` propagates through pipeline** — `enhance_face()` does not catch the exception from a failed enhancer init
  `modules/processors/frame/face_enhancer.py:115, 123`
  Fixed in `fix(face-enhancer,ui-webcam)`: wrapped `get_face_enhancer()` in `try/except RuntimeError`; logs and returns input frame.

- [x] **`PREVIOUS_FRAME_RESULT` not thread-safe** — read/written from multiple worker threads without a lock; frame interpolation may produce corrupted results in parallel video processing
  `modules/processors/frame/face_swapper.py:28`
  Fixed in `fix(face-swapper,face-analyser,ui-mapper)`: replaced module-level global with `threading.local()` per-thread storage via `_get_previous_frame()`/`_set_previous_frame()`.

- [x] **Built-in `map` shadowed** — `for map in modules.globals.source_target_map` used extensively; rename to `face_map` or `entry`
  `modules/face_analyser.py:98, 104, 112`, `modules/ui_mapper.py:51, 119, 180, 223, 241`
  Fixed in `fix(face-swapper,face-analyser,ui-mapper)`: renamed to `face_map` in face_analyser.py and `source_map` in ui_mapper.py.

- [x] **Camera enumeration can block GUI startup for ~30s** — `cv2.VideoCapture(i)` probes on slow drivers; run in background thread
  `modules/ui.py:783-795`
  Fixed in `perf(ui)`: collapsed duplicate platform branches; probe loop moved into `threading.Thread(daemon=True)`; combobox updated via `root.after(0, ...)`.

- [x] **`cluster_analysis.py` fails on single-element input** — `KMeans(n_clusters=k)` fails when `k > len(embeddings)`; `max(diffs)` raises `ValueError` on empty list
  `modules/cluster_analysis.py:18-19`
  Fixed in test commit: added `if len(embeddings) <= 1: return embeddings` guard and `if not diffs:` guard before `max(diffs)`.

---

## Low

- [x] **`has_image_extension` matches without a dot** — `endswith("png")` matches `myfilepng`; also misses `gif` and `bmp` which are in `globals.file_types`
  `modules/utilities.py:269`
  Fixed in test commit: changed to `.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))`.

- [x] **`FPS_CAP` type mismatch** — annotated as `int` but formatted with `:.1f` (implies float)
  `modules/globals.py:14`, `modules/core.py:253`
  Fixed in `fix(globals,core)`: changed to `FPS_CAP: float = 30.0`.

- [x] **Potential ffmpeg argument injection via crafted filenames** — file paths from the dialog are not validated as regular files before being passed to subprocess
  `modules/utilities.py:28-66`
  Fixed in `fix(utilities)`: added `_validate_path_for_subprocess()` that rejects basenames starting with `-`; called from `normalize_output_path`.

- [x] **Duplicate `tkinter_fix.py`** — identical files at root and `modules/tkinter_fix.py`; remove the root-level copy
  `tkinter_fix.py`, `modules/tkinter_fix.py`
  Fixed in `chore(cleanup)`: deleted root-level `tkinter_fix.py`.

- [x] **Duplicate type definitions** — `modules/typing.py` and `modules/custom_types.py` both define `Face` and `Frame`; consolidate
  `modules/typing.py`, `modules/custom_types.py`
  Fixed in `chore(cleanup)`: deleted `modules/custom_types.py`; all imports use `modules/typing.py`.

- [x] **Confusing duplicate entry point** — `modules/run.py` duplicates `run.py` with incorrect relative imports; remove or clarify
  `modules/run.py`
  Fixed in `chore(cleanup)`: deleted `modules/run.py`.

- [x] **Silent method-missing crash has no diagnostic** — already listed in High; silent `sys.exit()` gives no output
  *(see High section — fixed in `fix(ui-mapper,processor-core)`)*

---

## Test Coverage

21 tests added across 7 test files (1 skipped pending scikit-learn install).

- [x] `modules/__init__.py` — `imwrite_unicode` with/without extension (`tests/test_init.py`)
- [x] `modules/face_analyser.py` — None face guards, `default_target_face` null safety (`tests/test_face_analyser.py`)
- [x] `modules/utilities.py` — `has_image_extension` edge cases including dot-prefix (`tests/test_utilities.py`)
- [x] `modules/processors/frame/core.py` — `ImportError` on missing interface method (`tests/test_processor_core.py`)
- [x] `modules/processors/frame/face_masking.py` — undefined globals verified via `tests/test_globals.py`
- [x] `modules/cluster_analysis.py` — single embedding and empty input guards (`tests/test_cluster_analysis.py`)
- [x] `modules/processors/frame/face_enhancer.py` — `process_frame_v2` callable; `RuntimeError` pass-through (`tests/test_face_enhancer.py`)
