# Anti-Pattern Analysis: Deep-Live-Cam

> Generated: 2026-02-18
> Analysis scope: Full codebase (Python patterns, complexity, security)

---

## Critical

| # | Location | Issue |
|---|----------|-------|
| 1 | `modules/utilities.py` | SSL verification bypassed (`verify=False`) in HTTP requests — exposes model downloads to MITM attacks |
| 2 | `modules/ui.py:create_root` | Function exceeds ~400 lines; single function owns entire GUI construction and event wiring |
| 3 | `modules/processors/frame/face_swapper.py` | 7-level nesting depth in frame processing loop — control flow is untestable and fragile |
| 4 | `modules/ui.py` | Module is ~1400 lines — violates single-responsibility; UI layout, camera loop, state management all co-located |
| 5 | `modules/core.py` | Module is ~1219 lines — argument parsing, pre-checks, headless pipeline, and GUI launch all in one file |

---

## High

| # | Location | Issue |
|---|----------|-------|
| 1 | `modules/processors/frame/face_masking.py:493` | Bare `except: pass` silently swallows exceptions — processing failures are invisible |
| 2 | `modules/utilities.py:37` | `except Exception: pass` on file I/O — missing/corrupt state file silently ignored |
| 3 | `modules/utilities.py:59` | `except Exception: pass` on JSON parse — malformed state silently resets all settings |
| 4 | `modules/utilities.py` | No checksum validation on downloaded model files — corrupt or tampered models load silently |
| 5 | `modules/processors/frame/face_swapper.py` | `process_frame` function >120 lines; detection, validation, swap, masking, and enhancement mixed together |
| 6 | `modules/face_analyser.py` | `get_one_face` / `get_many_faces` duplicate detection logic with no shared helper |
| 7 | `modules/ui.py` | Camera preview loop runs on main thread — blocks UI event processing under load |

---

## Medium

| # | Location | Issue |
|---|----------|-------|
| 1 | `modules/globals.py` | All runtime state is mutable module-level variables — no locking, race conditions possible under `ThreadPoolExecutor` |
| 2 | `modules/globals.py` | `source_target_map` written from UI thread, read from worker threads — unsynchronised shared state |
| 3 | `modules/utilities.py` | State file written to current working directory (`./`) — path is implicit and environment-dependent |
| 4 | `modules/processors/frame/face_swapper.py` | Magic numbers for face confidence threshold (e.g. `0.5`, `0.35`) — no named constants or CLI exposure |
| 5 | `modules/processors/frame/face_enhancer.py` | GFPGAN upscale factor hardcoded as `2` — not configurable |
| 6 | `modules/ui.py` | Preview frame resize target hardcoded as pixel literals — breaks on HiDPI displays |
| 7 | `modules/processors/frame/face_masking.py` | Mouth region feather radius is a magic number; unrelated to face landmark scale |
| 8 | `modules/core.py` | `create_temp` / `clean_temp` called unconditionally even in headless mode when no temp dir is needed |

---

## Low

| # | Location | Issue |
|---|----------|-------|
| 1 | `modules/processors/frame/face_enhancer.py` | GPU fallback to CPU is silent — no log line when device selection changes at runtime |
| 2 | `modules/ui.py` | FPS cap `30` appears as an integer literal in multiple places — should be a named constant |
| 3 | `modules/face_analyser.py` | Detection cache size (`128`) is unnamed — should be `DETECTION_CACHE_SIZE` constant |

---

## Top Recommended Fixes

Priority order based on security risk, crash potential, and maintenance impact:

1. **[Critical/Security] Enable SSL verification** (`utilities.py`)
   Replace `verify=False` with `verify=True` (default). If a custom CA is needed, pass `verify=<ca_bundle_path>`. This is a one-line fix with zero functional impact on normal networks.

2. **[Critical/Security] Add model checksum validation** (`utilities.py:conditional_download`)
   After download, verify SHA-256 against a manifest. Prevents loading tampered or truncated ONNX/GFPGAN weights.

3. **[High] Replace silent exception handlers** (`face_masking.py:493`, `utilities.py:37,59`)
   Log and re-raise or return a sentinel value. Silent `pass` blocks hide root causes during debugging.

4. **[Critical] Extract face_swapper processing into sub-functions**
   Split `process_frame` into `_detect_faces`, `_validate_embeddings`, `_run_swap`, `_apply_mask`. Reduces nesting from 7 to ≤3 levels and makes each step independently testable.

5. **[Medium] Add `threading.Lock` guards around `globals.source_target_map`**
   A `RLock` acquired in the UI write path and all worker read paths eliminates the race condition without a larger refactor.

6. **[High] Move `create_root` into a builder module** (`ui.py`)
   Extract widget construction into `modules/ui_builder.py`, leaving `ui.py` as the orchestrator. Target: no function >80 lines.

7. **[Medium] Replace magic threshold constants with named symbols**
   Define `FACE_CONFIDENCE_THRESHOLD = 0.5` and `MOUTH_FEATHER_RADIUS = 10` in `globals.py` or a `constants.py` module. Expose performance-sensitive ones via CLI flags.

8. **[Low] Add a `logging.info` when GPU→CPU fallback occurs** (`face_enhancer.py`)
   Users running on Apple Silicon frequently report unexpected CPU usage; a single log line would surface this immediately.
