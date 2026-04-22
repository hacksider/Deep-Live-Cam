# Review TODOs — Apple Silicon + Windows CUDA Perf Commit

Post-merge review findings for commit `f65aeae` ("Apple Silicon + Windows CUDA
perf: 60 FPS pipeline, cross-platform routing"). Findings come from two
independent code reviews: Claude (in-tree read) and Codex (second opinion).
Severity reflects production impact, not difficulty to fix.

## Blockers

### CUDA-graph replay buffers not locked — `modules/processors/frame/face_swapper.py:232-238`
*Source: Claude + Codex (independent convergence)*

`_cuda_graph_swap_inference` mutates module-level `ort_input` / `ort_latent`
and runs `run_with_iobinding` with no lock. `multi_process_frame` runs frame
work concurrently, so two swap calls can overwrite the same bound input
buffers before `run_with_iobinding`, producing wrong-face output or
corrupted frames. Compare the DML path at `face_swapper.py:382-386` which
uses `modules.globals.dml_lock` for the same reason.

**Fix:** a dedicated `_cuda_graph_lock` around the full
update-run-get sequence inside `_cuda_graph_swap_inference`.

## Should-fix

### `many_faces` enhancer loop breaks after face #1 — `modules/processors/frame/face_enhancer.py:375`
*Source: Codex*

The `break` at line 375 is unconditional, so both the fresh-enhance and
cache-reuse paths exit the face loop after the first face. In live
`many_faces=True` mode, GFPGAN silently enhances only one face.

**Fix:** gate the `break` on `not modules.globals.many_faces`, and disable
the single-slot temporal cache in many-faces mode (cache would be
overwritten per face, pasting the wrong enhancement).

### `poisson_blend` operates on post-swap frame — `modules/processors/frame/face_swapper.py:437`
*Source: Claude*

`create_face_mask(target_face, temp_frame)` is called with `temp_frame`,
but `_fast_paste_back` wrote in-place into `temp_frame` a few lines earlier
(line 403). The mouth-mask path at line 414 correctly uses
`original_frame` — Poisson should do the same.

**Fix:** pass `original_frame` to `create_face_mask` on the Poisson path.

### Shape/Gather fold crashes on vector indices — `modules/onnx_optimize.py:150-152`
*Source: Codex*

`int(inits[idx_name])` assumes the Gather index is scalar. Models that
gather multiple dims at once pass a vector index — `int()` on a
multi-element numpy array raises `TypeError`, aborting the whole
optimization pass (no try/except around this section).

**Fix:** check `inits[idx_name].ndim == 0` or `.size == 1` before folding;
skip vector gathers (or fold to a vector constant initializer).

### Reflect-pad decompose silently wrong for asymmetric pads — `modules/onnx_optimize.py:253`
*Source: Codex*

Only reads `pads[2]` and `pads[3]` (H-start, W-start); ignores `pads[6]`
and `pads[7]` (H-end, W-end). Decomposition assumes start == end. Fine
for inswapper_128 (symmetric `[0,0,3,3,0,0,3,3]`) but silently produces
wrong output shape for any future asymmetric reflect pad.

**Fix:** read all four pad values and generate top/bottom/left/right
slice ranges separately. Or assert symmetry and skip otherwise.

### `FACE_DETECTION_CACHE` data race — `modules/processors/frame/face_swapper.py:476-506`
*Source: Claude*

`get_faces_optimized` reads and writes `FACE_DETECTION_CACHE` /
`LAST_DETECTION_TIME` module globals from multiple frame threads without
any lock. Benign in practice (worst case: a duplicate detection or a
stale read) but worth a lock wrap for hygiene.

**Fix:** wrap read-modify-write in `THREAD_LOCK`.

## Consider

### Split decompose misses opset-13+ input form — `modules/onnx_optimize.py:346-357`
*Source: Codex*

Only reads the legacy `split` attribute. Opset 13+ passes split sizes as
`input[1]`; those Split nodes stay on CPU. Depends on how GFPGAN was
exported — verify against `gfpgan-1024.onnx` as actually shipped.

**Fix:** additionally check `node.input[1]` against initializers for
newer opsets.

### `_preserve_emap_position` matches by shape, not name — `modules/onnx_optimize.py:408-423`
*Source: Claude*

Selects "first 512×512 initializer" as the emap. If insightface ever
adds another 512×512 initializer before emap, we'd misplace the tensor.

**Fix:** key on initializer name (insightface serializes it as `emap`
in the proto).

### One-frame detection lag + misleading comment — `modules/processors/frame/core.py:351-361`
*Source: Codex*

Pipelined detection result applied to the current frame is actually from
the previous frame. The inline comment "Get the detection result for
THIS frame" contradicts the later comment "the result will be used for
the next iteration." Documented latency-for-throughput trade, but the
first comment is wrong. Visible as a quality regression on fast motion
/ scene cuts.

**Fix:** correct the comment. Optionally add a config flag to disable
pipelining for high-motion footage.

### Monkey-patching `swapper.session.run` is fragile — `modules/processors/frame/face_swapper.py:223`
*Source: Claude*

`swapper.session.run = _graph_run` replaces the method. If insightface
rebuilds or swaps the session (e.g., on reconfigure), the patch is
silently lost and we fall back to the standard path without warning.

**Fix:** wrap the call site instead of monkey-patching the session,
or assert the patch survives at key lifecycle points.

### `_fast_paste_back` accepts unused `aimg` parameter — `modules/processors/frame/face_swapper.py:241`
*Source: Claude*

Caller at line 401-403 allocates a `_aimg_dummy` solely to satisfy the
signature. Only `aimg.shape` is used.

**Fix:** signature `(target_img, bgr_fake, face_h, face_w, M)`.

### `onnxruntime.get_available_providers()` called at import time — `modules/platform_info.py:33`
*Source: Claude*

Runs before any Windows CUDA DLL path setup from `run.py` takes effect,
unless `platform_info` is imported after that setup. Verify import
order; otherwise CUDA provider may fail to register.

**Fix:** lazy-evaluate on first use rather than at module load, or
confirm `run.py` imports `platform_info` only after DLL-path shim.

---

## ORT 1.26 cleanup

When ORT floor >= 1.26.0, delete `_decompose_reflect_pad` (pass 2) in
`modules/onnx_optimize.py` — fixed upstream by
[microsoft/onnxruntime#28073](https://github.com/microsoft/onnxruntime/pull/28073).
See the `TODO(ort>=1.26)` markers in the file.
