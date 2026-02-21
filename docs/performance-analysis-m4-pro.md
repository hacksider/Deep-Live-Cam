# Deep-Live-Cam Performance Analysis: M4 Pro

**Date**: 2026-02-20
**Hardware**: MacBook M4 Pro (12 cores, 20-core GPU, 38 TOPS Neural Engine)

## The Core Problem: M4 Pro is Severely Underutilized

The M4 Pro has 38 TOPS Neural Engine, a 20-core GPU with 273 GB/s bandwidth, and 12 CPU cores. Deep-Live-Cam is effectively running CPU-only inference. The ANE and GPU sit idle during the heavy compute work.

This explains the 1-5 FPS users report on Apple Silicon — ~80% of the chip's ML capability is unused.

## Why Hardware is Underutilized

| Hardware Unit | Status | Why |
|---|---|---|
| **Neural Engine (38 TOPS)** | Idle | CoreML EP falls back to CPU for dynamic-shape ONNX models |
| **GPU (20-core)** | Idle (inference) | No Metal backend in OpenCV; `cv2.cuda` is NVIDIA-only |
| **CPU (12 cores)** | Underutilized | Single processing thread in live mode; only 1 core doing inference |
| **Unified Memory (273 GB/s)** | Wasted | No zero-copy GPU/ANE pipeline; everything copies through CPU numpy arrays |

## Top Bottlenecks by Impact

### 1. Single Processing Thread in Live Mode — 2-4x FPS gain possible

`ui_webcam.py:186-190` — All inference (detect + swap + enhance) runs in ONE thread. On a 12-core M4 Pro, detection (CPU-bound via InsightFace) and swap (should be ANE-bound via CoreML) could overlap in separate threads since both release the GIL.

### 2. CoreML EP Not Actually Reaching ANE — 2-5x inference speedup possible

`face_swapper.py:93-106` — `RequireStaticShapes: 0` forces dynamic shape handling, likely causing CPU fallback. The inswapper model always uses fixed 128x128 input. Setting `RequireStaticShapes: 1` and verifying `ModelFormat: MLProgram` could enable actual ANE dispatch.

### 3. InsightFace Pinned to CPU — face detection is the single biggest time cost

`face_analyser.py:37-46` — InsightFace always uses `CPUExecutionProvider` (CoreML excluded due to dynamic output shapes). Face detection at 320x320 costs 30-80ms per frame on CPU alone.

### 4. GFPGAN Serial + Double Detection — 1.5-2x when enhancer enabled

`face_enhancer.py:23` — Semaphore(1) makes enhancement fully serial. Additionally, GFPGAN runs its own internal face detection, duplicating the InsightFace detection already performed by the swapper.

### 5. `get_face_swapper()` Lock Contention — ~5% per-frame overhead

`face_swapper.py:81-84` — Acquires `THREAD_LOCK` on every frame even after initialization. In video mode with 8+ threads, this is a hot contention point. Needs double-check locking pattern.

### 6. PNG Intermediate Format (Video Mode) — 30-50% I/O reduction

`processors/frame/core.py:82-83` — Each frame written/read as PNG. Even at compression level 3, PNG is 3-5x slower than JPEG or BMP for intermediates.

## ONNX Runtime CoreML Configuration Issues

### Current Configuration (`face_swapper.py:93-106`)

```python
providers=[("CoreMLExecutionProvider", {
    "MLComputeUnits": "ALL",
    "SpecializationStrategy": "FastPrediction",
    "RequireStaticShapes": 0,     # BUG: should be 1 for fixed 128x128 input
    "MaximumCacheSize": 512,
})]
```

### Recommended Configuration

```python
providers=[("CoreMLExecutionProvider", {
    "MLComputeUnits": "CPUAndNeuralEngine",
    "ModelFormat": "MLProgram",        # Prevents silent FP16 cast, better op coverage
    "RequireStaticShapes": 1,          # Fixed 128x128 input — skip dynamic shape checks
    "SpecializationStrategy": "FastPrediction",
    "MaximumCacheSize": 512,
    "ModelCacheDirectory": "models/cache/",  # Avoid recompilation on restart
})]
```

### Silent FP16 Conversion

When CoreML EP uses the legacy NeuralNetwork format (default), it silently casts models to FP16, even FP32 models. Using `ModelFormat: MLProgram` gives explicit type handling and better operator coverage (~50 ops vs ~35).

### Model Caching

Without `ModelCacheDirectory`, CoreML recompiles the model on every session creation — adding seconds of startup overhead per model load.

## NumPy / BLAS Performance

When installed via pip/uv, NumPy uses OpenBLAS, **not** Apple Accelerate. To get Accelerate BLAS (up to 10x speedup for some linear algebra ops), you must build NumPy from source or use conda with `libblas=*=*accelerate`.

Practical impact: NumPy ops in face processing (embedding normalization, affine transforms) use OpenBLAS. Minor bottleneck compared to model inference, but non-trivial for high-frequency operations.

Reference: [uv issue #13103](https://github.com/astral-sh/uv/issues/13103)

## Memory Allocation in Hot Paths

| Location | Issue | Fix |
|---|---|---|
| `face_swapper.py:297` | `apply_post_processing()` copies frame unconditionally | Only copy if sharpening is enabled |
| `face_swapper.py:234` | `np.clip(...).astype(np.uint8)` allocates new array per frame | Unavoidable post-ONNX, but could use `np.clip(..., out=...)` |
| `face_swapper.py:246-254` | `_apply_mouth_mask`/`_apply_poisson_blend` called unconditionally | Already short-circuit when disabled, but attribute lookups repeat |

## GIL Analysis

The GIL impact is minimal for this workload because:
- ONNX Runtime releases the GIL during `session.run()`
- InsightFace's C++ backend releases GIL during inference
- OpenCV releases GIL for most image operations
- Pure Python bytecode between calls is microseconds, not milliseconds

The architectural problem is not the GIL — it's that live mode uses a **single thread** for all processing, so even with GIL release, there's no concurrent work to overlap with.

## Rust Rewrite Assessment: Not Worth It

A full Python-to-Rust rewrite would yield **<5% FPS improvement** because:

- **Inference is already C++ code** — ONNX Runtime, InsightFace, OpenCV all release the GIL and run natively. Rust would call the exact same shared libraries.
- **Python overhead is ~3-8ms per frame** — orchestration glue, not the bottleneck.
- **Critical ecosystem gap** — No mature Rust equivalent to InsightFace's buffalo_l model bundle (detection + recognition + landmarks). Reimplementing = 4-8 weeks.
- **6-12 months development** for <5% gain is terrible ROI.

### Rust Ecosystem Maturity

| Component | Rust Status | Blocker? |
|---|---|---|
| ONNX Runtime inference | `ort` crate — mature, CoreML support | No |
| InsightFace models | `insightface` crate — immature | **Yes** |
| OpenCV operations | `opencv-rust` — mature | No |
| GUI (video display) | `egui`/`iced` — functional | No |
| Face landmarks, embedding math | Must reimplement manually | **Yes** |

### Hybrid Approach (PyO3)

The only Python code worth porting to Rust via PyO3:
- `apply_mouth_area()` — complex numpy blending (~3ms)
- `create_face_mask()` — convex hull + gaussian blur (~1ms)
- `apply_color_transfer()` — LAB color space conversion (~2ms)

~5ms savings per frame, achievable more easily by using `float32` instead of `float64` in blending and removing unnecessary `.copy()` calls.

## Recommended Improvements

### Quick Wins (hours of work, 10-30% improvement)

| Fix | Location | Estimated Gain |
|---|---|---|
| Set `RequireStaticShapes: 1` for inswapper | `face_swapper.py:97` | Enables CoreML optimization |
| Add warmup inference call after model load | `face_swapper.py:106` | Eliminates first-frame stall |
| Double-check locking on `get_face_swapper()` | `face_swapper.py:81-84` | ~5% per-frame in threaded mode |
| Remove unnecessary `.copy()` in `apply_post_processing()` | `face_swapper.py:297` | ~1ms/frame |
| Set CoreML `ModelCacheDirectory` | Session options | Eliminates recompilation on restart |
| Change display poll from 1ms to 16ms | `ui_webcam.py:201` | Reduces idle CPU usage |

### Medium-Term (1-2 weeks, 2-4x improvement)

| Fix | Estimated Gain |
|---|---|
| Pipeline threading in live mode: separate detection thread + swap thread | 2-4x FPS |
| GFPGAN: pass `has_aligned=True` to skip redundant face detection, increase semaphore count | 1.5-2x with enhancer |
| Use JPEG/BMP intermediates instead of PNG for video processing | 30-50% I/O savings |
| ProcessPoolExecutor for video batch mode with per-process ONNX sessions | 30-50% throughput |

### High-Impact / High-Effort (1-3 months)

| Approach | Potential Gain |
|---|---|
| Direct CoreML conversion via `coremltools` (skip ONNX Runtime entirely) | Could unlock ANE — 2-5x inference speed |
| MLX port of inference pipeline | Native unified memory + ANE, significant speedup |
| NumPy with Apple Accelerate (build from source or conda) | Faster BLAS ops in post-processing |
| PyO3 hybrid for post-processing hot paths | ~5ms/frame savings |

## Theoretical FPS Ceiling

| Configuration | Estimated FPS |
|---|---|
| Current (CPU-only inference, single thread) | 1-5 FPS |
| With quick wins + pipeline threading | 5-12 FPS |
| With direct CoreML/ANE dispatch | 10-20 FPS |
| With MLX + pipeline optimization (no enhancer) | 20-30 FPS |
| Theoretical M4 Pro maximum (swap only, no enhancer) | 30-45 FPS |

The single biggest unlock is getting inference off CPU and onto the Neural Engine. Everything else is secondary.

## References

- [CoreML EP docs](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [ONNX Runtime CoreML issues](https://github.com/microsoft/onnxruntime/issues/9433)
- [Deep-Live-Cam Apple Silicon FPS issue #1273](https://github.com/hacksider/Deep-Live-Cam/issues/1273)
- [MLX framework](https://github.com/ml-explore/mlx)
- [uv Accelerate issue #13103](https://github.com/astral-sh/uv/issues/13103)
- [ort crate (Rust ONNX)](https://github.com/pykeio/ort)
- [FP16 silent conversion analysis](https://ym2132.github.io/ONNX_MLProgram_NN_exploration)
