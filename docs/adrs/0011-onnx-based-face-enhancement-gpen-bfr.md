# ADR 0011: ONNX-Based Face Enhancement with GPEN-BFR Models

## Status
**Accepted** (Feb 2026)

## Context

The existing face enhancer uses GFPGAN v1.4, a PyTorch model running on MPS (macOS) or CUDA. While effective, this creates several issues:

1. **Dual-runtime overhead**: PyTorch is loaded solely for face enhancement, adding ~600 MB of dependencies and 2-3s startup time (see [ADR 0006](0006-dual-runtime-pytorch-onnx-separation.md))
2. **No Neural Engine utilization**: PyTorch/MPS uses only the GPU, missing Apple's Neural Engine which is idle during enhancement
3. **No MLX alternative**: Research confirmed no MLX face restoration models exist as of Feb 2026
4. **Performance ceiling**: GFPGAN at 512×512 is the only option — no lower-resolution fast path

### Options Evaluated

| Option | Runtime | Neural Engine | Resolution | Status |
|--------|---------|---------------|------------|--------|
| GFPGAN (current) | PyTorch/MPS | No | 512×512 | In production |
| GPEN-BFR ONNX | ONNX/CoreML EP | Yes | 256 or 512 | **Selected** |
| MLX restoration | MLX/Metal | No | — | No models exist |
| CodeFormer ONNX | ONNX/CoreML EP | Yes | 512×512 | Viable future option |

## Decision

Add GPEN-BFR-256 and GPEN-BFR-512 as alternative face enhancer plugins alongside the existing GFPGAN enhancer. All three coexist — the user selects via CLI flag or UI toggle.

### Architecture

```
modules/processors/frame/
├── _onnx_enhancer.py          # Shared: session creation, affine warp, pre/post processing
├── face_enhancer.py           # GFPGAN (PyTorch) — unchanged
├── face_enhancer_gpen256.py   # GPEN-BFR-256 (ONNX, 256×256)
└── face_enhancer_gpen512.py   # GPEN-BFR-512 (ONNX, 512×512)
```

### Key Design Choices

1. **Shared `_onnx_enhancer.py` module**: CoreML EP configuration, affine face alignment, and NCHW pre/post processing extracted into a reusable module. Adding future ONNX enhancers (e.g., CodeFormer) requires only a thin wrapper.

2. **Affine-aligned face warping**: Uses InsightFace 5-point landmarks to warp faces into FFHQ-aligned canonical position before inference, then inverse-warps with feathered blending. This matches the alignment GPEN was trained on.

3. **CoreML EP with same config as face_swapper**: Reuses the proven `MLComputeUnits: ALL` configuration (Neural Engine + GPU + CPU) from `face_swapper.py`, including model caching and static shapes.

4. **Coexistence, not replacement**: GFPGAN remains available. Users on NVIDIA/CUDA may prefer it. The GPEN alternatives primarily benefit macOS users via CoreML EP.

### Model Sources

- GPEN-BFR-256.onnx (~76 MB) — from [harisreedhar/Face-Upscalers-ONNX](https://github.com/harisreedhar/Face-Upscalers-ONNX/releases)
- GPEN-BFR-512.onnx (~285 MB) — same source
- Auto-downloaded via `conditional_download()` on first use

## Consequences

### Positive
- **Neural Engine utilization**: CoreML EP routes enhancement to Neural Engine, freeing GPU for other work
- **Resolution flexibility**: GPEN-256 provides ~4× faster enhancement for latency-sensitive live mode
- **Reduced dependency path**: ONNX enhancers don't require PyTorch, GFPGAN, or BasicSR
- **Consistent runtime**: Same ONNX Runtime used for swap and enhancement reduces memory footprint
- **Future extensibility**: `_onnx_enhancer.py` makes adding CodeFormer or other ONNX models trivial

### Negative
- **Additional model downloads**: Two new model files (76 MB + 285 MB)
- **Untested quality parity**: GPEN-BFR quality vs GFPGAN not yet benchmarked on this pipeline
- **Affine alignment assumptions**: 5-point landmark quality affects enhancement — poor detection degrades results
- **Three enhancer options**: More choices for users to understand

### Mitigations
- Models downloaded lazily on first use, not at install time
- UI toggle makes selection straightforward
- Default remains GFPGAN — GPEN variants are opt-in
- Tests verify passthrough behavior when no face detected

## Evidence

### Model Availability
- GPEN-BFR ONNX models widely used in FaceFusion, ReActor, and Rope ecosystems
- Pre-converted ONNX files available on HuggingFace and GitHub (multiple mirrors)

### Expected Performance (to be validated)
| Enhancer | Resolution | Expected Time | Neural Engine |
|----------|-----------|---------------|---------------|
| GFPGAN | 512×512 | ~25ms (MPS) | No |
| GPEN-256 | 256×256 | ~8-12ms (CoreML) | Yes |
| GPEN-512 | 512×512 | ~18-25ms (CoreML) | Yes |

## Related Decisions
- [ADR 0002: Plugin Architecture](0002-plugin-architecture-for-frame-processors.md) — GPEN enhancers follow the same plugin pattern
- [ADR 0004: Platform-specific GPU Runtimes](0004-platform-specific-gpu-runtime-selection.md) — CoreML EP selection
- [ADR 0006: Dual-Runtime Separation](0006-dual-runtime-pytorch-onnx-separation.md) — This ADR partially addresses the dual-runtime negative by offering ONNX-only enhancement

## Future Improvements
- Benchmark GPEN vs GFPGAN quality on standard test faces
- Add CodeFormer ONNX as a fourth enhancer option
- Auto-select enhancer based on platform (GPEN on macOS, GFPGAN on CUDA)
- Explore combining GPEN-256 for live mode with GPEN-512 for video export

**Last Reviewed**: Feb 21, 2026 | **Confidence**: Medium (quality benchmarks pending)
