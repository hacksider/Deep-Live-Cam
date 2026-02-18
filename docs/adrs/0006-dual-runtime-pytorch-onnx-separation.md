# ADR 0006: Dual-Runtime Approach (PyTorch for Enhancement, ONNX for Swap)

## Status
**Accepted** (Evolved 2024-2025; solidified in v2.0+)

## Context

Deep-Live-Cam performs two distinct inference operations:
1. **Face swapping** (fast, lightweight): inswapper_128_fp16 (379 MB)
2. **Face enhancement** (quality-focused, heavier): GFPGAN + BasicSR (PyTorch models, ~600 MB)

Design could use single runtime for both, but face swapping has different optimization requirements than enhancement.

### Optimization Goals
- **Face swap**: Minimize latency (<20ms per face) for 30-60 FPS real-time
- **Face enhance**: Maximize quality (remove artifacts, restore detail) even if slower

## Decision

Use separate runtimes optimized for each task:

1. **ONNX for face swapping** (inswapper_128_fp16)
   - Lightweight inference, minimal CPU overhead
   - Runs on CPU or GPU with ONNX Runtime
   - Cross-platform optimization (onnxruntime-silicon on ARM Mac, etc.)

2. **PyTorch for face enhancement** (GFPGAN + BasicSR)
   - TencentARC/GFPGAN: GAN-based enhancement
   - Built on BasicSR framework (PyTorch)
   - Quality-focused: artifact removal, detail restoration
   - Can disable (toggle in UI) for speed critical scenarios

### Processor Pipeline
```
Input Frame
    ↓
[Face Swapping - ONNX] (required, always active)
    ↓
[Face Enhancement - PyTorch] (optional, toggle via --frame-processor face_enhancer)
    ↓
[Mouth Masking - NumPy/OpenCV] (optional, toggle via --mouth-mask)
    ↓
Output Frame
```

## Consequences

### Positive
✓ **Optimized per task**: ONNX lightweight for swap; PyTorch for quality enhancement
✓ **Performance flexibility**: Users disable enhancement on weak GPUs
✓ **Quality focused**: GFPGAN/BasicSR provide best-in-class face restoration
✓ **Proven combination**: TencentARC models widely used in industry
✓ **Modular**: Enhancement can be swapped for alternative GAN-based models
✓ **Clear separation**: Concerns clearly separated (swap vs enhance)

### Negative
✗ **Dependency duplication**: Both ONNX and PyTorch in dependency tree
✗ **Memory footprint**: Two runtime engines in memory simultaneously
✗ **Model management**: Download/cache 3+ model files (inswapper, GFPGAN, BasicSR)
✗ **Version compatibility**: PyTorch, GFPGAN, BasicSR versions must align
✗ **Complexity**: Understanding two inference frameworks required for contributions
✗ **Startup time**: Loading both runtimes adds 2-3 seconds to startup

### Mitigations
- **Lazy loading**: Models downloaded on first use (not at startup)
- **Enhancement toggleable**: Users can disable enhancement for speed
- **Clear documentation**: pyproject.toml notes git dependencies for GFPGAN/BasicSR
- **Version pinning**: pyproject.toml pins compatible versions

## Evidence

### Git History
- **Initial ONNX swap** (Sep 2023): f522c4e, 1671247
- **GFPGAN enhancement added** (2024): Commit logs show face_enhancer.py introduction
- **Version compatibility fixes** (2024-2025): Multiple pytorch, gfpgan version updates indicate tuning
- **Solidified** (v2.x): Pattern unchanged through 182 commits in 2025

### Performance Trade-offs
| Operation | Runtime | Time | Quality |
|-----------|---------|------|---------|
| Face Swap | ONNX | ~15-20ms | Good (inswapper specialized) |
| Enhancement | PyTorch | ~20-30ms (optional) | Excellent (GAN-based) |
| Combined | ONNX + PyTorch | ~40-50ms | Best (both applied) |

### Architectural Benefits
- Face enhancement toggleable: `--frame-processor face_enhancer` flag
- UI checkbox to disable enhancement on weak GPUs
- Reduces FPS impact when enhancement not needed
- Clear separation of concerns in codebase

## Related Decisions
- [ADR 0001: ONNX/InsightFace](0001-use-onnx-and-insightface-for-face-detection-and-swap.md)
- [ADR 0002: Plugin Architecture](0002-plugin-architecture-for-frame-processors.md) (enhancement as plugin)
- [ADR 0004: Platform-specific GPU Runtimes](0004-platform-specific-gpu-runtime-selection.md)

## Future Improvements
- Single-runtime approach using ONNX for both swap and enhancement
- Custom ONNX enhancement model (vs GFPGAN) for reduced footprint
- Real-time quality profiling (auto-disable enhancement if FPS drops)
- Quantization strategy per runtime (int8 for PyTorch, fp16 for ONNX)

**Last Reviewed**: Feb 18, 2026 | **Confidence**: High
