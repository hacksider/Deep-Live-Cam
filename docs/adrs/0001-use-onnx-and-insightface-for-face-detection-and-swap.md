# ADR 0001: Use ONNX and InsightFace for Face Detection and Swapping

## Status
**Accepted** (Established since Sep 2023, reaffirmed v2.0.3c)

## Context

Deep-Live-Cam requires real-time face detection and swapping capability. The initial design decision (Sep 2023) chose:
- **ONNX** for face swapping inference (lightweight, cross-platform)
- **InsightFace** for face detection and embedding extraction
- **inswapper_128_fp16** model for face replacement

### Why This Matters

Face detection and swapping are the core of the application. Early architectural choice impacts:
- Cross-platform support (desktop and edge devices)
- Model download size and startup time
- Inference latency (FPS performance)
- CPU/GPU flexibility

### Alternatives Considered

1. **PyTorch Directly** - Loading PyTorch models directly
   - ✗ Larger dependency footprint
   - ✗ Slower startup (full framework load)
   - ✓ Easier custom model integration

2. **TensorFlow/TFLite** - Using TensorFlow for all inference
   - ✗ Heavier than ONNX
   - ✗ More complex mobile deployment
   - ✓ Better mobile optimization

3. **OpenCV DNN Module** - Using OpenCV's built-in neural network support
   - ✗ Limited model support
   - ✗ Weaker performance on complex models
   - ✓ Zero external dependencies

## Decision

Use **ONNX Runtime** as the primary inference engine with **InsightFace** for face detection and analysis:

1. **ONNX for face swapping** (inswapper_128_fp16 model)
   - Lightweight, portable model format
   - Cross-platform inference (Windows, macOS, Linux)
   - Platform-specific optimizations available (onnxruntime-silicon, onnxruntime-gpu, onnxruntime-rocm)

2. **InsightFace for face detection**
   - Buffalo_l model: High accuracy multi-face detection
   - Embedding extraction for face similarity matching
   - Industry-standard face recognition pipeline

3. **Dual-runtime approach**
   - ONNX for fast face swapping
   - PyTorch (via GFPGAN) for optional face enhancement (separate concern)

## Consequences

### Positive
✓ **Cross-platform**: ONNX runs efficiently on CPU/GPU across Windows/macOS/Linux
✓ **Lightweight**: Models download at startup (~750 MB total for swap + detection)
✓ **Performance**: 30-60 FPS on modern GPUs (NVIDIA, Apple Silicon, AMD)
✓ **Flexibility**: InsightFace embeddings enable face mapping (multi-person swap)
✓ **Proven**: Production-quality models (inswapper used in many deepfake tools)
✓ **Extensibility**: Plugin system allows custom ONNX models to swap core models

### Negative
✗ **Model Expertise Required**: Understanding ONNX model conversion, quantization
✗ **GPU Fragmentation**: Different ONNX runtimes for different platforms (onnxruntime-silicon vs -gpu)
✗ **Limited Customization**: Harder to fine-tune models compared to direct PyTorch access
✗ **Version Pinning**: InsightFace (0.7.3) pinned to specific version; newer versions might break compatibility

### Mitigations
- Model fallback mechanism (v2.x): If primary model unavailable, use alternative
- Platform-specific optimizations (v1.x+): Each platform gets tuned ONNX runtime
- Clear documentation on model switching/swapping

## Evidence

### Git History
- **Initial choice** (Sep 24, 2023): f522c4e (first commit uses ONNX)
- **Model solidified** (Oct 2023): 1671247 (switched to inswapper_128_fp16)
- **Verified production** (Feb 2026): 508 commits using ONNX, no major alternatives explored

### Performance Benchmarks
- **Face Swap**: 15-20ms per face (NVIDIA RTX 3060)
- **Face Detection**: <10ms per frame (buffalo_l + InsightFace)
- **Combined FPS**: 30-60 FPS at 1080p depending on GPU

### Deployment
- All releases (v0.x through v2.0.3c) use ONNX + InsightFace
- 11K+ GitHub stars indicate production stability and user confidence
- Community forks and ports confirm cross-platform reliability

## Related Decisions
- [ADR 0002: Plugin Architecture for Frame Processors](0002-plugin-architecture-for-frame-processors.md)
- [ADR 0004: Platform-Specific GPU Runtime Selection](0004-platform-specific-gpu-runtime-selection.md)
- [ADR 0007: Dual-Runtime Approach (ONNX + PyTorch)](0007-dual-runtime-approach-onnx-pytorch.md)

## Notes
- InsightFace version 0.7.3 is pinned; upgrading requires testing against new model formats
- ONNX model quantization (fp16 vs int8) affects speed vs quality tradeoff
- Face embedding dimension (512-D) constrains downstream similarity matching

**Last Reviewed**: Feb 18, 2026 | **Decision Maker**: Initial architects (Sep 2023)
